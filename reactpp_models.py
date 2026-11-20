"""Frozen REACT++/YOLO ONNX inference and causal, view-local bbox tracking.

ONNX contract follows Maelic/SGG-Benchmark tools/eval_onnx_psg.py and
tools/export_onnx.py at a6642012e66047c3544bed8784bbf9acdc87411b.
GPU/image dependencies are loaded only when needed.
"""

import json
from pathlib import Path
import time

from sgg_models import box_iou


MODEL = {"id": "maelic/REACTPlusPlus_PSG",
         "revision": "25aaf1838c3f6cee32f69a43871bab0870af4bea",
         "filename": "yolo12m/react_pp_yolo12m.onnx",
         "sha256": "38c4bdbb7e2a077131ffcbf57af726eb30574318e2344d5b4bfc1d3a48f5b2fe"}

SETTINGS = {
    "schema_version": 2, "backend": "reactpp_onnx_yolo12m",
    "graph_fps": 5, "tracking_fps": 5, "chunk_seconds": 4,
    "max_side": 960, "input_size": 640, "max_objects": 40, "max_relations": 100,
    "detection_threshold": 0.25, "relation_threshold": 0.1,
    "relation_threshold_basis": "triplet_score",
    "tracker": {"algorithm": "class_aware_greedy_iou_velocity_v1",
                "iou_threshold": 0.3, "max_gap_seconds": 1.0},
    "vocabulary": "ONNX embedded PSG object/relation class names; no manual relabeling",
    "models": {"scene_graph": MODEL},
}


class BBoxTracker:
    """Keep causal detection IDs with a bounded gap; no masks or learned re-ID.

    Only detections observed now become graph nodes. Unmatched tracks are kept
    briefly for re-association, never emitted as observed objects.
    """

    def __init__(self, carry, next_id, iou_threshold=0.3, max_gap_seconds=1.0):
        self.tracks = [dict(track) for track in carry]
        self.next_id = next_id
        self.iou_threshold = iou_threshold
        self.max_gap_seconds = max_gap_seconds

    def update(self, detections, timestamp):
        timestamp = float(timestamp)
        active = []
        for track in self.tracks:
            age = timestamp - track["last_detection_s"]
            if age < 0:
                raise ValueError("Tracker cannot consume future state")
            if age <= self.max_gap_seconds:
                active.append(track)
        candidates = []
        for di, detection in enumerate(detections):
            for ti, track in enumerate(active):
                if detection["class_id"] != track["class_id"]:
                    continue
                dt = timestamp - track["last_detection_s"]
                predicted = [x + v * dt for x, v in zip(track["box"], track["velocity"])]
                score = box_iou(detection["box"], predicted)
                if score >= self.iou_threshold:
                    candidates.append((score, di, ti))
        matched, used = {}, set()
        for _, di, ti in sorted(candidates, reverse=True):
            if di not in matched and ti not in used:
                matched[di] = ti
                used.add(ti)
        visible = []
        for di, detection in enumerate(detections):
            old = active[matched[di]] if di in matched else None
            velocity = [0.0] * 4
            if old is not None:
                dt = timestamp - old["last_detection_s"]
                velocity = list(old["velocity"])
                if dt > 0:
                    velocity = [0.5 * v + 0.5 * (new - prev) / dt
                                for v, new, prev in zip(velocity, detection["box"], old["box"])]
            track_id = old["track_id"] if old is not None else self.next_id
            if old is None:
                self.next_id += 1
            visible.append(dict(detection, track_id=track_id, velocity=velocity,
                                last_detection_s=timestamp))
        self.tracks = visible + [track for ti, track in enumerate(active) if ti not in used]
        return visible


def preprocess(bgr, size):
    """Centered letterbox, BGR→RGB, NCHW float32 /255 (upstream contract)."""
    import cv2
    import numpy as np
    height, width = bgr.shape[:2]
    gain = min(size / height, size / width)
    nw, nh = round(width * gain), round(height * gain)
    left = round((size - nw) / 2 - 0.1)
    right = round((size - nw) / 2 + 0.1)
    top = round((size - nh) / 2 - 0.1)
    bottom = round((size - nh) / 2 + 0.1)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255
    return tensor[None], gain, left, top


def decode_predictions(boxes, relations, image_shape, transform, objects, predicates, settings):
    """Validate and retain indices while selecting boxes/relations; keep source scores."""
    import numpy as np
    boxes, relations = np.asarray(boxes), np.asarray(relations)
    if boxes.ndim != 2 or boxes.shape[1] != 6 or relations.ndim != 2 or relations.shape[1] != 5:
        raise ValueError(f"Unexpected REACT++ output shapes: {boxes.shape}, {relations.shape}")
    if not np.isfinite(boxes).all() or not np.isfinite(relations).all():
        raise ValueError("REACT++ output contains nonfinite values")
    for values in (boxes[:, 4], relations[:, :3].reshape(-1)):
        if not np.equal(values, np.floor(values)).all():
            raise ValueError("REACT++ class and endpoint indices must be integers")
    height, width = image_shape[:2]
    gain, left, top = transform
    detections = []
    for index in np.argsort(-boxes[:, 5], kind="stable"):
        x1, y1, x2, y2, class_id, score = map(float, boxes[index])
        class_id = int(class_id)
        if not 1 <= class_id <= len(objects):
            raise ValueError(f"Unknown REACT++ object class ID {class_id}")
        if score < settings["detection_threshold"]:
            continue
        box = [max(0.0, min(width, (x1 - left) / gain)),
               max(0.0, min(height, (y1 - top) / gain)),
               max(0.0, min(width, (x2 - left) / gain)),
               max(0.0, min(height, (y2 - top) / gain))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append({"box": box, "class_id": class_id, "label": objects[class_id - 1],
                           "score": score, "model_detection_index": int(index)})
        if len(detections) == settings["max_objects"]:
            break
    kept = {d["model_detection_index"] for d in detections}
    edges = []
    for relation in relations:
        subject, obj, predicate = map(int, relation[:3])
        if not 0 <= subject < len(boxes) or not 0 <= obj < len(boxes):
            raise ValueError("REACT++ relation refers to a missing source detection")
        if not 0 <= predicate < len(predicates):
            raise ValueError(f"Unknown REACT++ predicate ID {predicate}")
        if predicate == 0 or subject == obj or subject not in kept or obj not in kept:
            continue
        if relation[3] < settings["relation_threshold"]:
            continue
        edges.append({"subject_index": subject, "object_index": obj,
                      "relation": predicates[predicate], "predicate_id": predicate,
                      "score": float(relation[4]), "triplet_score": float(relation[3])})
    edges.sort(key=lambda e: e["triplet_score"], reverse=True)
    return detections, edges[:settings["max_relations"]]


def require_cuda_provider(session):
    if "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("REACT++ CUDA provider did not initialize; refusing silent CPU inference. "
                           "Install requirements-sgg.txt in a T4 runtime and check CUDA/cuDNN libraries.")


class Perception:
    def __init__(self, settings, model_path=None, require_cuda=True):
        import onnxruntime as ort
        self.settings = settings
        if require_cuda:
            # Prefer the CUDA 12/cuDNN libraries shipped alongside the pinned ORT extra.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls(directory="")
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("ONNX Runtime has no CUDA provider; choose a T4 runtime and install requirements-sgg.txt")
        if model_path is None:
            from huggingface_hub import hf_hub_download
            model = settings["models"]["scene_graph"]
            print(f"Downloading/loading REACT++ YOLO12m ONNX ({model['revision'][:12]})", flush=True)
            model_path = hf_hub_download(model["id"], model["filename"], revision=model["revision"])
        import comind_download as source
        model_sha = source.file_digest(model_path)
        if model_sha != settings["models"]["scene_graph"]["sha256"]:
            raise ValueError("REACT++ checkpoint SHA256 differs from the verified pinned ONNX model")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if require_cuda else ["CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        if require_cuda:
            require_cuda_provider(self.session)
        self.session.disable_fallback()
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4
                or inputs[0].shape[1:] != [3, 640, 640]
                or isinstance(inputs[0].shape[0], int) and inputs[0].shape[0] != 1):
            raise ValueError(f"Unsupported REACT++ ONNX input: {[(i.name, i.shape, i.type) for i in inputs]}")
        if {out.name for out in outputs} != {"boxes", "rels"}:
            raise ValueError(f"Unsupported REACT++ outputs: {[out.name for out in outputs]}")
        self.input_name = inputs[0].name
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "obj_classes" not in metadata or "rel_classes" not in metadata:
            raise ValueError("Pinned REACT++ ONNX is missing its embedded class names; refusing guessed labels")
        self.objects, self.predicates = json.loads(metadata["obj_classes"]), json.loads(metadata["rel_classes"])
        if not isinstance(self.objects, list) or not isinstance(self.predicates, list) or not self.objects or not self.predicates:
            raise ValueError("Invalid REACT++ vocabulary metadata")
        self.info = {"backend": settings["backend"], "model": settings["models"]["scene_graph"],
                     "onnx_sha256": model_sha, "onnxruntime_version": ort.__version__,
                     "providers": self.session.get_providers(), "object_classes": self.objects,
                     "relation_classes": self.predicates,
                     "object_class_index_base": 1, "relation_background_index": 0,
                     "tracker": settings["tracker"], "cross_view_entity_matching": "unresolved"}
        if require_cuda:
            import numpy as np
            started = time.monotonic()
            print("REACT++ CUDA warmup START (one blank image; no graph is saved)", flush=True)
            self.predict(np.full((640, 640, 3), 114, dtype=np.uint8))
            print(f"REACT++ CUDA warmup DONE, {time.monotonic() - started:.1f}s", flush=True)
        print(f"REACT++ ready: providers={self.info['providers']}, "
              f"{len(self.objects)} object classes, {len(self.predicates) - 1} foreground predicates", flush=True)

    def predict(self, bgr):
        tensor, gain, left, top = preprocess(bgr, self.settings["input_size"])
        boxes, relations = self.session.run(["boxes", "rels"], {self.input_name: tensor})
        return decode_predictions(boxes, relations, bgr.shape, (gain, left, top),
                                  self.objects, self.predicates, self.settings)

    def chunk(self, frame_folder, times, view, carry, next_id, graph_stride):
        import cv2
        frames = sorted(Path(frame_folder).glob("*.jpg"))
        if len(frames) != len(times):
            raise ValueError("Decoded frame count disagrees with timestamps")
        tracker = BBoxTracker(carry, next_id, **{k: v for k, v in self.settings["tracker"].items()
                                               if k != "algorithm"})
        rows = []
        started = time.monotonic()
        print(f"{view}: REACT++ START, {len(frames)} sampled frames", flush=True)
        for i, sample in enumerate(times):
            # The previous chunk has already consumed this exact boundary frame.
            if i == 0 and sample["sample_index"] > 0:
                continue
            if sample["sample_index"] % graph_stride:
                continue
            image = cv2.imread(str(frames[i]))
            if image is None:
                raise RuntimeError(f"Could not read decoded frame {frames[i]}")
            detections, raw_edges = self.predict(image)
            tracks = tracker.update(detections, sample["time_s"])
            ids = {t["model_detection_index"]: f"{view}:{t['track_id']}" for t in tracks}
            nodes = [dict(t, id=ids[t["model_detection_index"]]) for t in tracks]
            edges = [dict(subject=ids[e["subject_index"]], object=ids[e["object_index"]],
                          **{k: v for k, v in e.items() if k not in ("subject_index", "object_index")})
                     for e in raw_edges]
            height, width = image.shape[:2]
            rows.append({"time_s": sample["time_s"], "sample_index": sample["sample_index"],
                         "source_frame_index": sample["source_frame_index"], "view": view,
                         "image_width": width, "image_height": height,
                         "backend": self.settings["backend"], "nodes": nodes, "edges": edges})
            if len(rows) == 1 or len(rows) % 10 == 0:
                print(f"{view}: REACT++ {len(rows)} graphs, elapsed={time.monotonic() - started:.1f}s", flush=True)
        return rows, tracker.tracks, tracker.next_id
