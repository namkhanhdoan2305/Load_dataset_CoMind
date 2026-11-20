"""Frozen perception backends. GPU dependencies are imported only on construction."""

import gc
import inspect
import time
from pathlib import Path


OBJECTS = ["person", "hand", "bowl", "plate", "cup", "glass", "mug", "spoon",
           "fork", "knife", "whisk", "spatula", "pan", "pot", "lid", "bottle",
           "jar", "box", "bag", "cutting board", "kitchen counter", "table",
           "drawer", "cabinet", "sink", "stove", "refrigerator", "egg",
           "vegetable", "fruit"]
PREDICATES = ["on", "inside", "holding", "touching"]


def box_iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def match_detections(detections, tracks, next_id, maximum, threshold=0.3):
    """Greedy one-to-one matching. Ambiguous identities remain a measurable error."""
    candidates = sorted(
        ((box_iou(d["box"], t["box"]), di, ti)
         for di, d in enumerate(detections) for ti, t in enumerate(tracks)
         if d["label"] == t["label"]), reverse=True)
    matched = {}
    used_tracks = set()
    for score, di, ti in candidates:
        if score >= threshold and di not in matched and ti not in used_tracks:
            matched[di] = tracks[ti]["track_id"]
            used_tracks.add(ti)
    available_ids = {track["track_id"] for track in tracks}
    result = []
    for di, detection in enumerate(detections):
        track_id = matched.get(di)
        if track_id is None:
            if len(available_ids) >= maximum:
                continue
            track_id = next_id
            next_id += 1
            available_ids.add(track_id)
        result.append(dict(detection, track_id=track_id))
    return result, next_id


class Perception:
    def __init__(self, settings):
        import torch
        import numpy as np
        from PIL import Image
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        from relsgg import RelateAnything
        from sam2.build_sam import build_sam2_video_predictor

        if not torch.cuda.is_available():
            raise RuntimeError("Choose a GPU runtime in Colab before extracting graphs")
        print(f"Perception GPU: {torch.cuda.get_device_name(0)}", flush=True)
        self.torch, self.np, self.Image = torch, np, Image
        self.settings = settings
        detector = settings["models"]["detector"]
        self.processor = AutoProcessor.from_pretrained(detector["id"], revision=detector["revision"])
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector["id"], revision=detector["revision"]).eval().cpu()
        relation = settings["models"]["relation"]
        self.relation = RelateAnything.from_pretrained(
            relation["id"], revision=relation["revision"], device="cpu",
            predicates=settings["predicates"])
        tracker = settings["models"]["tracker"]
        checkpoint = hf_hub_download(tracker["id"], "sam2.1_hiera_small.pt",
                                     revision=tracker["revision"])
        self.tracker = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_s.yaml", checkpoint,
            device="cpu", apply_postprocessing=False)

    def offload(self, model):
        model.cpu()
        gc.collect()
        self.torch.cuda.empty_cache()

    def detect(self, image):
        inputs = self.processor(images=image, text=". ".join(self.settings["objects"]) + ".",
                                return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            outputs = self.detector(**inputs)
        process = self.processor.post_process_grounded_object_detection
        parameters = inspect.signature(process).parameters
        score_name = "threshold" if "threshold" in parameters else "box_threshold"
        result = process(outputs, inputs.input_ids,
                         **{score_name: self.settings["detection_threshold"]},
                         text_threshold=0.25, target_sizes=[image.size[::-1]])[0]
        names = result.get("text_labels", result.get("labels"))
        detections = [{"box": box.tolist(), "score": float(score),
                       "label": str(label).lower().strip(" .")}
                      for box, score, label in zip(result["boxes"].cpu(),
                                                   result["scores"].cpu(), names)]
        kept = []
        for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
            if not any(box_iou(detection["box"], previous["box"]) > 0.85 for previous in kept):
                kept.append(detection)
        return kept[:self.settings["max_objects"]]

    def chunk(self, frame_folder, times, view, carry, next_id, graph_stride):
        """Detect keyframes, track forward causally, then score relations on them."""
        frames = sorted(Path(frame_folder).glob("*.jpg"))
        keyframes = [i for i, sample in enumerate(times)
                     if sample["sample_index"] % graph_stride == 0 or i == 0]
        detections = {}
        started = time.monotonic()
        print(f"{view}: detector START, {len(keyframes)} keyframes", flush=True)
        self.detector.cuda()
        try:
            for completed, i in enumerate(keyframes, 1):
                with self.Image.open(frames[i]) as image:
                    detections[i] = self.detect(image.convert("RGB"))
                if completed == 1 or completed % 5 == 0 or completed == len(keyframes):
                    print(f"{view}: detector {completed}/{len(keyframes)}, "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        finally:
            self.offload(self.detector)
        print(f"{view}: detector DONE, {time.monotonic() - started:.1f}s", flush=True)

        started = time.monotonic()
        print(f"{view}: SAM2 START, {len(frames)} tracking frames", flush=True)
        self.tracker.cuda()
        graph_nodes = {}
        tracks = [dict(track) for track in carry]
        known = {track["track_id"]: track for track in tracks}
        state = None
        try:
            state = self.tracker.init_state(str(frame_folder), offload_video_to_cpu=True,
                                            offload_state_to_cpu=True)
            print(f"{view}: SAM2 state ready, elapsed={time.monotonic() - started:.1f}s", flush=True)
            for i in range(len(frames)):
                # Carry comes from the exact shared boundary frame of the previous chunk.
                if i == 0:
                    for track in tracks:
                        self.tracker.add_new_points_or_box(
                            state, frame_idx=i, obj_id=track["track_id"],
                            box=self.np.asarray(track["box"], dtype=self.np.float32))
                if i in detections:
                    # Retain all allocated IDs in the chunk to bound SAM memory.
                    references = list(known.values())
                    assigned, next_id = match_detections(
                        detections[i], references, next_id, self.settings["max_objects"])
                    for track in assigned:
                        track["last_detection_s"] = times[i]["time_s"]
                        known[track["track_id"]] = track
                        self.tracker.add_new_points_or_box(
                            state, frame_idx=i, obj_id=track["track_id"],
                            box=self.np.asarray(track["box"], dtype=self.np.float32))
                tracks = []
                if known:
                    prediction = self.tracker.propagate_in_video(
                        state, start_frame_idx=i, max_frame_num_to_track=0)
                    _, ids, mask_logits = next(prediction)
                    prediction.close()
                    masks = (mask_logits[:, 0] > 0).cpu().numpy()
                    for track_id, mask in zip(ids, masks):
                        ys, xs = self.np.nonzero(mask)
                        if len(xs) == 0:
                            continue
                        node = dict(known[track_id])
                        node["box"] = [int(xs.min()), int(ys.min()),
                                       int(xs.max()) + 1, int(ys.max()) + 1]
                        known[track_id] = node
                        tracks.append(node)
                    del mask_logits, masks
                if i in keyframes:
                    graph_nodes[i] = [dict(track) for track in tracks]
                if i == 0 or (i + 1) % 10 == 0 or i == len(frames) - 1:
                    print(f"{view}: SAM2 {i + 1}/{len(frames)}, {len(known)} allocated tracks, "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        finally:
            del state
            self.offload(self.tracker)
        print(f"{view}: SAM2 DONE, {time.monotonic() - started:.1f}s", flush=True)

        rows = []
        started = time.monotonic()
        print(f"{view}: relations START", flush=True)
        self.relation.model.cuda()
        self.relation.device = self.torch.device("cuda")
        try:
            for i in keyframes:
                # The boundary-only frame may not lie on the requested graph grid.
                if times[i]["sample_index"] % graph_stride:
                    continue
                nodes = graph_nodes[i]
                ids = [f"{view}:{node['track_id']}" for node in nodes]
                edges = []
                if len(nodes) >= 2:
                    boxes = self.np.asarray([node["box"] for node in nodes], dtype=self.np.float32)
                    with self.Image.open(frames[i]) as image, self.torch.inference_mode():
                        triplets = self.relation.predict(
                            image.convert("RGB"), boxes,
                            topk=self.settings["max_relations"])
                    edges = [{"subject": ids[t.subject_idx], "relation": t.predicate,
                              "object": ids[t.object_idx], "score": float(t.score)}
                             for t in triplets]
                with self.Image.open(frames[i]) as image:
                    width, height = image.size
                rows.append({"time_s": times[i]["time_s"],
                             "source_frame_index": times[i]["source_frame_index"],
                             "sample_index": times[i]["sample_index"], "view": view,
                             "image_width": width, "image_height": height,
                             "nodes": [dict(node, id=node_id) for node, node_id in zip(nodes, ids)],
                             "edges": edges})
                if len(rows) == 1 or len(rows) % 5 == 0:
                    print(f"{view}: relations {len(rows)} graphs, "
                          f"elapsed={time.monotonic() - started:.1f}s", flush=True)
        finally:
            self.offload(self.relation.model)
            self.relation.device = self.torch.device("cpu")
        print(f"{view}: relations DONE, {len(rows)} graphs, {time.monotonic() - started:.1f}s", flush=True)
        return rows, tracks, next_id
