"""Causal bbox association, ONNX index mapping and migration guards."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from reactpp_models import BBoxTracker, Perception, SETTINGS, decode_predictions, preprocess, require_cuda_provider
from sgg_pipeline import resolved_settings


def detection(box=(0, 0, 10, 10), class_id=1, index=0):
    return {"box": list(box), "class_id": class_id, "label": str(class_id),
            "score": 0.9, "model_detection_index": index}


class TrackerTests(unittest.TestCase):
    def test_ids_survive_movement_and_resume(self):
        tracker = BBoxTracker([], 1)
        first = tracker.update([detection()], 0)
        second = tracker.update([detection((1, 0, 11, 10))], 0.2)
        resumed = BBoxTracker(json.loads(json.dumps(tracker.tracks)), tracker.next_id)
        third = resumed.update([detection((2, 0, 12, 10))], 0.4)
        self.assertEqual([first[0]["track_id"], second[0]["track_id"], third[0]["track_id"]], [1, 1, 1])

    def test_missing_detection_is_not_emitted_as_observed(self):
        tracker = BBoxTracker([], 1)
        tracker.update([detection()], 0)
        self.assertEqual(tracker.update([], 0.2), [])
        self.assertEqual(tracker.update([detection()], 0.4)[0]["track_id"], 1)

    def test_expired_track_gets_new_id(self):
        tracker = BBoxTracker([], 1, max_gap_seconds=1)
        tracker.update([detection()], 0)
        self.assertEqual(tracker.update([detection()], 1.2)[0]["track_id"], 2)

    def test_matching_is_class_aware_and_one_to_one(self):
        tracker = BBoxTracker([], 1)
        tracker.update([detection()], 0)
        rows = tracker.update([detection(), detection(index=1), detection(class_id=2, index=2)], 0.2)
        self.assertEqual(len({r["track_id"] for r in rows}), 3)
        self.assertNotEqual(rows[2]["track_id"], 1)

    def test_future_carry_is_rejected(self):
        tracker = BBoxTracker([], 1)
        tracker.update([detection()], 1)
        with self.assertRaisesRegex(ValueError, "future state"):
            tracker.update([], 0.2)


class ConfigurationTests(unittest.TestCase):
    def test_old_settings_are_rejected_without_modifying_them(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sgg-settings.json"
            old = {"schema_version": 1, "models": {"tracker": {"id": "SAM2"}}}
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "separate target"):
                resolved_settings(Path(folder))
            self.assertEqual(json.loads(path.read_text()), old)

    def test_new_config_is_pinned_and_reusable_without_network(self):
        with tempfile.TemporaryDirectory() as folder:
            first = resolved_settings(Path(folder))
            self.assertEqual(first, resolved_settings(Path(folder)))
            self.assertEqual(first["tracking_fps"], first["graph_fps"])
            self.assertEqual(len(first["models"]["scene_graph"]["revision"]), 40)

    def test_silent_cpu_provider_is_rejected(self):
        class CPU:
            def get_providers(self):
                return ["CPUExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "refusing silent CPU"):
            require_cuda_provider(CPU())


HAS_IMAGES = all(importlib.util.find_spec(name) for name in ("numpy", "cv2"))


@unittest.skipUnless(HAS_IMAGES, "Image dependencies required (Colab requirements-sgg.txt)")
class OutputContractTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.settings = copy.deepcopy(SETTINGS)
        self.objects = ["person", "bowl"]
        self.predicates = ["__background__", "holding"]

    def decode(self, boxes, edges):
        return decode_predictions(self.np.array(boxes).reshape(-1, 6),
                                  self.np.array(edges).reshape(-1, 5),
                                  (640, 640, 3), (1, 0, 0),
                                  self.objects, self.predicates, self.settings)

    def test_detection_sort_does_not_change_relation_endpoints(self):
        boxes = [[0, 0, 100, 100, 1, 0.7], [100, 100, 150, 150, 2, 0.9]]
        detections, edges = self.decode(boxes, [[0, 1, 1, 0.6, 0.95]])
        self.assertEqual([d["model_detection_index"] for d in detections], [1, 0])
        self.assertEqual(edges[0]["subject_index"], 0)
        self.assertEqual(edges[0]["object_index"], 1)
        self.assertEqual(edges[0]["relation"], "holding")
        self.assertEqual(edges[0]["triplet_score"], 0.6)
        self.assertEqual(edges[0]["score"], 0.95)

    def test_filtered_node_removes_its_edges(self):
        boxes = [[0, 0, 100, 100, 1, 0.1], [100, 100, 150, 150, 2, 0.9]]
        detections, edges = self.decode(boxes, [[0, 1, 1, 0.6, 0.95]])
        self.assertEqual(len(detections), 1)
        self.assertEqual(edges, [])

    def test_unknown_classes_and_invalid_endpoints_fail(self):
        with self.assertRaisesRegex(ValueError, "object class ID"):
            self.decode([[0, 0, 10, 10, 999, 0.8]], [])
        with self.assertRaisesRegex(ValueError, "missing source detection"):
            self.decode([[0, 0, 10, 10, 1, 0.8]], [[0, 2, 1, 0.5, 0.8]])

    def test_nonfinite_and_noninteger_indices_fail(self):
        for cls in (float("nan"), 1.5):
            with self.assertRaises(ValueError):
                self.decode([[0, 0, 10, 10, cls, 0.8]], [])

    def test_empty_model_outputs_are_valid(self):
        self.assertEqual(self.decode([], []), ([], []))

    def test_letterbox_color_shape_and_coordinate_inverse(self):
        image = self.np.zeros((240, 640, 3), dtype=self.np.uint8)
        image[:, :, 2] = 255  # BGR red
        tensor, gain, left, top = preprocess(image, 640)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, self.np.float32)
        self.assertEqual((gain, left, top), (1, 0, 200))
        self.assertEqual(tensor[0, 0, 300, 300], 1)
        self.assertEqual(tensor[0, 2, 300, 300], 0)
        self.assertAlmostEqual(float(tensor[0, 0, 0, 0]), 114 / 255)
        detections, _ = decode_predictions(self.np.array([[10, 220, 30, 240, 1, 0.9]]),
                                          self.np.empty((0, 5)), image.shape, (gain, left, top),
                                          self.objects, self.predicates, self.settings)
        self.assertEqual(detections[0]["box"], [10, 20, 30, 40])

    def test_chunk_resume_preserves_ids_and_skips_consumed_boundary(self):
        import cv2
        model = object.__new__(Perception)
        model.settings = copy.deepcopy(SETTINGS)
        calls = []
        def predict(image):
            calls.append(1)
            return ([detection(index=1), detection((20, 0, 30, 10), class_id=2, index=0)],
                    [{"subject_index": 1, "object_index": 0, "relation": "holding",
                      "predicate_id": 1, "score": 0.9, "triplet_score": 0.8}])
        model.predict = predict
        with tempfile.TemporaryDirectory() as folder:
            for i in range(3):
                self.assertTrue(cv2.imwrite(str(Path(folder) / f"{i:06d}.jpg"),
                                            self.np.zeros((40, 40, 3), dtype=self.np.uint8)))
            def times(start):
                return [{"sample_index": i, "time_s": i / 5, "source_frame_index": i * 6}
                        for i in range(start, start + 3)]
            first, carry, next_id = model.chunk(folder, times(0), "helper", [], 1, 1)
            second, _, _ = model.chunk(folder, times(2), "helper", json.loads(json.dumps(carry)), next_id, 1)
        rows = first + second
        self.assertEqual([row["sample_index"] for row in rows], list(range(5)))
        self.assertEqual(len(calls), 5)
        self.assertEqual({row["edges"][0]["subject"] for row in rows}, {"helper:1"})
        self.assertEqual({row["edges"][0]["object"] for row in rows}, {"helper:2"})
        for row in rows:
            ids = {node["id"] for node in row["nodes"]}
            self.assertTrue(all(e["subject"] in ids and e["object"] in ids for e in row["edges"]))


if __name__ == "__main__":
    unittest.main()
