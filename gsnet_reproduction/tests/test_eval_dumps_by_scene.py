from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from gsnet_reproduction.scripts.eval_dumps_by_scene import (
    aggregate_scene_results,
    compute_graspnet_metrics,
    scene_ids_for_split,
    scene_result_path,
)


class EvalDumpsBySceneTest(unittest.TestCase):
    def test_scene_ids_match_graspnet_test_splits(self) -> None:
        self.assertEqual(scene_ids_for_split("test_seen")[0], 100)
        self.assertEqual(scene_ids_for_split("test_seen")[-1], 129)
        self.assertEqual(scene_ids_for_split("test_similar")[0], 130)
        self.assertEqual(scene_ids_for_split("test_similar")[-1], 159)
        self.assertEqual(scene_ids_for_split("test_novel")[0], 160)
        self.assertEqual(scene_ids_for_split("test_novel")[-1], 189)

        with self.assertRaisesRegex(ValueError, "unsupported split"):
            scene_ids_for_split("train")

    def test_scene_result_path_uses_stable_scene_name(self) -> None:
        self.assertEqual(
            scene_result_path(Path("/tmp/cache"), 107),
            Path("/tmp/cache/scene_0107.npy"),
        )

    def test_compute_graspnet_metrics_reports_percent_values(self) -> None:
        res = np.zeros((1, 1, 1, 6), dtype=np.float32)
        res[0, 0, 0] = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

        metrics = compute_graspnet_metrics(res)

        self.assertAlmostEqual(metrics["AP"], 70.0, places=5)
        self.assertAlmostEqual(metrics["AP0.4"], 40.0, places=5)
        self.assertAlmostEqual(metrics["AP0.8"], 80.0, places=5)

    def test_aggregate_scene_results_keeps_requested_scene_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            np.save(scene_result_path(cache_dir, 102), np.full((2, 3, 6), 102.0))
            np.save(scene_result_path(cache_dir, 100), np.full((2, 3, 6), 100.0))

            aggregated = aggregate_scene_results([100, 102], cache_dir)

            self.assertEqual(aggregated.shape, (2, 2, 3, 6))
            self.assertEqual(float(aggregated[0, 0, 0, 0]), 100.0)
            self.assertEqual(float(aggregated[1, 0, 0, 0]), 102.0)


if __name__ == "__main__":
    unittest.main()
