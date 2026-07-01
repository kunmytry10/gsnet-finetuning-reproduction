from __future__ import annotations

import unittest

from gsnet_reproduction.scripts.run_test_seen_probe import (
    build_scene_frame_pairs,
    compare_checkpoint_summaries,
)


class TestSeenProbeTest(unittest.TestCase):
    def test_build_scene_frame_pairs_keeps_scene_major_order(self) -> None:
        self.assertEqual(
            build_scene_frame_pairs([100, 102], [0, 31]),
            [(100, 0), (100, 31), (102, 0), (102, 31)],
        )

    def test_compare_checkpoint_summaries_reports_metric_delta(self) -> None:
        comparison = compare_checkpoint_summaries(
            official={"metrics_percent": {"AP": 10.0, "AP0.4": 4.0, "AP0.8": 8.0}},
            candidate={"metrics_percent": {"AP": 12.5, "AP0.4": 3.5, "AP0.8": 11.0}},
        )

        self.assertEqual(
            comparison["delta_metrics_percent"],
            {"AP": 2.5, "AP0.4": -0.5, "AP0.8": 3.0},
        )
        self.assertTrue(comparison["ap_improved"])


if __name__ == "__main__":
    unittest.main()
