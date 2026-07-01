from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_PY = REPO_ROOT / "external" / "graspness_unofficial" / "test.py"
PATCH_FILE = (
    REPO_ROOT
    / "gsnet_reproduction"
    / "patches"
    / "graspness_unofficial_eval_split.patch"
)


class GraspnessTestSplitPatchTest(unittest.TestCase):
    def test_upstream_test_py_accepts_and_uses_eval_split(self) -> None:
        source = TEST_PY.read_text(encoding="utf-8")

        self.assertIn("parser.add_argument('--split'", source)
        self.assertIn("choices=['test_seen', 'test_similar', 'test_novel']", source)
        self.assertIn("GraspNetDataset(cfgs.dataset_root, split=cfgs.split", source)
        self.assertIn(
            "GraspNetEval(root=cfgs.dataset_root, camera=cfgs.camera, split=cfgs.split)",
            source,
        )
        self.assertIn("'test_seen': ge.eval_seen", source)
        self.assertIn("'test_similar': ge.eval_similar", source)
        self.assertIn("'test_novel': ge.eval_novel", source)

    def test_upstream_test_py_supports_utils_package_import(self) -> None:
        source = TEST_PY.read_text(encoding="utf-8")

        self.assertIn("sys.path.append(ROOT_DIR)", source)
        self.assertIn("from collision_detector import ModelFreeCollisionDetector", source)

    def test_eval_split_patch_is_recorded_for_ignored_external_source(self) -> None:
        self.assertTrue(
            PATCH_FILE.exists(),
            f"missing recorded patch file: {PATCH_FILE}",
        )
        patch = PATCH_FILE.read_text(encoding="utf-8")

        self.assertIn("external/graspness_unofficial/test.py", patch)
        self.assertIn("--split", patch)
        self.assertIn("eval_similar", patch)
        self.assertIn("eval_novel", patch)
        self.assertIn("sys.path.append(ROOT_DIR)", patch)


if __name__ == "__main__":
    unittest.main()
