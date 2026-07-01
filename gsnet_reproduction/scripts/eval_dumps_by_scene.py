#!/usr/bin/env python3
"""Evaluate GSNet dump files scene by scene with resumable result files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GSNET_ROOT = REPO_ROOT / "external" / "graspness_unofficial"
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "graspnet"
DEFAULT_DUMP_DIR = REPO_ROOT / "gsnet_reproduction" / "outputs" / "test_seen_eval"


def scene_ids_for_split(split: str) -> list[int]:
    if split == "test_seen":
        return list(range(100, 130))
    if split == "test_similar":
        return list(range(130, 160))
    if split == "test_novel":
        return list(range(160, 190))
    raise ValueError(f"unsupported split: {split}")


def scene_result_path(cache_dir: Path, scene_id: int) -> Path:
    return Path(cache_dir) / f"scene_{scene_id:04d}.npy"


def aggregate_scene_results(scene_ids: Iterable[int], cache_dir: Path) -> np.ndarray:
    scene_results = []
    for scene_id in scene_ids:
        path = scene_result_path(cache_dir, scene_id)
        if not path.exists():
            raise FileNotFoundError(f"missing scene result: {path}")
        scene_results.append(np.load(path))
    return np.asarray(scene_results)


def compute_graspnet_metrics(res: np.ndarray) -> dict[str, float]:
    if res.ndim != 4 or res.shape[-1] < 4:
        raise ValueError(f"expected AP tensor shape (scene, ann, top_k, friction), got {res.shape}")
    return {
        "AP": float(np.mean(res, dtype=np.float64) * 100.0),
        "AP0.4": float(np.mean(res[:, :, :, 1], dtype=np.float64) * 100.0),
        "AP0.8": float(np.mean(res[:, :, :, 3], dtype=np.float64) * 100.0),
    }


def import_graspnet_eval() -> type:
    sys.path.insert(0, str(GSNET_ROOT))
    from graspnetAPI.graspnet_eval import GraspNetEval

    return GraspNetEval


def evaluate_scene(
    *,
    scene_id: int,
    dataset_root: Path,
    dump_dir: Path,
    camera: str,
    split: str,
    cache_dir: Path,
    overwrite: bool,
) -> Path:
    output_path = scene_result_path(cache_dir, scene_id)
    if output_path.exists() and not overwrite:
        print(f"[skip] scene_{scene_id:04d}: {output_path}", flush=True)
        return output_path

    GraspNetEval = import_graspnet_eval()
    evaluator = GraspNetEval(root=str(dataset_root), camera=camera, split=split)
    start = time.time()
    print(f"[eval] scene_{scene_id:04d} start", flush=True)
    scene_res = np.asarray(evaluator.eval_scene(scene_id, str(dump_dir)))
    np.save(output_path, scene_res)
    elapsed = time.time() - start
    print(
        f"\n[done] scene_{scene_id:04d}: shape={scene_res.shape}, "
        f"mean={np.mean(scene_res) * 100.0:.4f}, sec={elapsed:.1f}, path={output_path}",
        flush=True,
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dump_dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--camera", default="kinect", choices=["kinect", "realsense"])
    parser.add_argument(
        "--split",
        default="test_seen",
        choices=["test_seen", "test_similar", "test_novel"],
    )
    parser.add_argument("--cache_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--scene", type=int, action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    dump_dir = args.dump_dir.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else dump_dir / f"scene_eval_{args.split}_{args.camera}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    scene_ids = args.scene if args.scene is not None else scene_ids_for_split(args.split)
    for scene_id in scene_ids:
        evaluate_scene(
            scene_id=scene_id,
            dataset_root=dataset_root,
            dump_dir=dump_dir,
            camera=args.camera,
            split=args.split,
            cache_dir=cache_dir,
            overwrite=args.overwrite,
        )

    expected_scene_ids = scene_ids_for_split(args.split)
    if all(scene_result_path(cache_dir, scene_id).exists() for scene_id in expected_scene_ids):
        res = aggregate_scene_results(expected_scene_ids, cache_dir)
        metrics = compute_graspnet_metrics(res)
        ap_path = dump_dir / f"ap_{args.camera}.npy"
        summary_path = dump_dir / f"ap_{args.camera}_summary.json"
        np.save(ap_path, res)
        summary = {
            "camera": args.camera,
            "split": args.split,
            "shape": list(res.shape),
            "metrics_percent": metrics,
            "scene_result_dir": str(cache_dir),
            "ap_path": str(ap_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            "\nEvaluation Result:\n----------\n"
            f"{args.camera}, AP {args.split}={metrics['AP'] / 100.0:.12f}\n"
            f"AP={metrics['AP']:.4f}, AP0.8={metrics['AP0.8']:.4f}, AP0.4={metrics['AP0.4']:.4f}\n"
            f"saved: {ap_path}\nsummary: {summary_path}",
            flush=True,
        )
    else:
        missing = [
            scene_id
            for scene_id in expected_scene_ids
            if not scene_result_path(cache_dir, scene_id).exists()
        ]
        print(f"[partial] missing scene results: {missing}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
