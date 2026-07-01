#!/usr/bin/env python3
"""Compare two checkpoints on a small official GraspNet test_seen subset."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "graspnet"
DEFAULT_OFFICIAL_CHECKPOINT = REPO_ROOT / "weights" / "kinect" / "minkuresunet_kinect.tar"
DEFAULT_CANDIDATE_CHECKPOINT = DEFAULT_OFFICIAL_CHECKPOINT
DEFAULT_OUTPUT_DIR = REPO_ROOT / "gsnet_reproduction" / "outputs" / "test_seen_probe"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_single_frame_inference import (  # noqa: E402
    build_frame_paths,
    load_frame_data,
)
from eval_utils import (  # noqa: E402
    compute_metric_delta,
    evaluate_single_frame,
    parse_frame_spec,
    parse_scene_spec,
    run_checkpoint_inference,
    summarize_frame_metrics,
    write_json,
)


def build_scene_frame_pairs(scenes: list[int], frames: list[int]) -> list[tuple[int, int]]:
    return [(scene, frame) for scene in scenes for frame in frames]


def compare_checkpoint_summaries(
    *,
    official: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    delta = compute_metric_delta(
        candidate["metrics_percent"],
        official["metrics_percent"],
    )
    return {
        "delta_metrics_percent": delta,
        "ap_improved": delta["AP"] > 0,
    }


def evaluate_checkpoint_on_pairs(
    *,
    tag: str,
    checkpoint_path: Path,
    scene_frame_pairs: list[tuple[int, int]],
    dataset_root: Path,
    split_dir: str,
    camera: str,
    output_dir: Path,
    eval_num_points: int,
    voxel_size: float,
    seed_feat_dim: int,
    top_k: int,
    device: str,
    seed: int,
) -> dict[str, Any]:
    frame_results: list[dict[str, Any]] = []
    for scene_id, frame in scene_frame_pairs:
        print(f"[{tag}] inference/eval scene={scene_id:04d} frame={frame:04d}", flush=True)
        paths = build_frame_paths(
            dataset_root=dataset_root,
            split=split_dir,
            scene=f"scene_{scene_id:04d}",
            camera=camera,
            frame=f"{frame:04d}",
        )
        frame_data = load_frame_data(
            paths,
            num_points=eval_num_points,
            voxel_size=voxel_size,
            rng=np.random.default_rng(seed + scene_id * 1000 + frame),
        )
        inference = run_checkpoint_inference(
            tag=tag,
            frame_data=frame_data,
            checkpoint_path=checkpoint_path,
            scene_id=scene_id,
            ann_id=frame,
            camera=camera,
            seed_feat_dim=seed_feat_dim,
            output_dir=output_dir,
            device=device,
        )
        evaluation = evaluate_single_frame(
            eval_root=dataset_root,
            dump_path=Path(inference["dump_path"]),
            scene_id=scene_id,
            ann_id=frame,
            camera=camera,
            top_k=top_k,
        )
        frame_results.append(
            {
                "scene": f"scene_{scene_id:04d}",
                "frame": f"{frame:04d}",
                "inference": inference,
                **evaluation,
            }
        )
        del frame_data
        gc.collect()
    return {
        "summary": summarize_frame_metrics(frame_results),
        "frames": frame_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-dir", default="scenes")
    parser.add_argument("--scenes", default="100-102")
    parser.add_argument("--frames", default="0-31")
    parser.add_argument("--camera", default="kinect")
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--candidate-name", default="tiny_cross_scene_800iters")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--eval-num-points", type=int, default=15000)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--seed-feat-dim", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_json = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json is not None
        else output_dir / "summary.json"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = parse_scene_spec(args.scenes)
    frames = parse_frame_spec(args.frames)
    scene_frame_pairs = build_scene_frame_pairs(scenes, frames)

    required_paths = [dataset_root, args.official_checkpoint, args.candidate_checkpoint]
    missing = [Path(path) for path in required_paths if not Path(path).exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing required paths:\n{formatted}")

    result: dict[str, Any] = {
        "status": "started",
        "config": {
            "dataset_root": str(dataset_root),
            "split_dir": args.split_dir,
            "scenes": [f"scene_{scene:04d}" for scene in scenes],
            "frames": [f"{frame:04d}" for frame in frames],
            "camera": args.camera,
            "official_checkpoint": str(Path(args.official_checkpoint).expanduser().resolve()),
            "candidate_checkpoint": str(Path(args.candidate_checkpoint).expanduser().resolve()),
            "candidate_name": args.candidate_name,
            "eval_num_points": args.eval_num_points,
            "top_k": args.top_k,
        },
    }

    try:
        print(
            "[official] test_seen probe "
            f"scenes={len(scenes)} frames_per_scene={len(frames)} total={len(scene_frame_pairs)}",
            flush=True,
        )
        result["official"] = evaluate_checkpoint_on_pairs(
            tag="official",
            checkpoint_path=Path(args.official_checkpoint).expanduser().resolve(),
            scene_frame_pairs=scene_frame_pairs,
            dataset_root=dataset_root,
            split_dir=args.split_dir,
            camera=args.camera,
            output_dir=output_dir,
            eval_num_points=args.eval_num_points,
            voxel_size=args.voxel_size,
            seed_feat_dim=args.seed_feat_dim,
            top_k=args.top_k,
            device=args.device,
            seed=args.seed,
        )
        print(
            "[candidate] test_seen probe "
            f"checkpoint={args.candidate_name}",
            flush=True,
        )
        result["candidate"] = evaluate_checkpoint_on_pairs(
            tag=args.candidate_name,
            checkpoint_path=Path(args.candidate_checkpoint).expanduser().resolve(),
            scene_frame_pairs=scene_frame_pairs,
            dataset_root=dataset_root,
            split_dir=args.split_dir,
            camera=args.camera,
            output_dir=output_dir,
            eval_num_points=args.eval_num_points,
            voxel_size=args.voxel_size,
            seed_feat_dim=args.seed_feat_dim,
            top_k=args.top_k,
            device=args.device,
            seed=args.seed,
        )
        comparison = compare_checkpoint_summaries(
            official=result["official"]["summary"],
            candidate=result["candidate"]["summary"],
        )
        result.update(comparison)
        result["status"] = "ok"
        print(
            "[delta] AP={AP:+.4f} AP0.8={AP08:+.4f} AP0.4={AP04:+.4f}".format(
                AP=result["delta_metrics_percent"]["AP"],
                AP08=result["delta_metrics_percent"]["AP0.8"],
                AP04=result["delta_metrics_percent"]["AP0.4"],
            ),
            flush=True,
        )
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        write_json(output_json, result)
        raise

    write_json(output_json, result)
    print(f"wrote {output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
