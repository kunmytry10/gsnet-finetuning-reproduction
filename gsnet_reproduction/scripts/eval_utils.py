#!/usr/bin/env python3
"""Shared helpers for GSNet reproduction evaluation scripts."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GSNET_ROOT = REPO_ROOT / "external" / "graspness_unofficial"
SCRIPT_DIR = Path(__file__).resolve().parent
FRICTION_COEFFICIENTS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]


def setup_gsnet_paths() -> None:
    for path in (
        GSNET_ROOT,
        GSNET_ROOT / "pointnet2",
        GSNET_ROOT / "utils",
        GSNET_ROOT / "models",
        GSNET_ROOT / "dataset",
        SCRIPT_DIR,
    ):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def parse_frame_spec(spec: str) -> list[int]:
    frames: list[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = [x.strip() for x in part.split("-", 1)]
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"invalid frame range: {part!r}")
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"descending frame range is not supported: {part!r}")
            frames.extend(range(start, end + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid frame id: {part!r}")
            frames.append(int(part))
    if not frames:
        raise ValueError("frame spec produced no frames")
    return list(dict.fromkeys(frames))


def _parse_scene_token(token: str) -> int:
    value = token.strip()
    if value.startswith("scene_"):
        value = value.removeprefix("scene_")
    if not value.isdigit():
        raise ValueError(f"invalid scene id: {token!r}")
    return int(value)


def parse_scene_spec(spec: str) -> list[int]:
    scenes: list[int] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = [x.strip() for x in part.split("-", 1)]
            start = _parse_scene_token(start_text)
            end = _parse_scene_token(end_text)
            if end < start:
                raise ValueError(f"descending scene range is not supported: {part!r}")
            scenes.extend(range(start, end + 1))
        else:
            scenes.append(_parse_scene_token(part))
    if not scenes:
        raise ValueError("scene spec produced no scenes")
    return list(dict.fromkeys(scenes))


def compute_metric_delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in sorted(before)}


def summarize_frame_metrics(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not frame_results:
        raise ValueError("cannot summarize an empty frame result list")
    metric_names = sorted(frame_results[0]["metrics_percent"])
    metrics = {
        name: float(
            np.mean([frame["metrics_percent"][name] for frame in frame_results], dtype=np.float64)
        )
        for name in metric_names
    }
    return {
        "frame_count": len(frame_results),
        "metrics_percent": metrics,
        "evaluated_grasp_count": int(
            sum(frame.get("evaluated_grasp_count", 0) for frame in frame_results)
        ),
        "positive_score_count": int(
            sum(frame.get("positive_score_count", 0) for frame in frame_results)
        ),
        "collision_count": int(sum(frame.get("collision_count", 0) for frame in frame_results)),
    }


def save_prediction_dump(predictions: np.ndarray, dump_path: Path) -> None:
    setup_gsnet_paths()
    from graspnetAPI.graspnet_eval import GraspGroup

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    GraspGroup(predictions).save_npy(str(dump_path))


def compute_accuracy_from_scores(score_list: np.ndarray, top_k: int) -> np.ndarray:
    grasp_accuracy = np.zeros((top_k, len(FRICTION_COEFFICIENTS)), dtype=np.float64)
    for fric_idx, fric in enumerate(FRICTION_COEFFICIENTS):
        for k in range(top_k):
            if k + 1 > len(score_list):
                grasp_accuracy[k, fric_idx] = np.sum(
                    ((score_list <= fric) & (score_list > 0)).astype(int)
                ) / float(k + 1)
            else:
                grasp_accuracy[k, fric_idx] = np.sum(
                    ((score_list[: k + 1] <= fric) & (score_list[: k + 1] > 0)).astype(int)
                ) / float(k + 1)
    return grasp_accuracy


def metrics_from_accuracy(accuracy: np.ndarray) -> dict[str, float]:
    return {
        "AP": float(np.mean(accuracy, dtype=np.float64) * 100.0),
        "AP0.4": float(np.mean(accuracy[:, 1], dtype=np.float64) * 100.0),
        "AP0.8": float(np.mean(accuracy[:, 3], dtype=np.float64) * 100.0),
    }


def evaluate_single_frame(
    *,
    eval_root: Path,
    dump_path: Path,
    scene_id: int,
    ann_id: int,
    camera: str,
    top_k: int,
) -> dict[str, Any]:
    setup_gsnet_paths()
    from graspnetAPI.graspnet_eval import GraspGroup, GraspNetEval
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.eval_utils import (
        create_table_points,
        eval_grasp,
        transform_points,
        voxel_sample_points,
    )

    started = time.perf_counter()
    evaluator = GraspNetEval(root=str(eval_root), camera=camera, split="custom")
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)

    model_list, dexmodel_list, _ = evaluator.get_scene_models(scene_id, ann_id=0)
    model_sampled_list = [voxel_sample_points(model, 0.008) for model in model_list]

    grasp_group = GraspGroup().from_npy(str(dump_path))
    _, pose_list, camera_pose, align_mat = evaluator.get_model_poses(scene_id, ann_id)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))

    gg_array = grasp_group.grasp_group_array
    if len(gg_array) > 0:
        gg_array[gg_array[:, 1] < 0, 1] = 0
        gg_array[gg_array[:, 1] > 0.1, 1] = 0.1
        grasp_group.grasp_group_array = gg_array

    grasp_list, score_list, collision_mask_list = eval_grasp(
        grasp_group,
        model_sampled_list,
        dexmodel_list,
        pose_list,
        config,
        table=table_trans,
        voxel_size=0.008,
        TOP_K=top_k,
    )

    grasp_list = [x for x in grasp_list if len(x) != 0]
    score_list = [x for x in score_list if len(x) != 0]
    collision_mask_list = [x for x in collision_mask_list if len(x) != 0]

    if len(grasp_list) == 0:
        accuracy = np.zeros((top_k, len(FRICTION_COEFFICIENTS)), dtype=np.float64)
        return {
            "metrics_percent": metrics_from_accuracy(accuracy),
            "accuracy_shape": list(accuracy.shape),
            "evaluated_grasp_count": 0,
            "positive_score_count": 0,
            "collision_count": 0,
            "elapsed_sec": time.perf_counter() - started,
        }

    grasp_array = np.concatenate(grasp_list)
    scores = np.concatenate(score_list)
    collisions = np.concatenate(collision_mask_list)
    order = np.argsort(-grasp_array[:, 0])
    grasp_array = grasp_array[order]
    scores = scores[order]
    collisions = collisions[order]

    accuracy = compute_accuracy_from_scores(scores, top_k=top_k)
    return {
        "metrics_percent": metrics_from_accuracy(accuracy),
        "accuracy_shape": list(accuracy.shape),
        "evaluated_grasp_count": int(len(grasp_array)),
        "positive_score_count": int(np.count_nonzero(scores > 0)),
        "collision_count": int(np.count_nonzero(collisions)),
        "top_scores": [float(x) for x in scores[: min(10, len(scores))].tolist()],
        "elapsed_sec": time.perf_counter() - started,
    }


def run_checkpoint_inference(
    *,
    tag: str,
    frame_data: Any,
    checkpoint_path: Path,
    scene_id: int,
    ann_id: int,
    camera: str,
    seed_feat_dim: int,
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    from run_single_frame_inference import run_inference

    infer_result = run_inference(
        frame_data.model_input,
        checkpoint_path=checkpoint_path,
        seed_feat_dim=seed_feat_dim,
        device_name=device,
    )
    dump_path = (
        Path(output_dir)
        / "dumps"
        / tag
        / f"scene_{scene_id:04d}"
        / camera
        / f"{ann_id:04d}.npy"
    )
    save_prediction_dump(infer_result.predictions, dump_path)
    predictions = np.asarray(infer_result.predictions)
    return {
        "tag": tag,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": infer_result.checkpoint_epoch,
        "dump_path": str(dump_path),
        "raw_prediction_count": int(len(predictions)),
        "raw_score_min": float(np.min(predictions[:, 0])) if len(predictions) else None,
        "raw_score_max": float(np.max(predictions[:, 0])) if len(predictions) else None,
        "nms_sorted_count": int(len(infer_result.top_grasps)),
        "timings_sec": infer_result.timings_sec,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
