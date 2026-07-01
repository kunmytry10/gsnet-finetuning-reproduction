#!/usr/bin/env python3
"""Run a single-frame GSNet inference smoke test and save 3D results."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
GSNET_ROOT = REPO_ROOT / "external" / "graspness_unofficial"

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "graspnet"
DEFAULT_SPLIT = "train_1"
DEFAULT_SCENE = "scene_0000"
DEFAULT_CAMERA = "kinect"
DEFAULT_FRAME = "0000"
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "kinect" / "minkuresunet_kinect.tar"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "gsnet_reproduction" / "outputs"
DEFAULT_VIS_POINT_LIMIT = 120000
DEFAULT_RENDER_POINT_SIZE = 5.0
RENDER_VIEWS = ("top", "front", "side", "iso")
LEGACY_RENDER_OUTPUTS = (
    "grasp_result_3d.png",
    "grasp_result_front.png",
    "grasp_result_side.png",
)
DEFAULT_RENDER_FOREGROUND_THRESHOLD = 8
DEFAULT_RENDER_CROP_PADDING = 72
DEFAULT_RGB_REPORT_TOPK = 5
DEFAULT_REPORT_TOPKS = (1, 5)


@dataclass(frozen=True)
class FramePaths:
    dataset_root: Path
    split: str
    scene: str
    camera: str
    frame: str
    scene_dir: Path
    depth_path: Path
    label_path: Path
    meta_path: Path
    camera_poses_path: Path
    cam0_wrt_table_path: Path
    rgb_path: Path


@dataclass(frozen=True)
class FrameData:
    model_input: dict[str, np.ndarray]
    intrinsic: np.ndarray
    label_image: np.ndarray
    raw_cloud: np.ndarray
    raw_colors: np.ndarray | None
    sampled_cloud: np.ndarray
    sampled_colors: np.ndarray | None
    sampled_indices: np.ndarray
    depth_shape: tuple[int, int]
    masked_point_count: int
    valid_depth_count: int
    workspace_point_count: int


@dataclass(frozen=True)
class InferenceResult:
    predictions: np.ndarray
    top_grasps: Any
    checkpoint_epoch: int | str
    device: str
    timings_sec: dict[str, float]


@dataclass(frozen=True)
class ModeOutputPaths:
    grasps_path: Path
    gripper_mesh_path: Path
    rgb_overlay_path: Path
    top_render_path: Path


def canonical_scene(scene: str) -> str:
    value = str(scene).strip()
    if value.startswith("scene_"):
        value = value.removeprefix("scene_")
    if not value.isdigit():
        raise ValueError(f"scene must be numeric or scene_XXXX, got {scene!r}")
    return f"scene_{int(value):04d}"


def canonical_frame(frame: str) -> str:
    value = str(frame).strip()
    if not value.isdigit():
        raise ValueError(f"frame must be numeric, got {frame!r}")
    return f"{int(value):04d}"


def parse_positive_int_csv(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            continue
        number = int(part)
        if number <= 0:
            raise ValueError(f"top-k values must be positive, got {number}")
        parsed.append(number)
    if not parsed:
        raise ValueError("expected at least one top-k value")
    return tuple(dict.fromkeys(parsed))


def build_frame_paths(
    dataset_root: Path,
    split: str,
    scene: str,
    camera: str,
    frame: str,
) -> FramePaths:
    dataset_root = Path(dataset_root).expanduser().resolve()
    scene_name = canonical_scene(scene)
    frame_name = canonical_frame(frame)
    scene_dir = dataset_root / split / scene_name / camera
    return FramePaths(
        dataset_root=dataset_root,
        split=split,
        scene=scene_name,
        camera=camera,
        frame=frame_name,
        scene_dir=scene_dir,
        depth_path=scene_dir / "depth" / f"{frame_name}.png",
        label_path=scene_dir / "label" / f"{frame_name}.png",
        meta_path=scene_dir / "meta" / f"{frame_name}.mat",
        camera_poses_path=scene_dir / "camera_poses.npy",
        cam0_wrt_table_path=scene_dir / "cam0_wrt_table.npy",
        rgb_path=scene_dir / "rgb" / f"{frame_name}.png",
    )


def build_output_dir(
    output_root: Path,
    scene: str,
    camera: str,
    frame: str,
) -> Path:
    return Path(output_root) / f"single_frame_{scene}_{camera}_{frame}"


def sample_point_cloud(
    points: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if len(points) == 0:
        raise ValueError("workspace mask produced no points")

    if len(points) >= num_points:
        indices = rng.choice(len(points), num_points, replace=False)
    else:
        base_indices = np.arange(len(points), dtype=np.int64)
        repeat_indices = rng.choice(
            len(points),
            num_points - len(points),
            replace=True,
        )
        indices = np.concatenate([base_indices, repeat_indices], axis=0)
    return points[indices], indices.astype(np.int64, copy=False)


def select_visualization_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    max_points: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    if max_points <= 0:
        raise ValueError(f"max_points must be positive, got {max_points}")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if colors is not None and colors.shape != points.shape:
        raise ValueError(
            f"colors must have the same shape as points, got {colors.shape} and {points.shape}"
        )
    if len(points) <= max_points:
        indices = np.arange(len(points), dtype=np.int64)
    else:
        indices = rng.choice(len(points), max_points, replace=False).astype(
            np.int64,
            copy=False,
        )
    selected_colors = colors[indices] if colors is not None else None
    return points[indices], selected_colors, indices


def build_render_output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "top": Path(output_dir) / "grasp_result_top.png",
    }


def build_mode_output_paths(output_dir: Path, mode: str) -> ModeOutputPaths:
    output_dir = Path(output_dir)
    if mode == "top30":
        return ModeOutputPaths(
            grasps_path=output_dir / "top30_grasps.npy",
            gripper_mesh_path=output_dir / "grasp_result_3d.ply",
            rgb_overlay_path=output_dir / "grasp_result_rgb_overlay.png",
            top_render_path=output_dir / "grasp_result_top.png",
        )
    if mode == "collision":
        return ModeOutputPaths(
            grasps_path=output_dir / "top30_collision_filtered.npy",
            gripper_mesh_path=output_dir / "grasp_result_collision_3d.ply",
            rgb_overlay_path=output_dir / "grasp_result_collision_rgb_overlay.png",
            top_render_path=output_dir / "grasp_result_collision_top.png",
        )
    if mode == "balanced":
        return ModeOutputPaths(
            grasps_path=output_dir / "top30_balanced_by_label.npy",
            gripper_mesh_path=output_dir / "grasp_result_balanced_3d.ply",
            rgb_overlay_path=output_dir / "grasp_result_balanced_rgb_overlay.png",
            top_render_path=output_dir / "grasp_result_balanced_top.png",
        )
    raise ValueError(f"unknown output mode {mode!r}")


def remove_legacy_render_outputs(output_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in LEGACY_RENDER_OUTPUTS:
        path = Path(output_dir) / name
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def build_grasp_control_points(grasp: np.ndarray) -> np.ndarray:
    if grasp.shape != (17,):
        raise ValueError(f"expected one grasp row with shape (17,), got {grasp.shape}")
    width = max(float(grasp[1]), 0.01)
    depth = max(float(grasp[3]), 0.0)
    rotation = grasp[4:13].reshape(3, 3).astype(np.float64)
    center = grasp[13:16].astype(np.float64)

    axis_depth = rotation[:, 0]
    axis_width = rotation[:, 1]
    half_width = width / 2.0
    palm_back = 0.02
    tail_length = 0.04

    left_tip = center + axis_depth * depth + axis_width * half_width
    right_tip = center + axis_depth * depth - axis_width * half_width
    palm_center = center - axis_depth * palm_back
    palm_left = palm_center + axis_width * half_width
    palm_right = palm_center - axis_width * half_width
    wrist = palm_center - axis_depth * tail_length
    return np.vstack([center, left_tip, right_tip, palm_left, palm_right, wrist])


def project_camera_points(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsic must have shape (3, 3), got {intrinsic.shape}")
    if np.any(points[:, 2] <= 0):
        raise ValueError("all projected points must have positive z")
    x = intrinsic[0, 0] * points[:, 0] / points[:, 2] + intrinsic[0, 2]
    y = intrinsic[1, 1] * points[:, 1] / points[:, 2] + intrinsic[1, 2]
    return np.stack([x, y], axis=1).astype(np.float32)


def assign_grasp_center_labels(
    grasps: np.ndarray,
    intrinsic: np.ndarray,
    label_image: np.ndarray,
) -> np.ndarray:
    if grasps.ndim != 2 or grasps.shape[1] != 17:
        raise ValueError(f"grasps must have shape (N, 17), got {grasps.shape}")
    if label_image.ndim != 2:
        raise ValueError(f"label_image must be 2D, got {label_image.shape}")

    labels = np.full((len(grasps),), -1, dtype=np.int32)
    centers = grasps[:, 13:16]
    valid_z = centers[:, 2] > 0
    if not np.any(valid_z):
        return labels

    valid_indices = np.flatnonzero(valid_z)
    pixels = project_camera_points(centers[valid_z], intrinsic)
    rounded = np.rint(pixels).astype(np.int32)
    height, width = label_image.shape
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
    )
    if np.any(in_bounds):
        source_indices = valid_indices[in_bounds]
        xs = rounded[in_bounds, 0]
        ys = rounded[in_bounds, 1]
        labels[source_indices] = label_image[ys, xs].astype(np.int32)
    return labels


def _score_sorted_indices(grasps: np.ndarray) -> np.ndarray:
    scores = grasps[:, 0]
    return np.lexsort((np.arange(len(grasps)), -scores))


def balance_grasps_by_label(
    grasps: np.ndarray,
    labels: np.ndarray,
    topk: int,
) -> np.ndarray:
    if topk <= 0 or len(grasps) == 0:
        return np.empty((0,), dtype=np.int64)
    if labels.shape != (len(grasps),):
        raise ValueError(f"labels must have shape ({len(grasps)},), got {labels.shape}")

    sorted_indices = _score_sorted_indices(grasps)
    valid_sorted = [int(index) for index in sorted_indices if labels[index] > 0]
    if not valid_sorted:
        return sorted_indices[:topk].astype(np.int64, copy=False)

    ordered_labels: list[int] = []
    for index in valid_sorted:
        label = int(labels[index])
        if label not in ordered_labels:
            ordered_labels.append(label)

    quota = max(1, topk // len(ordered_labels))
    selected: list[int] = []
    selected_set: set[int] = set()
    for label in ordered_labels:
        added_for_label = 0
        for index in valid_sorted:
            if labels[index] != label or index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            added_for_label += 1
            if len(selected) >= topk or added_for_label >= quota:
                break
        if len(selected) >= topk:
            return np.array(selected[:topk], dtype=np.int64)

    for index in valid_sorted:
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= topk:
                return np.array(selected[:topk], dtype=np.int64)

    for index in sorted_indices:
        index = int(index)
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= topk:
                break
    return np.array(selected[:topk], dtype=np.int64)


def setup_gsnet_paths() -> None:
    paths = [
        GSNET_ROOT,
        GSNET_ROOT / "utils",
        GSNET_ROOT / "models",
        GSNET_ROOT / "dataset",
        GSNET_ROOT / "pointnet2",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def require_inputs(paths: FramePaths, checkpoint_path: Path) -> None:
    required = [
        paths.depth_path,
        paths.label_path,
        paths.meta_path,
        paths.camera_poses_path,
        paths.cam0_wrt_table_path,
        checkpoint_path,
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"missing required input files:\n{formatted}")


def load_frame_data(
    paths: FramePaths,
    num_points: int,
    voxel_size: float,
    rng: np.random.Generator,
) -> FrameData:
    setup_gsnet_paths()

    import scipy.io as scio
    from PIL import Image

    from data_utils import (  # type: ignore[import-not-found]
        CameraInfo,
        create_point_cloud_from_depth_image,
        get_workspace_mask,
    )

    depth = np.array(Image.open(paths.depth_path))
    seg = np.array(Image.open(paths.label_path))
    meta = scio.loadmat(paths.meta_path)

    intrinsic = np.asarray(meta["intrinsic_matrix"], dtype=np.float32)
    factor_depth = float(np.asarray(meta["factor_depth"]).squeeze())
    height, width = depth.shape
    camera = CameraInfo(
        float(width),
        float(height),
        float(intrinsic[0, 0]),
        float(intrinsic[1, 1]),
        float(intrinsic[0, 2]),
        float(intrinsic[1, 2]),
        factor_depth,
    )

    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    depth_mask = depth > 0
    if not np.any(seg > 0):
        raise ValueError(f"segmentation has no foreground labels: {paths.label_path}")

    camera_poses = np.load(paths.camera_poses_path)
    align_mat = np.load(paths.cam0_wrt_table_path)
    frame_index = int(paths.frame)
    trans = np.dot(align_mat, camera_poses[frame_index])
    workspace_mask = get_workspace_mask(
        cloud,
        seg,
        trans=trans,
        organized=True,
        outlier=0.02,
    )
    mask = depth_mask & workspace_mask
    cloud_masked = cloud[mask].astype(np.float32)
    cloud_sampled, sampled_indices = sample_point_cloud(cloud_masked, num_points, rng)

    raw_colors = None
    sampled_colors = None
    if paths.rgb_path.exists():
        rgb = np.array(Image.open(paths.rgb_path).convert("RGB"))
        if rgb.shape[:2] == depth.shape:
            raw_colors = rgb[mask].astype(np.uint8)
            sampled_colors = raw_colors[sampled_indices].astype(np.uint8)

    model_input = {
        "point_clouds": cloud_sampled.astype(np.float32),
        "coors": (cloud_sampled.astype(np.float32) / float(voxel_size)),
        "feats": np.ones_like(cloud_sampled, dtype=np.float32),
    }
    return FrameData(
        model_input=model_input,
        intrinsic=intrinsic,
        label_image=seg,
        raw_cloud=cloud_masked,
        raw_colors=raw_colors,
        sampled_cloud=cloud_sampled.astype(np.float32),
        sampled_colors=sampled_colors,
        sampled_indices=sampled_indices,
        depth_shape=(height, width),
        masked_point_count=int(len(cloud_masked)),
        valid_depth_count=int(depth_mask.sum()),
        workspace_point_count=int(workspace_mask.sum()),
    )


def move_batch_to_device(batch_data: dict[str, Any], device: Any) -> dict[str, Any]:
    for key, value in list(batch_data.items()):
        if "list" in key:
            for i in range(len(value)):
                for j in range(len(value[i])):
                    value[i][j] = value[i][j].to(device)
        elif hasattr(value, "to"):
            batch_data[key] = value.to(device)
    return batch_data


def load_checkpoint(torch_module: Any, checkpoint_path: Path, device: Any) -> dict[str, Any]:
    try:
        return torch_module.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch_module.load(checkpoint_path, map_location=device)


def run_inference(
    data_input: dict[str, np.ndarray],
    checkpoint_path: Path,
    seed_feat_dim: int,
    device_name: str,
) -> InferenceResult:
    setup_gsnet_paths()

    import torch
    from dataset.graspnet_dataset import minkowski_collate_fn  # type: ignore[import-not-found]
    from graspnetAPI.graspnet_eval import GraspGroup
    from models.graspnet import GraspNet, pred_decode  # type: ignore[import-not-found]

    timings: dict[str, float] = {}
    started = time.perf_counter()
    batch_data = minkowski_collate_fn([data_input])
    timings["collate"] = time.perf_counter() - started

    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    started = time.perf_counter()
    net = GraspNet(seed_feat_dim=seed_feat_dim, is_training=False)
    net.to(device)
    checkpoint = load_checkpoint(torch, checkpoint_path, device)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    timings["load_model"] = time.perf_counter() - started

    batch_data = move_batch_to_device(batch_data, device)
    started = time.perf_counter()
    with torch.no_grad():
        end_points = net(batch_data)
        grasp_preds = pred_decode(end_points)
    timings["forward_decode"] = time.perf_counter() - started

    predictions = grasp_preds[0].detach().cpu().numpy()
    grasp_group = GraspGroup(predictions)
    started = time.perf_counter()
    if len(grasp_group) > 0:
        grasp_group = grasp_group.nms().sort_by_score()
    timings["nms_sort"] = time.perf_counter() - started

    return InferenceResult(
        predictions=predictions,
        top_grasps=grasp_group,
        checkpoint_epoch=checkpoint.get("epoch", "unknown"),
        device=str(device),
        timings_sec=timings,
    )


def save_point_cloud(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray | None,
) -> Path:
    import open3d as o3d

    path.parent.mkdir(parents=True, exist_ok=True)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None and len(colors) == len(points):
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    ok = o3d.io.write_point_cloud(str(path), cloud)
    if not ok:
        raise RuntimeError(f"failed to write point cloud: {path}")
    return path


def save_gripper_mesh(path: Path, grasp_group: Any) -> Path | None:
    import open3d as o3d

    if len(grasp_group) == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = o3d.geometry.TriangleMesh()
    for geometry in grasp_group.to_open3d_geometry_list():
        combined += geometry
    combined.compute_vertex_normals()
    ok = o3d.io.write_triangle_mesh(str(path), combined)
    if not ok:
        raise RuntimeError(f"failed to write gripper mesh: {path}")
    return path


def _render_geometries(
    cloud: Any | None,
    grippers: Any | None,
    center: np.ndarray,
    eye: np.ndarray,
    up: np.ndarray,
    width: int,
    height: int,
    point_size: float,
) -> np.ndarray:
    from open3d.visualization import rendering

    renderer = rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    if cloud is not None:
        point_material = rendering.MaterialRecord()
        point_material.shader = "defaultUnlit"
        point_material.point_size = float(point_size)
        renderer.scene.add_geometry("cloud", cloud, point_material)

    if grippers is not None:
        mesh_material = rendering.MaterialRecord()
        mesh_material.shader = "defaultLit"
        renderer.scene.add_geometry("grippers", grippers, mesh_material)

    renderer.setup_camera(52.0, center, eye, up)
    return np.asarray(renderer.render_to_image()).copy()


def _foreground_mask_from_corners(
    image: np.ndarray,
    threshold: int = DEFAULT_RENDER_FOREGROUND_THRESHOLD,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"expected RGB/RGBA image, got shape {image.shape}")
    rgb = image[:, :, :3].astype(np.int16, copy=False)
    corners = np.array(
        [
            rgb[0, 0],
            rgb[0, -1],
            rgb[-1, 0],
            rgb[-1, -1],
        ],
        dtype=np.int16,
    )
    background = np.median(corners, axis=0)
    diff = np.abs(rgb - background)
    return np.any(diff > int(threshold), axis=2)


def _crop_render_to_content(
    image: np.ndarray,
    padding_px: int = DEFAULT_RENDER_CROP_PADDING,
    threshold: int = DEFAULT_RENDER_FOREGROUND_THRESHOLD,
) -> np.ndarray:
    if padding_px < 0:
        raise ValueError(f"padding_px must be non-negative, got {padding_px}")
    mask = _foreground_mask_from_corners(image, threshold=threshold)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0 or len(ys) == 0:
        return image
    y0 = max(int(ys.min()) - padding_px, 0)
    y1 = min(int(ys.max()) + padding_px + 1, image.shape[0])
    x0 = max(int(xs.min()) - padding_px, 0)
    x1 = min(int(xs.max()) + padding_px + 1, image.shape[1])
    return image[y0:y1, x0:x1].copy()


def _composite_grippers_over_cloud(cloud_image: np.ndarray, gripper_image: np.ndarray) -> np.ndarray:
    if cloud_image.shape != gripper_image.shape:
        raise ValueError("rendered cloud and gripper images must have the same shape")
    composite = cloud_image.copy()
    mask = _foreground_mask_from_corners(gripper_image)
    composite[mask] = gripper_image[mask]
    return composite


def try_render_preview(
    pointcloud_path: Path,
    gripper_mesh_path: Path | None,
    output_path: Path,
    timeout_sec: int,
    view: str = "top",
    point_size: float = DEFAULT_RENDER_POINT_SIZE,
) -> dict[str, Any]:
    if gripper_mesh_path is None:
        return {"ok": False, "reason": "no gripper mesh to render"}

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render-only",
        "--render-pointcloud",
        str(pointcloud_path),
        "--render-grippers",
        str(gripper_mesh_path),
        "--render-output",
        str(output_path),
        "--render-view",
        view,
        "--render-point-size",
        str(point_size),
    ]
    env = os.environ.copy()
    env.setdefault("EGL_PLATFORM", "surfaceless")
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": f"render timed out after {timeout_sec}s"}

    if result.returncode != 0:
        return {
            "ok": False,
            "returncode": result.returncode,
            "reason": result.stdout.strip()[-2000:],
        }
    return {"ok": output_path.exists(), "path": str(output_path)}


def try_render_previews(
    pointcloud_path: Path,
    gripper_mesh_path: Path | None,
    output_paths: dict[str, Path],
    timeout_sec: int,
    point_size: float,
) -> dict[str, dict[str, Any]]:
    return {
        view: try_render_preview(
            pointcloud_path,
            gripper_mesh_path,
            output_path,
            timeout_sec,
            view=view,
            point_size=point_size,
        )
        for view, output_path in output_paths.items()
    }


def camera_for_view(
    center: np.ndarray,
    extent: float,
    view: str,
) -> tuple[np.ndarray, np.ndarray]:
    camera_offsets = {
        "top": (np.array([0.0, -0.02, 2.2]), np.array([0.0, -1.0, 0.0])),
        "front": (np.array([0.0, -2.0, 0.7]), np.array([0.0, 0.0, 1.0])),
        "side": (np.array([2.0, -0.15, 0.85]), np.array([0.0, 0.0, 1.0])),
        "iso": (np.array([1.45, -1.65, 1.2]), np.array([0.0, 0.0, 1.0])),
    }
    if view not in camera_offsets:
        raise ValueError(f"unknown render view {view!r}; expected one of {RENDER_VIEWS}")
    offset, up = camera_offsets[view]
    return center + offset * extent, up


def render_scene_png(
    pointcloud_path: Path,
    gripper_mesh_path: Path,
    output_path: Path,
    width: int = 1400,
    height: int = 900,
    view: str = "top",
    point_size: float = DEFAULT_RENDER_POINT_SIZE,
    crop_to_content: bool = True,
    crop_padding_px: int = DEFAULT_RENDER_CROP_PADDING,
) -> Path:
    import open3d as o3d
    from PIL import Image

    cloud = o3d.io.read_point_cloud(str(pointcloud_path))
    grippers = o3d.io.read_triangle_mesh(str(gripper_mesh_path))
    geometries = [cloud, grippers]

    bbox = o3d.geometry.AxisAlignedBoundingBox()
    for geometry in geometries:
        bbox += geometry.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = max(float(np.max(bbox.get_extent())), 0.1)
    eye, up = camera_for_view(center, extent, view)

    cloud_image = _render_geometries(
        cloud,
        None,
        center,
        eye,
        up,
        width,
        height,
        point_size,
    )
    gripper_image = _render_geometries(
        None,
        grippers,
        center,
        eye,
        up,
        width,
        height,
        point_size,
    )
    image = _composite_grippers_over_cloud(cloud_image, gripper_image)
    if crop_to_content:
        image = _crop_render_to_content(image, padding_px=crop_padding_px)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(output_path)
    return output_path


def save_report_render_contact_sheet(
    render_paths: list[Path],
    output_path: Path,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> Path:
    from PIL import Image

    if not render_paths:
        raise ValueError("render_paths must not be empty")

    images = [Image.open(path).convert("RGB") for path in render_paths]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images)
    canvas = Image.new("RGB", (total_width, max_height), color=background_rgb)

    x_offset = 0
    for image in images:
        y_offset = (max_height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def save_report_renders(
    output_dir: Path,
    dense_pointcloud_path: Path,
    rgb_path: Path,
    intrinsic: np.ndarray,
    top_grasps: Any,
    report_topks: tuple[int, ...] = DEFAULT_REPORT_TOPKS,
    rgb_only: bool = False,
    overlay_edge_thickness: int = 2,
    overlay_fill_alpha: float = 0.10,
    overlay_outline_extra: int = 3,
) -> dict[str, str | None]:
    report_dir = Path(output_dir) / "report_pngs"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_outputs: dict[str, str | None] = {
        "rgb_only": str(rgb_only),
        "report_topks": ",".join(str(topk) for topk in report_topks),
    }

    if len(top_grasps) == 0:
        return report_outputs

    for topk in report_topks:
        selected = top_grasps[: min(topk, len(top_grasps))]
        rgb_path_out = save_rgb_grasp_overlay(
            report_dir / f"top{topk}_rgb_overlay.png",
            rgb_path,
            selected,
            intrinsic,
            max_grasps=topk,
            edge_thickness=overlay_edge_thickness,
            fill_alpha=overlay_fill_alpha,
            outline_extra=overlay_outline_extra,
        )
        report_outputs[f"top{topk}_rgb_overlay"] = str(rgb_path_out) if rgb_path_out else None

    if rgb_only:
        return report_outputs

    if 1 in report_topks:
        top1_mesh_path = save_gripper_mesh(report_dir / "top1_gripper_mesh.ply", top_grasps[:1])
    else:
        top1_mesh_path = None
    if top1_mesh_path is not None:
        top1_iso_path = render_scene_png(
            dense_pointcloud_path,
            top1_mesh_path,
            report_dir / "top1_iso.png",
            width=1600,
            height=1200,
            view="iso",
            point_size=DEFAULT_RENDER_POINT_SIZE,
        )
        top1_top_path = render_scene_png(
            dense_pointcloud_path,
            top1_mesh_path,
            report_dir / "top1_top.png",
            width=1600,
            height=1200,
            view="top",
            point_size=DEFAULT_RENDER_POINT_SIZE,
        )
        report_outputs["top1_iso"] = str(top1_iso_path)
        report_outputs["top1_top"] = str(top1_top_path)

    top5 = top_grasps[: min(DEFAULT_RGB_REPORT_TOPK, len(top_grasps))]
    top5_mesh_path = (
        save_gripper_mesh(report_dir / "top5_gripper_mesh.ply", top5)
        if DEFAULT_RGB_REPORT_TOPK in report_topks
        else None
    )
    if top5_mesh_path is not None:
        top5_iso_path = render_scene_png(
            dense_pointcloud_path,
            top5_mesh_path,
            report_dir / "top5_iso.png",
            width=1600,
            height=1200,
            view="iso",
            point_size=DEFAULT_RENDER_POINT_SIZE,
        )
        top5_view_paths: list[Path] = []
        for view in ("top", "front", "side", "iso"):
            render_path = render_scene_png(
                dense_pointcloud_path,
                top5_mesh_path,
                report_dir / f"top5_{view}.png",
                width=1400,
                height=1000,
                view=view,
                point_size=DEFAULT_RENDER_POINT_SIZE,
            )
            top5_view_paths.append(render_path)
        contact_path = save_report_render_contact_sheet(
            top5_view_paths,
            report_dir / "top5_multiview.png",
        )
        report_outputs["top5_iso"] = str(top5_iso_path)
        report_outputs["top5_multiview"] = str(contact_path)

    return report_outputs


def _score_color_bgr(score: float, min_score: float, max_score: float) -> tuple[int, int, int]:
    if max_score <= min_score:
        t = 1.0
    else:
        t = (score - min_score) / (max_score - min_score)
    t = float(np.clip(t, 0.0, 1.0))
    blue = int(255 * (1.0 - t))
    red = int(255 * t)
    return (blue, 0, red)


def _draw_polyline_with_outline(
    image_bgr: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    outline_extra: int,
) -> None:
    import cv2

    if outline_extra > 0:
        cv2.line(image_bgr, start, end, (15, 15, 15), thickness + outline_extra, cv2.LINE_AA)
    cv2.line(image_bgr, start, end, color, thickness, cv2.LINE_AA)


def _mesh_unique_edges(triangles: np.ndarray) -> np.ndarray:
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"triangles must have shape (N, 3), got {triangles.shape}")
    if len(triangles) == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.concatenate(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ],
        axis=0,
    )
    edges = np.sort(edges.astype(np.int32, copy=False), axis=1)
    return np.unique(edges, axis=0)


def _grasp_mesh_arrays(grasp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    setup_gsnet_paths()
    import open3d as o3d

    group = make_grasp_group(grasp.reshape(1, -1))
    geometry_list = group.to_open3d_geometry_list()
    if not geometry_list:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int32),
        )
    mesh = o3d.geometry.TriangleMesh()
    for geometry in geometry_list:
        mesh += geometry
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    return vertices, triangles


def _draw_projected_mesh_overlay(
    image_bgr: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    intrinsic: np.ndarray,
    color: tuple[int, int, int],
    edge_thickness: int,
    fill_alpha: float,
    outline_extra: int,
) -> bool:
    import cv2

    if len(vertices) == 0 or len(triangles) == 0:
        return False
    if np.any(vertices[:, 2] <= 0):
        return False

    height, width = image_bgr.shape[:2]
    pixels = project_camera_points(vertices, intrinsic)
    rounded = np.rint(pixels).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= -0.1 * width)
        & (rounded[:, 0] <= 1.1 * width)
        & (rounded[:, 1] >= -0.1 * height)
        & (rounded[:, 1] <= 1.1 * height)
    )
    if not np.any(in_bounds):
        return False

    overlay = image_bgr.copy()
    hull = cv2.convexHull(rounded)
    if len(hull) >= 3:
        cv2.fillConvexPoly(overlay, hull, color, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, fill_alpha, image_bgr, 1.0 - fill_alpha, 0.0, dst=image_bgr)

    edges = _mesh_unique_edges(triangles)
    for start_idx, end_idx in edges:
        if not in_bounds[start_idx] and not in_bounds[end_idx]:
            continue
        start = tuple(rounded[start_idx])
        end = tuple(rounded[end_idx])
        _draw_polyline_with_outline(image_bgr, start, end, color, edge_thickness, outline_extra)
    return True


def save_rgb_grasp_overlay(
    path: Path,
    rgb_path: Path,
    grasp_group: Any,
    intrinsic: np.ndarray,
    max_grasps: int | None = None,
    edge_thickness: int = 2,
    fill_alpha: float = 0.10,
    outline_extra: int = 3,
) -> Path | None:
    import cv2
    from PIL import Image

    if not rgb_path.exists() or len(grasp_group) == 0:
        return None

    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    grasps = np.asarray(grasp_group.grasp_group_array)
    if max_grasps is not None:
        grasps = grasps[:max_grasps]
    scores = grasps[:, 0]
    min_score = float(scores.min())
    max_score = float(scores.max())

    for rank, grasp in enumerate(grasps):
        color = _score_color_bgr(float(grasp[0]), min_score, max_score)
        rank_edge_thickness = max(1, int(edge_thickness) + (1 if rank == 0 and edge_thickness > 1 else 0))
        rank_fill_alpha = float(fill_alpha) * (1.25 if rank == 0 else 1.0)
        vertices, triangles = _grasp_mesh_arrays(grasp)
        _draw_projected_mesh_overlay(
            image_bgr,
            vertices,
            triangles,
            intrinsic,
            color,
            edge_thickness=rank_edge_thickness,
            fill_alpha=rank_fill_alpha,
            outline_extra=max(0, int(outline_extra)),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).save(path)
    return path


def make_grasp_group(grasps: np.ndarray) -> Any:
    setup_gsnet_paths()
    from graspnetAPI.graspnet_eval import GraspGroup

    return GraspGroup(grasps)


def collision_filter_grasps(
    grasp_group: Any,
    scene_points: np.ndarray,
    collision_thresh: float,
    voxel_size: float,
    approach_dist: float,
) -> tuple[Any, dict[str, Any]]:
    setup_gsnet_paths()
    from collision_detector import ModelFreeCollisionDetector  # type: ignore[import-not-found]

    before_count = len(grasp_group)
    if before_count == 0:
        return grasp_group, {
            "enabled": True,
            "threshold": collision_thresh,
            "voxel_size": voxel_size,
            "approach_dist": approach_dist,
            "input_count": 0,
            "removed_count": 0,
            "remaining_count": 0,
        }

    detector = ModelFreeCollisionDetector(scene_points, voxel_size=voxel_size)
    collision_mask = detector.detect(
        grasp_group,
        approach_dist=approach_dist,
        collision_thresh=collision_thresh,
    )
    filtered = grasp_group[~collision_mask]
    return filtered, {
        "enabled": True,
        "threshold": collision_thresh,
        "voxel_size": voxel_size,
        "approach_dist": approach_dist,
        "input_count": before_count,
        "removed_count": int(np.asarray(collision_mask).sum()),
        "remaining_count": int(len(filtered)),
    }


def label_distribution(labels: np.ndarray) -> dict[str, int]:
    positive = labels[labels > 0]
    if len(positive) == 0:
        return {}
    unique, counts = np.unique(positive, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts, strict=True)}


def save_grasp_mode_outputs(
    output_paths: ModeOutputPaths,
    grasp_group: Any,
    frame_paths: FramePaths,
    frame_data: FrameData,
    dense_pointcloud_path: Path,
    skip_render: bool,
    render_timeout_sec: int,
    render_point_size: float,
) -> dict[str, Any]:
    output_paths.grasps_path.parent.mkdir(parents=True, exist_ok=True)
    grasp_group.save_npy(output_paths.grasps_path)
    saved_mesh = save_gripper_mesh(output_paths.gripper_mesh_path, grasp_group)
    saved_rgb_overlay = save_rgb_grasp_overlay(
        output_paths.rgb_overlay_path,
        frame_paths.rgb_path,
        grasp_group,
        frame_data.intrinsic,
    )

    if skip_render:
        render_status = {"ok": False, "reason": "skipped by --skip-render"}
    else:
        render_status = try_render_preview(
            dense_pointcloud_path,
            saved_mesh,
            output_paths.top_render_path,
            render_timeout_sec,
            view="top",
            point_size=render_point_size,
        )

    return {
        "count": int(len(grasp_group)),
        "outputs": {
            "grasps": str(output_paths.grasps_path),
            "gripper_mesh": str(saved_mesh) if saved_mesh else None,
            "rgb_overlay": str(saved_rgb_overlay) if saved_rgb_overlay else None,
            "top_render": str(output_paths.top_render_path)
            if output_paths.top_render_path.exists()
            else None,
        },
        "render_status": render_status,
    }


def save_summary(path: Path, summary: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--camera", default=DEFAULT_CAMERA, choices=["kinect", "realsense"])
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-point", type=int, default=15000)
    parser.add_argument("--seed-feat-dim", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--render-timeout-sec", type=int, default=60)
    parser.add_argument("--visualization-point-limit", type=int, default=DEFAULT_VIS_POINT_LIMIT)
    parser.add_argument("--render-point-size", type=float, default=DEFAULT_RENDER_POINT_SIZE)
    parser.add_argument(
        "--report-rgb-only",
        action="store_true",
        help="Only write RGB grasp overlays in report_pngs; skip report point-cloud renders.",
    )
    parser.add_argument(
        "--report-topks",
        default="1,5",
        help="Comma-separated top-k overlays to write under report_pngs, for example 1,5,50.",
    )
    parser.add_argument(
        "--overlay-edge-thickness",
        type=int,
        default=2,
        help="RGB overlay gripper line thickness in pixels.",
    )
    parser.add_argument(
        "--overlay-fill-alpha",
        type=float,
        default=0.10,
        help="RGB overlay transparent fill alpha. Use lower values for dense top-k overlays.",
    )
    parser.add_argument(
        "--overlay-outline-extra",
        type=int,
        default=3,
        help="Extra black outline pixels around RGB overlay gripper lines. Use 0 or 1 for thin plots.",
    )
    parser.add_argument(
        "--collision-thresh",
        type=float,
        default=-1.0,
        help="Enable model-free collision filtering when > 0, for example 0.01.",
    )
    parser.add_argument("--voxel-size-cd", type=float, default=0.01)
    parser.add_argument("--collision-approach-dist", type=float, default=0.05)
    parser.add_argument(
        "--balanced-by-label",
        action="store_true",
        help="Select up to top-k grasps with a per-visible-label balanced first pass.",
    )

    parser.add_argument("--render-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-pointcloud", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-grippers", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--render-view", default="top", choices=RENDER_VIEWS, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.render_only:
        if not args.render_pointcloud or not args.render_grippers or not args.render_output:
            raise SystemExit("--render-only requires render input/output paths")
        render_scene_png(
            args.render_pointcloud,
            args.render_grippers,
            args.render_output,
            view=args.render_view,
            point_size=args.render_point_size,
        )
        return 0

    paths = build_frame_paths(
        dataset_root=args.dataset_root,
        split=args.split,
        scene=args.scene,
        camera=args.camera,
        frame=args.frame,
    )
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    output_dir = build_output_dir(
        args.output_root,
        paths.scene,
        paths.camera,
        paths.frame,
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] dataset frame: {paths.scene_dir}/{paths.frame}")
    print(f"[INFO] checkpoint: {checkpoint_path}")
    print(f"[INFO] output dir: {output_dir}")

    require_inputs(paths, checkpoint_path)
    rng = np.random.default_rng(args.seed)
    vis_rng = np.random.default_rng(args.seed + 1)
    report_topks = parse_positive_int_csv(args.report_topks)

    total_started = time.perf_counter()
    data_started = time.perf_counter()
    frame_data = load_frame_data(paths, args.num_point, args.voxel_size, rng)
    data_time = time.perf_counter() - data_started
    print(
        "[INFO] point cloud: "
        f"masked={frame_data.masked_point_count}, sampled={len(frame_data.sampled_cloud)}"
    )

    inference = run_inference(
        frame_data.model_input,
        checkpoint_path,
        args.seed_feat_dim,
        args.device,
    )
    top_count = min(args.topk, len(inference.top_grasps))
    top_grasps = inference.top_grasps[:top_count]

    predictions_path = output_dir / "predictions.npy"
    pointcloud_path = output_dir / "pointcloud.ply"
    dense_pointcloud_path = output_dir / "pointcloud_dense.ply"
    summary_path = output_dir / "summary.json"
    removed_legacy_renders = remove_legacy_render_outputs(output_dir)

    np.save(predictions_path, inference.predictions)
    save_point_cloud(pointcloud_path, frame_data.sampled_cloud, frame_data.sampled_colors)
    visualization_cloud, visualization_colors, _ = select_visualization_points(
        frame_data.raw_cloud,
        frame_data.raw_colors,
        args.visualization_point_limit,
        vis_rng,
    )
    save_point_cloud(dense_pointcloud_path, visualization_cloud, visualization_colors)
    mode_summaries: dict[str, Any] = {}

    top30_paths = build_mode_output_paths(output_dir, "top30")
    mode_summaries["top30"] = save_grasp_mode_outputs(
        top30_paths,
        top_grasps,
        paths,
        frame_data,
        dense_pointcloud_path,
        args.skip_render,
        args.render_timeout_sec,
        args.render_point_size,
    )
    report_renders = save_report_renders(
        output_dir,
        dense_pointcloud_path,
        paths.rgb_path,
        frame_data.intrinsic,
        top_grasps,
        report_topks=report_topks,
        rgb_only=args.report_rgb_only,
        overlay_edge_thickness=args.overlay_edge_thickness,
        overlay_fill_alpha=args.overlay_fill_alpha,
        overlay_outline_extra=args.overlay_outline_extra,
    )

    collision_summary: dict[str, Any] = {"enabled": False}
    if args.collision_thresh > 0:
        collision_filtered, collision_summary = collision_filter_grasps(
            inference.top_grasps,
            frame_data.raw_cloud,
            args.collision_thresh,
            args.voxel_size_cd,
            args.collision_approach_dist,
        )
        collision_top = collision_filtered[: min(args.topk, len(collision_filtered))]
        mode_summaries["collision"] = save_grasp_mode_outputs(
            build_mode_output_paths(output_dir, "collision"),
            collision_top,
            paths,
            frame_data,
            dense_pointcloud_path,
            args.skip_render,
            args.render_timeout_sec,
            args.render_point_size,
        )

    balanced_summary: dict[str, Any] = {"enabled": False}
    if args.balanced_by_label:
        nms_grasps = np.asarray(inference.top_grasps.grasp_group_array)
        assigned_labels = assign_grasp_center_labels(
            nms_grasps,
            frame_data.intrinsic,
            frame_data.label_image,
        )
        balanced_indices = balance_grasps_by_label(nms_grasps, assigned_labels, args.topk)
        balanced_group = make_grasp_group(nms_grasps[balanced_indices])
        selected_labels = assigned_labels[balanced_indices] if len(balanced_indices) else np.array([], dtype=np.int32)
        balanced_summary = {
            "enabled": True,
            "input_count": int(len(nms_grasps)),
            "selected_count": int(len(balanced_group)),
            "visible_label_distribution": label_distribution(assigned_labels),
            "selected_label_distribution": label_distribution(selected_labels),
        }
        mode_summaries["balanced_by_label"] = save_grasp_mode_outputs(
            build_mode_output_paths(output_dir, "balanced"),
            balanced_group,
            paths,
            frame_data,
            dense_pointcloud_path,
            args.skip_render,
            args.render_timeout_sec,
            args.render_point_size,
        )

    timings = dict(inference.timings_sec)
    timings["load_frame"] = data_time
    timings["total"] = time.perf_counter() - total_started
    summary = {
        "camera": paths.camera,
        "checkpoint_epoch": inference.checkpoint_epoch,
        "checkpoint_path": str(checkpoint_path),
        "dataset_root": str(paths.dataset_root),
        "depth_shape": list(frame_data.depth_shape),
        "device": inference.device,
        "frame": paths.frame,
        "masked_point_count": frame_data.masked_point_count,
        "num_point": args.num_point,
        "outputs": {
            "pointcloud": str(pointcloud_path),
            "pointcloud_dense": str(dense_pointcloud_path),
            "predictions": str(predictions_path),
            "top30_grasps": mode_summaries["top30"]["outputs"]["grasps"],
            "rgb_overlay": mode_summaries["top30"]["outputs"]["rgb_overlay"],
            "render_png": mode_summaries["top30"]["outputs"]["top_render"],
            "report_pngs": report_renders,
        },
        "modes": mode_summaries,
        "prediction_count": int(len(inference.predictions)),
        "collision_filter": collision_summary,
        "balanced_by_label": balanced_summary,
        "render_status": mode_summaries["top30"]["render_status"],
        "report_visualization": {
            "rgb_only": bool(args.report_rgb_only),
            "topks": list(report_topks),
            "overlay_edge_thickness": int(args.overlay_edge_thickness),
            "overlay_fill_alpha": float(args.overlay_fill_alpha),
            "overlay_outline_extra": int(args.overlay_outline_extra),
            "color_encoding": "grasp score: low=blue, high=red, normalized within each top-k overlay",
        },
        "removed_legacy_renders": removed_legacy_renders,
        "sampled_point_count": int(len(frame_data.sampled_cloud)),
        "scene": paths.scene,
        "seed": args.seed,
        "split": paths.split,
        "timings_sec": {key: round(value, 6) for key, value in timings.items()},
        "top_grasp_count": int(len(top_grasps)),
        "valid_depth_count": frame_data.valid_depth_count,
        "visualization_point_count": int(len(visualization_cloud)),
        "visualization_point_limit": args.visualization_point_limit,
        "voxel_size": args.voxel_size,
        "workspace_point_count": frame_data.workspace_point_count,
    }
    save_summary(summary_path, summary)

    print(f"[INFO] checkpoint epoch: {inference.checkpoint_epoch}")
    print(f"[INFO] predictions: {len(inference.predictions)}")
    print(f"[INFO] top grasps: {len(top_grasps)}")
    print(f"[INFO] saved: {predictions_path}")
    print(f"[INFO] saved: {pointcloud_path}")
    print(f"[INFO] saved: {dense_pointcloud_path}")
    for mode_name, mode_summary in mode_summaries.items():
        mode_outputs = mode_summary["outputs"]
        print(f"[INFO] {mode_name} grasps: {mode_summary['count']}")
        print(f"[INFO] saved: {mode_outputs['grasps']}")
        if mode_outputs.get("gripper_mesh"):
            print(f"[INFO] saved: {mode_outputs['gripper_mesh']}")
        if mode_outputs.get("rgb_overlay"):
            print(f"[INFO] saved RGB overlay: {mode_outputs['rgb_overlay']}")
        if mode_outputs.get("top_render"):
            print(f"[INFO] saved top render: {mode_outputs['top_render']}")
    for removed_path in removed_legacy_renders:
        print(f"[INFO] removed legacy render: {removed_path}")
    print(f"[INFO] saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
