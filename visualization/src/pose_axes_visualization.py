from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visualization.src.annotations_visualization import AnnotationObject, parse_annotations_xml
from visualization.src.sample_paths import SamplePaths


def quaternion_to_rotation_matrix(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    qx, qy, qz, qw = quaternion
    norm = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if norm == 0.0:
        raise ValueError("Quaternion norm must be non-zero")

    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, camera_pose: np.ndarray) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if camera_pose.shape != (4, 4):
        raise ValueError(f"Expected camera pose with shape (4, 4), got {camera_pose.shape}")

    world_to_camera = np.linalg.inv(camera_pose)
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (world_to_camera @ homogeneous.T).T
    return transformed[:, :3]


def project_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    if points_camera.ndim != 2 or points_camera.shape[1] != 3:
        raise ValueError(
            f"Expected camera points with shape (N, 3), got {points_camera.shape}"
        )
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected intrinsic with shape (3, 3), got {intrinsic.shape}")
    if np.any(points_camera[:, 2] <= 0):
        raise ValueError("All projected points must have positive z in camera space")

    x = intrinsic[0, 0] * points_camera[:, 0] / points_camera[:, 2] + intrinsic[0, 2]
    y = intrinsic[1, 1] * points_camera[:, 1] / points_camera[:, 2] + intrinsic[1, 2]
    return np.stack([x, y], axis=1)


def _axis_points_for_object(
    obj: AnnotationObject, axis_length: float
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    origin = np.array(obj.position, dtype=np.float64)
    rotation = quaternion_to_rotation_matrix(obj.orientation)
    endpoints = origin + rotation @ (np.eye(3, dtype=np.float64) * axis_length)
    points = np.vstack([origin, endpoints])

    # OpenCV works in BGR order.
    colors = [(36, 36, 230), (65, 170, 40), (235, 105, 35)]
    return points, colors


def _draw_axis_tripod(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    colors_bgr: list[tuple[int, int, int]],
) -> None:
    rounded = np.rint(pixels).astype(int)
    origin = tuple(rounded[0])

    for endpoint, color in zip(rounded[1:], colors_bgr, strict=True):
        endpoint_tuple = tuple(endpoint)
        cv2.line(image_bgr, origin, endpoint_tuple, color, thickness=5, lineType=cv2.LINE_AA)
        cv2.circle(image_bgr, endpoint_tuple, 7, color, thickness=-1, lineType=cv2.LINE_AA)

    cv2.circle(image_bgr, origin, 7, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(image_bgr, origin, 7, (30, 30, 30), thickness=2, lineType=cv2.LINE_AA)


def write_pose_axes_overlay(
    rgb_path: str | Path,
    annotations_path: str | Path,
    intrinsic_path: str | Path,
    camera_poses_path: str | Path,
    output_path: str | Path,
    frame_index: int = 0,
    axis_length: float = 0.055,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(rgb_path) as rgb_image:
        rgb = np.array(rgb_image.convert("RGB"))
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    intrinsic = np.load(intrinsic_path)
    camera_poses = np.load(camera_poses_path)
    camera_pose = camera_poses[frame_index]
    objects = parse_annotations_xml(annotations_path)

    height, width = image_bgr.shape[:2]
    for obj in objects:
        points_world, colors_bgr = _axis_points_for_object(obj, axis_length)
        points_camera = transform_points(points_world, camera_pose)
        if np.any(points_camera[:, 2] <= 0):
            continue
        pixels = project_points(points_camera, intrinsic)
        if np.all(
            (pixels[:, 0] >= -width * 0.2)
            & (pixels[:, 0] <= width * 1.2)
            & (pixels[:, 1] >= -height * 0.2)
            & (pixels[:, 1] <= height * 1.2)
        ):
            _draw_axis_tripod(image_bgr, pixels, colors_bgr)

    output_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(output_rgb).save(output_path)
    return output_path


def export_sample_pose_axes_overlay(paths: SamplePaths, frame_index: int = 0) -> Path:
    return write_pose_axes_overlay(
        paths.rgb_path,
        paths.annotations_path,
        paths.camK_path,
        paths.camera_poses_path,
        paths.output_pose_axes_overlay_path,
        frame_index=frame_index,
    )
