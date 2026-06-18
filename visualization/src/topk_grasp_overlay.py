from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from visualization.src.annotations_visualization import (
    AnnotationObject,
    parse_annotations_xml,
)
from visualization.src.grasp_label_visualization import (
    generate_fibonacci_sphere_directions,
    load_grasp_label,
)
from visualization.src.pose_axes_visualization import (
    project_points,
    quaternion_to_rotation_matrix,
    transform_points,
)
from visualization.src.sample_paths import SamplePaths


@dataclass(frozen=True)
class GraspCandidate:
    object_id: int
    point_index: int
    direction_index: int
    angle_index: int
    depth_index: int
    friction: float
    quality: float


def select_top_grasp_candidates(
    scores: np.ndarray,
    topk: int,
    object_id: int = -1,
    min_quality: float | None = None,
    point_priorities: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> list[GraspCandidate]:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {scores.shape}")
    if topk <= 0:
        return []
    if point_priorities is not None and point_priorities.shape != (scores.shape[0],):
        raise ValueError(
            f"Expected point_priorities with shape ({scores.shape[0]},), "
            f"got {point_priorities.shape}"
        )
    if candidate_mask is not None and candidate_mask.shape != scores.shape:
        raise ValueError(
            f"Expected candidate_mask with shape {scores.shape}, got {candidate_mask.shape}"
        )

    valid = scores >= 0
    if candidate_mask is not None:
        valid &= candidate_mask
    valid_indices = np.argwhere(valid)
    candidates: list[GraspCandidate] = []
    for point_index, direction_index, angle_index, depth_index in valid_indices:
        friction = float(scores[point_index, direction_index, angle_index, depth_index])
        quality = float(np.clip(1.1 - friction, 0.0, 1.0))
        if min_quality is not None and quality < min_quality:
            continue
        candidates.append(
            GraspCandidate(
                object_id=object_id,
                point_index=int(point_index),
                direction_index=int(direction_index),
                angle_index=int(angle_index),
                depth_index=int(depth_index),
                friction=friction,
                quality=quality,
            )
        )

    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.quality,
            float(point_priorities[candidate.point_index])
            if point_priorities is not None
            else 0.0,
        ),
        reverse=True,
    )[:topk]


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction.astype(np.float64)
    direction = direction / np.linalg.norm(direction)
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, helper))) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    axis_u = np.cross(direction, helper)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(direction, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    return axis_u, axis_v


def build_gripper_control_points(
    points: np.ndarray,
    offsets: np.ndarray,
    candidate: GraspCandidate,
    direction: np.ndarray,
    finger_length: float = 0.035,
) -> np.ndarray:
    point = points[candidate.point_index].astype(np.float64)
    angle, depth, width = offsets[
        candidate.point_index,
        candidate.direction_index,
        candidate.angle_index,
        candidate.depth_index,
    ].astype(np.float64)
    direction = direction.astype(np.float64)
    direction = direction / np.linalg.norm(direction)
    axis_u, axis_v = _orthonormal_basis(direction)
    opening_axis = math.cos(float(angle)) * axis_u + math.sin(float(angle)) * axis_v
    opening_axis = opening_axis / np.linalg.norm(opening_axis)

    center = point + direction * float(depth)
    half_width = max(float(width), 0.01) / 2.0
    left_tip = center + opening_axis * half_width
    right_tip = center - opening_axis * half_width
    wrist = center - direction * float(finger_length)
    palm_left = wrist + opening_axis * half_width
    palm_right = wrist - opening_axis * half_width
    return np.vstack([center, left_tip, right_tip, wrist, palm_left, palm_right]).astype(
        np.float64
    )


def object_points_to_world(points_object: np.ndarray, obj: AnnotationObject) -> np.ndarray:
    rotation = quaternion_to_rotation_matrix(obj.orientation)
    translation = np.array(obj.position, dtype=np.float64)
    return points_object.astype(np.float64) @ rotation.T + translation


def _object_color(index: int) -> tuple[int, int, int]:
    palette = [
        (230, 57, 70),
        (42, 157, 143),
        (38, 70, 83),
        (244, 162, 97),
        (29, 78, 216),
        (123, 44, 191),
        (46, 125, 50),
        (255, 193, 7),
        (0, 150, 199),
    ]
    color_rgb = palette[index % len(palette)]
    return (color_rgb[2], color_rgb[1], color_rgb[0])


def _draw_projected_gripper(
    image_bgr: np.ndarray,
    pixels: np.ndarray,
    color_bgr: tuple[int, int, int],
    quality: float,
) -> None:
    pts = np.rint(pixels).astype(int)
    center, left_tip, right_tip, wrist, palm_left, palm_right = [tuple(p) for p in pts]
    thickness = 2 if quality < 0.85 else 3
    cv2.line(image_bgr, center, left_tip, color_bgr, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, center, right_tip, color_bgr, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, palm_left, left_tip, color_bgr, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, palm_right, right_tip, color_bgr, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, palm_left, palm_right, color_bgr, thickness, cv2.LINE_AA)
    cv2.line(image_bgr, wrist, center, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image_bgr, center, 3, (255, 255, 255), -1, cv2.LINE_AA)


def _project_gripper(
    control_points_object: np.ndarray,
    obj: AnnotationObject,
    camera_pose: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray | None:
    world = object_points_to_world(control_points_object, obj)
    camera = transform_points(world, camera_pose)
    if np.any(camera[:, 2] <= 0):
        return None
    return project_points(camera, intrinsic)


def _point_visibility_priorities(
    points_object: np.ndarray,
    obj: AnnotationObject,
    camera_pose: np.ndarray,
    intrinsic: np.ndarray,
    label_image: np.ndarray | None,
) -> np.ndarray | None:
    if label_image is None:
        return None

    points_world = object_points_to_world(points_object, obj)
    points_camera = transform_points(points_world, camera_pose)
    priorities = np.zeros(points_object.shape[0], dtype=np.float32)
    in_front = points_camera[:, 2] > 0
    if not np.any(in_front):
        return priorities

    pixels = project_points(points_camera[in_front], intrinsic)
    rounded = np.rint(pixels).astype(int)
    valid_indices = np.flatnonzero(in_front)
    height, width = label_image.shape[:2]
    expected_label = obj.obj_id + 1
    ys, xs = np.where(label_image == expected_label)
    if len(xs) == 0:
        return priorities

    mask_center = np.array([xs.mean(), ys.mean()], dtype=np.float64)
    mask_radius = max(
        float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min())) / 2.0,
        1.0,
    )
    for source_index, (x, y), pixel in zip(valid_indices, rounded, pixels, strict=True):
        if not (0 <= x < width and 0 <= y < height):
            continue
        distance_score = max(
            0.0,
            1.0 - float(np.linalg.norm(pixel - mask_center)) / (mask_radius * 1.5),
        )
        if int(label_image[y, x]) == expected_label:
            priorities[source_index] = 2.0 + distance_score
        else:
            priorities[source_index] = distance_score
    return priorities


def _draw_legend(
    image_bgr: np.ndarray,
    labels: list[tuple[int, str, tuple[int, int, int], int]],
) -> None:
    if not labels:
        return
    x0, y0 = 14, 16
    row_h = 22
    width = 310
    height = row_h * len(labels) + 18
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0 - 8, y0 - 8), (x0 + width, y0 + height), (20, 24, 27), -1)
    cv2.addWeighted(overlay, 0.62, image_bgr, 0.38, 0, image_bgr)
    for index, (obj_id, name, color_bgr, count) in enumerate(labels):
        y = y0 + index * row_h
        cv2.line(image_bgr, (x0, y + 7), (x0 + 22, y + 7), color_bgr, 4, cv2.LINE_AA)
        text = f"{obj_id}: {name.removesuffix('.ply')}  top-{count}"
        cv2.putText(
            image_bgr,
            text,
            (x0 + 30, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )


def write_topk_grasp_overlay(
    rgb_path: str | Path,
    annotations_path: str | Path,
    intrinsic_path: str | Path,
    camera_poses_path: str | Path,
    grasp_label_root: str | Path,
    output_path: str | Path,
    label_path: str | Path | None = None,
    topk_per_object: int = 3,
    frame_index: int = 0,
    min_quality: float | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(rgb_path) as rgb_image:
        rgb = np.array(rgb_image.convert("RGB"))
    image_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    height, width = image_bgr.shape[:2]

    objects = parse_annotations_xml(annotations_path)
    intrinsic = np.load(intrinsic_path)
    camera_pose = np.load(camera_poses_path)[frame_index]
    label_image = None
    if label_path is not None and Path(label_path).exists():
        label_image = np.array(Image.open(label_path))
    grasp_label_root = Path(grasp_label_root)
    legend: list[tuple[int, str, tuple[int, int, int], int]] = []

    for object_index, obj in enumerate(objects):
        grasp_label_path = grasp_label_root / f"{obj.obj_id:03d}_labels.npz"
        if not grasp_label_path.exists():
            continue
        data = load_grasp_label(grasp_label_path)
        point_priorities = _point_visibility_priorities(
            data.points,
            obj,
            camera_pose,
            intrinsic,
            label_image,
        )
        candidates = select_top_grasp_candidates(
            data.scores,
            topk=topk_per_object,
            object_id=obj.obj_id,
            min_quality=min_quality,
            point_priorities=point_priorities,
        )
        if not candidates:
            continue
        directions = generate_fibonacci_sphere_directions(data.scores.shape[1])
        color_bgr = _object_color(object_index)
        drawn = 0
        for candidate in candidates:
            control_points = build_gripper_control_points(
                data.points,
                data.offsets,
                candidate,
                directions[candidate.direction_index],
            )
            pixels = _project_gripper(control_points, obj, camera_pose, intrinsic)
            if pixels is None:
                continue
            if not np.all(
                (pixels[:, 0] >= -width * 0.2)
                & (pixels[:, 0] <= width * 1.2)
                & (pixels[:, 1] >= -height * 0.2)
                & (pixels[:, 1] <= height * 1.2)
            ):
                continue
            _draw_projected_gripper(image_bgr, pixels, color_bgr, candidate.quality)
            drawn += 1
        if drawn:
            legend.append((obj.obj_id, obj.obj_name, color_bgr, drawn))

    _draw_legend(image_bgr, legend)
    output_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(output_rgb).save(output_path)
    return output_path


def export_sample_topk_grasp_overlay(
    paths: SamplePaths,
    topk_per_object: int = 3,
    frame_index: int = 0,
    min_quality: float | None = None,
) -> Path:
    return write_topk_grasp_overlay(
        rgb_path=paths.rgb_path,
        annotations_path=paths.annotations_path,
        intrinsic_path=paths.camK_path,
        camera_poses_path=paths.camera_poses_path,
        grasp_label_root=paths.data_root / "grasp_label",
        output_path=paths.output_topk_grasp_overlay_path,
        label_path=paths.label_path,
        topk_per_object=topk_per_object,
        frame_index=frame_index,
        min_quality=min_quality,
    )
