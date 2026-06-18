from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import numpy as np
from matplotlib import pyplot as plt

from visualization.src.grasp_label_visualization import (
    CJK_FONT,
    GraspLabelData,
    _save_rgb_figure,
    _set_equal_axes,
    generate_fibonacci_sphere_directions,
    load_grasp_label,
    load_ply_vertices,
)
from visualization.src.topk_grasp_overlay import GraspCandidate


@dataclass(frozen=True)
class AntipodalContactEstimate:
    center: np.ndarray
    left: np.ndarray
    right: np.ndarray
    approach_direction: np.ndarray
    opening_axis: np.ndarray
    angle: float
    depth: float
    width: float


def _unit_vector(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = _unit_vector(direction, "direction")
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, helper))) > 0.92:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    axis_u = _unit_vector(np.cross(direction, helper), "axis_u")
    axis_v = _unit_vector(np.cross(direction, axis_u), "axis_v")
    return axis_u, axis_v


def _nearest_model_point(model_points: np.ndarray, query: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(model_points.astype(np.float64) - query[None, :], axis=1)
    return model_points[int(np.argmin(distances))].astype(np.float64)


def estimate_contact_points(
    points: np.ndarray,
    offsets: np.ndarray,
    candidate: GraspCandidate,
    direction: np.ndarray,
    object_model_points: np.ndarray | None = None,
) -> AntipodalContactEstimate:
    if offsets.ndim != 5 or offsets.shape[-1] != 3:
        raise ValueError(
            f"Expected offsets with shape (N, V, A, D, 3), got {offsets.shape}"
        )

    point = points[candidate.point_index].astype(np.float64)
    angle, depth, width = offsets[
        candidate.point_index,
        candidate.direction_index,
        candidate.angle_index,
        candidate.depth_index,
    ].astype(np.float64)
    approach_direction = _unit_vector(direction, "direction")
    axis_u, axis_v = _orthonormal_basis(approach_direction)
    opening_axis = (
        math.cos(float(angle)) * axis_u + math.sin(float(angle)) * axis_v
    )
    opening_axis = _unit_vector(opening_axis, "opening_axis")

    center = point + approach_direction * float(depth)
    half_width = max(float(width), 0.01) / 2.0
    left_guess = center - opening_axis * half_width
    right_guess = center + opening_axis * half_width
    left = left_guess
    right = right_guess

    if object_model_points is not None and len(object_model_points) > 0:
        model = object_model_points.astype(np.float64)
        left_candidate = _nearest_model_point(model, left_guess)
        right_candidate = _nearest_model_point(model, right_guess)
        if np.linalg.norm(left_candidate - right_candidate) > half_width * 0.45:
            left = left_candidate
            right = right_candidate

    return AntipodalContactEstimate(
        center=center.astype(np.float64),
        left=left.astype(np.float64),
        right=right.astype(np.float64),
        approach_direction=approach_direction.astype(np.float64),
        opening_axis=opening_axis.astype(np.float64),
        angle=float(angle),
        depth=float(depth),
        width=float(width),
    )


def estimate_surface_normal(
    points: np.ndarray,
    query_point: np.ndarray,
    orient_toward: np.ndarray | None = None,
    neighbor_count: int = 48,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError("At least three points are required to estimate a normal")

    query = np.asarray(query_point, dtype=np.float64)
    neighbor_count = max(3, min(int(neighbor_count), points.shape[0]))
    distances = np.linalg.norm(points.astype(np.float64) - query[None, :], axis=1)
    indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
    neighbors = points[indices].astype(np.float64)
    centered = neighbors - neighbors.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(neighbor_count - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = _unit_vector(eigenvectors[:, int(np.argmin(eigenvalues))], "normal")

    if orient_toward is not None:
        orient = _unit_vector(orient_toward, "orient_toward")
        if float(np.dot(normal, orient)) < 0.0:
            normal = -normal
    return normal.astype(np.float64)


def compute_antipodal_score(
    left: np.ndarray,
    right: np.ndarray,
    left_normal: np.ndarray,
    right_normal: np.ndarray,
    friction_coefficient: float = 0.5,
) -> float:
    contact_line = _unit_vector(np.asarray(right, dtype=np.float64) - left, "line")
    left_normal = _unit_vector(left_normal, "left_normal")
    right_normal = _unit_vector(right_normal, "right_normal")

    left_alignment = max(float(np.dot(left_normal, contact_line)), 0.0)
    right_alignment = max(float(np.dot(right_normal, -contact_line)), 0.0)
    opposing = max(float(-np.dot(left_normal, right_normal)), 0.0)
    cone_threshold = math.cos(math.atan(max(float(friction_coefficient), 1e-6)))
    cone_score = min(left_alignment, right_alignment) / cone_threshold
    cone_score = float(np.clip(cone_score, 0.0, 1.0))
    score = 0.55 * min(left_alignment, right_alignment) + 0.25 * opposing + 0.20 * cone_score
    return float(np.clip(score, 0.0, 1.0))


def select_representative_antipodal_candidate(
    data: GraspLabelData,
    object_model_points: np.ndarray,
    object_id: int,
    directions: np.ndarray | None = None,
    top_pool_size: int = 800,
) -> GraspCandidate:
    if data.scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {data.scores.shape}")
    if directions is None:
        directions = generate_fibonacci_sphere_directions(data.scores.shape[1])
    if directions.shape != (data.scores.shape[1], 3):
        raise ValueError(
            f"Expected directions with shape ({data.scores.shape[1]}, 3), got {directions.shape}"
        )

    valid = data.scores >= 0.0
    valid_flat = np.flatnonzero(valid.ravel())
    if valid_flat.size == 0:
        raise ValueError("No valid grasp candidates in scores")

    quality = np.where(valid, 1.1 - data.scores, -np.inf).ravel()
    pool_size = min(max(int(top_pool_size), 1), valid_flat.size)
    if pool_size < valid_flat.size:
        candidate_positions = np.argpartition(quality[valid_flat], -pool_size)[-pool_size:]
        pool = valid_flat[candidate_positions]
    else:
        pool = valid_flat
    pool = pool[np.argsort(quality[pool])[::-1]]

    reference_points = (
        object_model_points
        if object_model_points is not None and len(object_model_points) > 0
        else data.points
    )
    span = max(float(np.ptp(reference_points, axis=0).max()), 1e-6)
    centroid = reference_points.astype(np.float64).mean(axis=0)
    best_candidate: GraspCandidate | None = None
    best_rank: tuple[float, float, float, float] | None = None

    for flat_index in pool:
        point_index, direction_index, angle_index, depth_index = np.unravel_index(
            int(flat_index),
            data.scores.shape,
        )
        friction = float(data.scores[point_index, direction_index, angle_index, depth_index])
        candidate = GraspCandidate(
            object_id=object_id,
            point_index=int(point_index),
            direction_index=int(direction_index),
            angle_index=int(angle_index),
            depth_index=int(depth_index),
            friction=friction,
            quality=float(np.clip(1.1 - friction, 0.0, 1.0)),
        )
        contact = estimate_contact_points(
            data.points,
            data.offsets,
            candidate,
            directions[direction_index],
            object_model_points=object_model_points,
        )
        separation = float(np.linalg.norm(contact.right - contact.left))
        separation_score = min(separation / span, 1.0)
        center_score = 1.0 - min(float(np.linalg.norm(contact.center - centroid)) / span, 1.0)
        antipodal_score = 0.0
        if object_model_points is not None and len(object_model_points) >= 3 and separation > 1e-8:
            try:
                contact_line = _unit_vector(contact.right - contact.left, "contact_line")
                left_normal = estimate_surface_normal(
                    object_model_points,
                    contact.left,
                    orient_toward=contact_line,
                )
                right_normal = estimate_surface_normal(
                    object_model_points,
                    contact.right,
                    orient_toward=-contact_line,
                )
                antipodal_score = compute_antipodal_score(
                    contact.left,
                    contact.right,
                    left_normal,
                    right_normal,
                    friction_coefficient=max(candidate.friction, 1e-3),
                )
            except ValueError:
                antipodal_score = 0.0

        rank = (
            round(candidate.quality, 6),
            round(antipodal_score, 6),
            round(separation_score, 6),
            round(center_score, 6),
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidate = candidate

    if best_candidate is None:
        raise ValueError("Unable to select a representative antipodal candidate")
    return best_candidate


def _compute_candidate_analysis(
    data: GraspLabelData,
    object_model_points: np.ndarray,
    object_id: int,
) -> tuple[GraspCandidate, AntipodalContactEstimate, np.ndarray, np.ndarray, float]:
    directions = generate_fibonacci_sphere_directions(data.scores.shape[1])
    candidate = select_representative_antipodal_candidate(
        data,
        object_model_points,
        object_id=object_id,
        directions=directions,
    )
    contact = estimate_contact_points(
        data.points,
        data.offsets,
        candidate,
        directions[candidate.direction_index],
        object_model_points=object_model_points,
    )
    contact_line = _unit_vector(contact.right - contact.left, "contact_line")
    left_normal = estimate_surface_normal(
        object_model_points,
        contact.left,
        orient_toward=contact_line,
    )
    right_normal = estimate_surface_normal(
        object_model_points,
        contact.right,
        orient_toward=-contact_line,
    )
    score = compute_antipodal_score(
        contact.left,
        contact.right,
        left_normal,
        right_normal,
        friction_coefficient=max(candidate.friction, 1e-3),
    )
    return candidate, contact, left_normal, right_normal, score


def _scatter_object(ax, object_model_points: np.ndarray, alpha: float = 0.16) -> None:
    ax.scatter(
        object_model_points[:, 0],
        object_model_points[:, 1],
        object_model_points[:, 2],
        c="#9aa5aa",
        s=3.2,
        linewidths=0,
        alpha=alpha,
    )


def _draw_contact_pair(
    ax,
    contact: AntipodalContactEstimate,
    left_normal: np.ndarray,
    right_normal: np.ndarray,
    color: str,
    normal_length: float,
    label: str,
) -> None:
    ax.scatter(
        [contact.left[0], contact.right[0]],
        [contact.left[1], contact.right[1]],
        [contact.left[2], contact.right[2]],
        c=color,
        s=70,
        edgecolors="#263238",
        linewidths=0.7,
        depthshade=False,
        label=label,
    )
    ax.plot(
        [contact.left[0], contact.right[0]],
        [contact.left[1], contact.right[1]],
        [contact.left[2], contact.right[2]],
        color=color,
        linewidth=2.4,
        alpha=0.95,
    )
    ax.quiver(
        contact.left[0],
        contact.left[1],
        contact.left[2],
        left_normal[0],
        left_normal[1],
        left_normal[2],
        length=normal_length,
        color="#1565c0",
        linewidth=1.8,
        arrow_length_ratio=0.22,
    )
    ax.quiver(
        contact.right[0],
        contact.right[1],
        contact.right[2],
        right_normal[0],
        right_normal[1],
        right_normal[2],
        length=normal_length,
        color="#1565c0",
        linewidth=1.8,
        arrow_length_ratio=0.22,
    )
    ax.quiver(
        contact.center[0],
        contact.center[1],
        contact.center[2],
        contact.approach_direction[0],
        contact.approach_direction[1],
        contact.approach_direction[2],
        length=normal_length * 1.15,
        color="#ef6c00",
        linewidth=2.0,
        arrow_length_ratio=0.2,
    )
    ax.quiver(
        contact.center[0],
        contact.center[1],
        contact.center[2],
        contact.opening_axis[0],
        contact.opening_axis[1],
        contact.opening_axis[2],
        length=normal_length,
        color="#00897b",
        linewidth=2.0,
        arrow_length_ratio=0.2,
    )


def _draw_friction_cone(
    ax,
    contact_point: np.ndarray,
    normal: np.ndarray,
    length: float,
    friction_coefficient: float,
    color: str = "#42a5f5",
) -> None:
    normal = _unit_vector(normal, "normal")
    axis_u, axis_v = _orthonormal_basis(normal)
    radius = max(float(friction_coefficient), 0.28) * length
    tip = contact_point + normal * length
    for theta in np.linspace(0.0, 2.0 * np.pi, 9, endpoint=False):
        edge = tip + radius * (math.cos(float(theta)) * axis_u + math.sin(float(theta)) * axis_v)
        ax.plot(
            [contact_point[0], edge[0]],
            [contact_point[1], edge[1]],
            [contact_point[2], edge[2]],
            color=color,
            linewidth=0.7,
            alpha=0.34,
        )


def _set_common_3d_style(
    ax,
    object_model_points: np.ndarray,
    title: str,
    elev: float = 24.0,
    azim: float = -58.0,
) -> None:
    ax.set_title(title, fontsize=12.5, weight="bold", fontproperties=CJK_FONT, pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, linestyle="--", color="#d8d2c5", alpha=0.42)
    _set_equal_axes(ax, object_model_points)


def _object_display_name(object_id: int, object_name: str | None) -> str:
    if not object_name:
        return f"grasp_label/{object_id:03d}_object"
    stem = Path(object_name).stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return f"grasp_label/{object_id:03d}_{parts[1]} (annotation: {stem})"
    return f"grasp_label/{object_id:03d} ({stem})"


def _object_short_name(object_id: int, object_name: str | None) -> str:
    if not object_name:
        return f"{object_id:03d}_object"
    stem = Path(object_name).stem
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        return f"{object_id:03d}_{parts[1]}"
    return f"{object_id:03d}_{stem}"


def _write_concept_plot(
    object_model_points: np.ndarray,
    contact: AntipodalContactEstimate,
    left_normal: np.ndarray,
    right_normal: np.ndarray,
    score: float,
    output_path: Path,
    object_id: int,
    object_name: str | None,
    friction: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    span = float(np.ptp(object_model_points, axis=0).max())
    normal_length = max(span * 0.18, 0.016)

    fig = plt.figure(figsize=(11.2, 8.7))
    fig.patch.set_facecolor("#fbfbf7")
    ax = fig.add_subplot(111, projection="3d")
    _scatter_object(ax, object_model_points, alpha=0.18)
    _draw_contact_pair(
        ax,
        contact,
        left_normal,
        right_normal,
        color="#2e7d32",
        normal_length=normal_length,
        label="对踵接触点",
    )
    _draw_friction_cone(ax, contact.left, left_normal, normal_length * 0.9, friction)
    _draw_friction_cone(ax, contact.right, right_normal, normal_length * 0.9, friction)
    _set_common_3d_style(
        ax,
        np.vstack([object_model_points, contact.left, contact.right, contact.center]),
        title=f"真实物体上的对踵概念: {_object_display_name(object_id, object_name)}",
    )
    ax.legend(loc="upper left", prop=CJK_FONT)
    fig.text(
        0.06,
        0.055,
        (
            "读图: 绿色两点是夹爪两指的接触点；绿色连线是两接触点之间的开合方向；"
            "蓝色箭头是局部表面法向；浅蓝锥体表示简化摩擦锥。"
        ),
        fontsize=10.2,
        color="#37474f",
        fontproperties=CJK_FONT,
    )
    fig.text(
        0.06,
        0.025,
        f"解释性对踵评分 {score:.2f}: 两侧法向越沿接触线相向，越接近 1。",
        fontsize=10.2,
        color="#37474f",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def _write_positive_negative_plot(
    object_model_points: np.ndarray,
    contact: AntipodalContactEstimate,
    left_normal: np.ndarray,
    right_normal: np.ndarray,
    output_path: Path,
    object_id: int,
    object_name: str | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    span = float(np.ptp(object_model_points, axis=0).max())
    normal_length = max(span * 0.16, 0.014)
    contact_line = _unit_vector(contact.right - contact.left, "contact_line")
    bad_left_normal = _unit_vector(contact.approach_direction, "bad_left")
    bad_right_normal = bad_left_normal.copy()
    bad_score = compute_antipodal_score(
        contact.left,
        contact.right,
        bad_left_normal,
        bad_right_normal,
        friction_coefficient=0.5,
    )
    good_score = compute_antipodal_score(
        contact.left,
        contact.right,
        left_normal,
        right_normal,
        friction_coefficient=0.5,
    )

    fig = plt.figure(figsize=(15.4, 7.0))
    fig.patch.set_facecolor("#fbfbf7")
    axes = [fig.add_subplot(1, 2, 1, projection="3d"), fig.add_subplot(1, 2, 2, projection="3d")]
    for ax in axes:
        _scatter_object(ax, object_model_points, alpha=0.15)

    _draw_contact_pair(
        axes[0],
        contact,
        left_normal,
        right_normal,
        color="#2e7d32",
        normal_length=normal_length,
        label="正例",
    )
    _set_common_3d_style(
        axes[0],
        np.vstack([object_model_points, contact.left, contact.right]),
        title=f"正例: 两侧法向沿接触线相向 score={good_score:.2f}",
    )

    _draw_contact_pair(
        axes[1],
        contact,
        bad_left_normal,
        bad_right_normal,
        color="#c62828",
        normal_length=normal_length,
        label="反例",
    )
    axes[1].quiver(
        contact.center[0],
        contact.center[1],
        contact.center[2],
        contact_line[0],
        contact_line[1],
        contact_line[2],
        length=normal_length,
        color="#455a64",
        linewidth=1.6,
        arrow_length_ratio=0.2,
    )
    _set_common_3d_style(
        axes[1],
        np.vstack([object_model_points, contact.left, contact.right]),
        title=f"反例: 法向不夹住接触线 score={bad_score:.2f}",
    )

    fig.suptitle(
        f"对踵/非对踵对比都落在真实 {_object_display_name(object_id, object_name)} 模型上",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    fig.text(
        0.05,
        0.035,
        "更容易理解的说法: 好抓取像两根手指从物体两侧互相顶住；坏抓取虽然也有两个点，但力的方向没有形成稳定夹持。",
        fontsize=10.2,
        color="#37474f",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def _write_top_grasp_analysis_plot(
    object_model_points: np.ndarray,
    candidate: GraspCandidate,
    contact: AntipodalContactEstimate,
    left_normal: np.ndarray,
    right_normal: np.ndarray,
    score: float,
    output_path: Path,
    object_id: int,
    object_name: str | None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    span = float(np.ptp(object_model_points, axis=0).max())
    normal_length = max(span * 0.18, 0.016)

    fig = plt.figure(figsize=(13.8, 8.2))
    fig.patch.set_facecolor("#fbfbf7")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.9], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0], projection="3d")
    info_ax = fig.add_subplot(grid[0, 1])
    info_ax.axis("off")

    _scatter_object(ax, object_model_points, alpha=0.17)
    _draw_contact_pair(
        ax,
        contact,
        left_normal,
        right_normal,
        color="#00695c",
        normal_length=normal_length,
        label="Top grasp",
    )
    _set_common_3d_style(
        ax,
        np.vstack([object_model_points, contact.left, contact.right, contact.center]),
        title="Top 抓取候选的对踵解释",
    )

    rows = [
        ("object", _object_short_name(object_id, object_name)),
        ("point index", str(candidate.point_index)),
        ("direction index", str(candidate.direction_index)),
        ("angle index", str(candidate.angle_index)),
        ("depth index", str(candidate.depth_index)),
        ("offset angle", f"{contact.angle:.4f} rad"),
        ("offset depth", f"{contact.depth:.4f} m"),
        ("offset width", f"{contact.width:.4f} m"),
        ("friction score", f"{candidate.friction:.4f}"),
        ("quality", f"{candidate.quality:.4f}"),
        ("antipodal score", f"{score:.4f}"),
    ]
    table = info_ax.table(
        cellText=rows,
        colLabels=["字段", "数值"],
        colWidths=[0.42, 0.48],
        cellLoc="left",
        bbox=[0.04, 0.25, 0.92, 0.66],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6d1c4")
        cell.set_linewidth(0.8)
        cell.PAD = 0.045
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold", fontproperties=CJK_FONT)
        elif row % 2 == 0:
            cell.set_facecolor("#f1efe7")
            cell.set_text_props(fontproperties=CJK_FONT)
        else:
            cell.set_facecolor("#fffdfa")
            cell.set_text_props(fontproperties=CJK_FONT)
    info_ax.set_title(
        "候选参数",
        fontsize=14,
        weight="bold",
        fontproperties=CJK_FONT,
        pad=8,
    )
    info_ax.text(
        0.05,
        0.12,
        "橙色: approach direction\n绿色: opening axis\n蓝色: 接触点局部法向",
        fontsize=10.4,
        color="#37474f",
        fontproperties=CJK_FONT,
        transform=info_ax.transAxes,
    )
    fig.suptitle(
        "从 grasp_label 里选出的高质量候选，拆解成对踵接触关系",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def write_antipodal_visualizations(
    grasp_label_path: str | Path,
    object_model_path: str | Path,
    output_concept_path: str | Path,
    output_positive_negative_path: str | Path,
    output_top_grasp_analysis_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    max_model_points: int | None = 9000,
) -> tuple[Path, Path, Path]:
    data = load_grasp_label(grasp_label_path)
    object_model_points = load_ply_vertices(object_model_path, max_points=max_model_points)
    if object_model_points is None or len(object_model_points) == 0:
        object_model_points = data.points

    candidate, contact, left_normal, right_normal, score = _compute_candidate_analysis(
        data,
        object_model_points,
        object_id=object_id,
    )
    concept = _write_concept_plot(
        object_model_points,
        contact,
        left_normal,
        right_normal,
        score,
        Path(output_concept_path),
        object_id=object_id,
        object_name=object_name,
        friction=max(candidate.friction, 1e-3),
    )
    positive_negative = _write_positive_negative_plot(
        object_model_points,
        contact,
        left_normal,
        right_normal,
        Path(output_positive_negative_path),
        object_id=object_id,
        object_name=object_name,
    )
    top_grasp = _write_top_grasp_analysis_plot(
        object_model_points,
        candidate,
        contact,
        left_normal,
        right_normal,
        score,
        Path(output_top_grasp_analysis_path),
        object_id=object_id,
        object_name=object_name,
    )
    return concept, positive_negative, top_grasp


def export_antipodal_visualization(
    paths,
    object_id: int,
    object_name: str | None = None,
    max_model_points: int | None = 9000,
) -> tuple[Path, Path, Path]:
    return write_antipodal_visualizations(
        grasp_label_path=paths.grasp_label_path,
        object_model_path=paths.grasp_object_model_path,
        output_concept_path=paths.output_antipodal_concept_path,
        output_positive_negative_path=paths.output_antipodal_positive_negative_path,
        output_top_grasp_analysis_path=paths.output_antipodal_top_grasp_analysis_path,
        object_id=object_id,
        object_name=object_name,
        max_model_points=max_model_points,
    )
