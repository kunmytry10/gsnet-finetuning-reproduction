from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image


CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CJK_FONT = (
    font_manager.FontProperties(fname=str(CJK_FONT_PATH))
    if CJK_FONT_PATH.exists()
    else font_manager.FontProperties()
)


@dataclass(frozen=True)
class GraspLabelData:
    points: np.ndarray
    offsets: np.ndarray
    scores: np.ndarray
    collision: np.ndarray


@dataclass(frozen=True)
class PointGraspness:
    valid_counts: np.ndarray
    valid_ratios: np.ndarray
    best_friction: np.ndarray
    best_quality: np.ndarray


@dataclass(frozen=True)
class DirectionQuality:
    valid_counts: np.ndarray
    best_friction: np.ndarray
    best_quality: np.ndarray


_PLY_PROPERTY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_grasp_label(grasp_label_path: str | Path) -> GraspLabelData:
    grasp_label_path = Path(grasp_label_path)
    if not grasp_label_path.exists():
        raise FileNotFoundError(f"Grasp label file not found: {grasp_label_path}")

    data = np.load(grasp_label_path)
    return GraspLabelData(
        points=data["points"].astype(np.float32),
        offsets=data["offsets"].astype(np.float32),
        scores=data["scores"].astype(np.float32),
        collision=data["collision"].astype(bool),
    )


def load_ply_vertices(
    ply_path: str | Path,
    max_points: int | None = 12000,
) -> np.ndarray | None:
    ply_path = Path(ply_path)
    if not ply_path.exists():
        return None

    with ply_path.open("rb") as file:
        format_name = None
        vertex_count = 0
        vertex_properties: list[tuple[str, str]] = []
        in_vertex_element = False
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {ply_path}")
            text = line.decode("ascii").strip()
            if text.startswith("format "):
                format_name = text.split()[1]
            elif text.startswith("element vertex "):
                vertex_count = int(text.split()[2])
                in_vertex_element = True
            elif text.startswith("element "):
                in_vertex_element = False
            elif in_vertex_element and text.startswith("property "):
                parts = text.split()
                if len(parts) == 3:
                    vertex_properties.append((parts[2], parts[1]))
            elif text == "end_header":
                break

        if format_name != "binary_little_endian":
            raise ValueError(f"Only binary_little_endian PLY is supported: {ply_path}")
        if vertex_count <= 0:
            return np.empty((0, 3), dtype=np.float32)

        dtype = np.dtype(
            [
                (name, _PLY_PROPERTY_DTYPES[property_type])
                for name, property_type in vertex_properties
            ]
        )
        vertices = np.fromfile(file, dtype=dtype, count=vertex_count)

    coords = np.column_stack(
        [vertices["x"], vertices["y"], vertices["z"]]
    ).astype(np.float32)
    if max_points is not None and coords.shape[0] > max_points:
        indices = np.linspace(0, coords.shape[0] - 1, max_points, dtype=np.int64)
        coords = coords[indices]
    return coords


def compute_point_graspness(scores: np.ndarray) -> PointGraspness:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {scores.shape}")

    valid = scores >= 0
    valid_counts = valid.reshape(scores.shape[0], -1).sum(axis=1)
    valid_ratios = valid_counts.astype(np.float32) / float(np.prod(scores.shape[1:]))

    masked = np.where(valid, scores, np.inf)
    best_friction = masked.reshape(scores.shape[0], -1).min(axis=1)
    best_quality = np.where(np.isfinite(best_friction), 1.1 - best_friction, 0.0)
    best_quality = np.clip(best_quality, 0.0, 1.0)

    return PointGraspness(
        valid_counts=valid_counts,
        valid_ratios=valid_ratios,
        best_friction=best_friction,
        best_quality=best_quality,
    )


def generate_fibonacci_sphere_directions(count: int = 300) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"Direction count must be positive, got {count}")

    indices = np.arange(count, dtype=np.float32)
    golden_angle = np.float32(np.pi * (3.0 - np.sqrt(5.0)))
    z_values = 1.0 - (2.0 * (indices + 0.5) / float(count))
    radius = np.sqrt(np.maximum(0.0, 1.0 - z_values * z_values))
    theta = indices * golden_angle
    directions = np.column_stack(
        [np.cos(theta) * radius, np.sin(theta) * radius, z_values]
    ).astype(np.float32)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def select_best_graspness_point(scores: np.ndarray) -> int:
    graspness = compute_point_graspness(scores)
    return int(np.argmax(graspness.valid_ratios))


def compute_direction_quality(scores_for_point: np.ndarray) -> DirectionQuality:
    if scores_for_point.ndim != 3:
        raise ValueError(
            "Expected scores for one point with shape (V, A, D), "
            f"got {scores_for_point.shape}"
        )

    valid = scores_for_point >= 0
    valid_counts = valid.reshape(scores_for_point.shape[0], -1).sum(axis=1)
    masked = np.where(valid, scores_for_point, np.inf)
    best_friction = masked.reshape(scores_for_point.shape[0], -1).min(axis=1)
    best_quality = np.where(np.isfinite(best_friction), 1.1 - best_friction, 0.0)
    best_quality = np.clip(best_quality, 0.0, 1.0)
    return DirectionQuality(
        valid_counts=valid_counts,
        best_friction=best_friction,
        best_quality=best_quality,
    )


def select_best_direction(scores: np.ndarray, point_index: int) -> int:
    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {scores.shape}")
    if point_index < 0 or point_index >= scores.shape[0]:
        raise IndexError(
            f"point_index must be in [0, {scores.shape[0] - 1}], got {point_index}"
        )

    quality = compute_direction_quality(scores[point_index])
    return int(np.argmax(quality.best_quality))


def compute_grasp_config_quality(scores_for_direction: np.ndarray) -> np.ndarray:
    if scores_for_direction.ndim != 2:
        raise ValueError(
            "Expected scores for one direction with shape (A, D), "
            f"got {scores_for_direction.shape}"
        )

    quality = np.where(scores_for_direction >= 0, 1.1 - scores_for_direction, np.nan)
    quality = np.clip(quality, 0.0, 1.0)
    return quality.astype(np.float32)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    if radius == 0.0:
        radius = 0.01

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _set_equal_2d_axes(ax, x_values: np.ndarray, y_values: np.ndarray) -> None:
    x_min, x_max = float(x_values.min()), float(x_values.max())
    y_min, y_max = float(y_values.min()), float(y_values.max())
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    radius = max((x_max - x_min) / 2.0, (y_max - y_min) / 2.0, 0.01)
    ax.set_xlim(x_center - radius, x_center + radius)
    ax.set_ylim(y_center - radius, y_center + radius)
    ax.set_aspect("equal", adjustable="box")


def _draw_object_projection_background(
    ax,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    ax.scatter(
        x_values,
        y_values,
        c="#b0bec5",
        s=24.0,
        linewidths=0,
        alpha=0.22,
        zorder=1,
    )


def _save_rgb_figure(fig, output_path: Path) -> Path:
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(output_path) as image:
        image.convert("RGB").save(output_path)
    return output_path


def write_grasp_label_structure_table(
    data: GraspLabelData,
    output_path: str | Path,
    object_id: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        [
            "points",
            str(tuple(data.points.shape)),
            "物体坐标系中的表面采样点，每一行是 x/y/z。",
            "后续每个抓取候选都锚定在这些点上。",
        ],
        [
            "offsets",
            str(tuple(data.offsets.shape)),
            "每个点对应 300 个方向、12 个平面内角度、4 个深度配置。",
            "最后一维依次是 angle、depth、width。",
        ],
        [
            "scores",
            str(tuple(data.scores.shape)),
            "原始抓取质量标签，官方加载时作为摩擦系数使用。",
            "-1 表示无效；有效值越低越好。",
        ],
        [
            "collision",
            str(tuple(data.collision.shape)),
            "与 scores 同形状的布尔数组。",
            "物体级/场景级碰撞过滤会在最后单独解释。",
        ],
    ]

    fig, ax = plt.subplots(figsize=(16.2, 4.8))
    fig.patch.set_facecolor("#fbfbf7")
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["字段名", "形状", "含义", "后续用途"],
        cellLoc="left",
        colWidths=[0.12, 0.26, 0.32, 0.26],
        bbox=[0.02, 0.03, 0.96, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6d1c4")
        cell.set_linewidth(0.8)
        cell.PAD = 0.05
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(
                color="white", weight="bold", ha="left", fontproperties=CJK_FONT
            )
        elif row % 2 == 0:
            cell.set_facecolor("#f1efe7")
        else:
            cell.set_facecolor("#fffdfa")
        if col == 0 and row > 0:
            cell.set_text_props(
                weight="bold", color="#263238", fontproperties=CJK_FONT
            )
        elif row > 0:
            cell.set_text_props(fontproperties=CJK_FONT)

    ax.set_title(
        f"grasp_label/{object_id:03d}_labels.npz 结构总览",
        pad=10,
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    ax.text(
        0.02,
        0.87,
        "核心维度：采样点 N x 300 个抓取方向 x 12 个平面内角度 x 4 个深度配置。",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def write_grasp_label_points_preview(
    points: np.ndarray,
    output_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    object_model_points: np.ndarray | None = None,
    show_object_projection: bool = True,
    show_object_mesh: bool | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if show_object_mesh is not None:
        show_object_projection = show_object_mesh

    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    fig.patch.set_facecolor("#fbfbf7")
    background_points = object_model_points if object_model_points is not None else points
    views = [
        ("正视图: X-Z", 0, 2, "X", "Z"),
        ("侧视图: Y-Z", 1, 2, "Y", "Z"),
        ("俯视图: X-Y", 0, 1, "X", "Y"),
    ]
    for ax, (title, x_index, y_index, xlabel, ylabel) in zip(
        axes, views, strict=True
    ):
        x_values = points[:, x_index]
        y_values = points[:, y_index]
        bg_x_values = background_points[:, x_index]
        bg_y_values = background_points[:, y_index]
        if show_object_projection:
            _draw_object_projection_background(ax, bg_x_values, bg_y_values)
        ax.scatter(
            x_values,
            y_values,
            c="#26a69a",
            s=5.0,
            linewidths=0,
            alpha=0.9,
            zorder=2,
        )
        ax.set_title(title, fontproperties=CJK_FONT)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", color="#d8d2c5", alpha=0.75)
        _set_equal_2d_axes(
            ax,
            np.concatenate([x_values, bg_x_values]),
            np.concatenate([y_values, bg_y_values]),
        )

    fig.suptitle(
        f"grasp_label/{object_id:03d}: {object_name or 'object'} 表面采样点",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    plt.tight_layout()
    return _save_rgb_figure(fig, output_path)


def write_grasp_label_graspness_heatmap(
    points: np.ndarray,
    graspness_values: np.ndarray,
    output_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    object_model_points: np.ndarray | None = None,
    show_object_projection: bool = True,
    show_object_mesh: bool | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if show_object_mesh is not None:
        show_object_projection = show_object_mesh

    fig, axes = plt.subplots(ncols=3, figsize=(15.8, 5.2))
    fig.patch.set_facecolor("#fbfbf7")
    value_min = float(np.min(graspness_values))
    value_max = float(np.max(graspness_values))
    if value_max <= value_min:
        value_max = value_min + 0.001
    background_points = object_model_points if object_model_points is not None else points
    views = [
        ("正视图: X-Z", 0, 2, "X", "Z"),
        ("侧视图: Y-Z", 1, 2, "Y", "Z"),
        ("俯视图: X-Y", 0, 1, "X", "Y"),
    ]
    scatter = None
    for ax, (title, x_index, y_index, xlabel, ylabel) in zip(
        axes, views, strict=True
    ):
        x_values = points[:, x_index]
        y_values = points[:, y_index]
        bg_x_values = background_points[:, x_index]
        bg_y_values = background_points[:, y_index]
        if show_object_projection:
            _draw_object_projection_background(ax, bg_x_values, bg_y_values)
        scatter = ax.scatter(
            x_values,
            y_values,
            c=graspness_values,
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
            s=6.0,
            linewidths=0,
            zorder=2,
        )
        ax.set_title(title, fontproperties=CJK_FONT)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", color="#d8d2c5", alpha=0.75)
        _set_equal_2d_axes(
            ax,
            np.concatenate([x_values, bg_x_values]),
            np.concatenate([y_values, bg_y_values]),
        )

    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes, shrink=0.78, pad=0.02)
        cbar.set_label(
            "有效抓取候选比例 = valid / (300 x 12 x 4)", fontproperties=CJK_FONT
        )
    fig.suptitle(
        f"grasp_label/{object_id:03d}: {object_name or 'object'} 点级 graspness 热力图（有效候选比例）",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def write_grasp_label_300_directions(
    points: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    object_model_points: np.ndarray | None = None,
    point_index: int | None = None,
    direction_length: float | None = None,
    directions: np.ndarray | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {scores.shape}")
    if point_index is None:
        point_index = select_best_graspness_point(scores)
    if point_index < 0 or point_index >= points.shape[0]:
        raise IndexError(
            f"point_index must be in [0, {points.shape[0] - 1}], got {point_index}"
        )

    view_count = scores.shape[1]
    if directions is None:
        directions = generate_fibonacci_sphere_directions(view_count)
    if directions.shape != (view_count, 3):
        raise ValueError(
            f"Expected directions with shape ({view_count}, 3), got {directions.shape}"
        )

    background_points = object_model_points if object_model_points is not None else points
    anchor = points[point_index]
    quality = compute_direction_quality(scores[point_index])
    valid = quality.valid_counts > 0

    if direction_length is None:
        span = float(np.ptp(background_points, axis=0).max())
        direction_length = max(span * 0.18, 0.015)

    endpoints = anchor[None, :] + directions * float(direction_length)
    plot_points = np.vstack([background_points, anchor[None, :], endpoints])

    fig = plt.figure(figsize=(10.6, 9.0))
    fig.patch.set_facecolor("#fbfbf7")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        background_points[:, 0],
        background_points[:, 1],
        background_points[:, 2],
        c="#b0bec5",
        s=3.0,
        linewidths=0,
        alpha=0.14,
        zorder=1,
    )
    ax.scatter(
        [anchor[0]],
        [anchor[1]],
        [anchor[2]],
        c="#e53935",
        s=62.0,
        edgecolors="#3e2723",
        linewidths=0.8,
        zorder=4,
    )

    valid_quality = quality.best_quality[valid]
    color_min = float(valid_quality.min()) if valid_quality.size else 0.0
    color_max = float(valid_quality.max()) if valid_quality.size else 1.0
    if color_max <= color_min:
        color_max = color_min + 0.001
    norm = plt.Normalize(vmin=color_min, vmax=color_max)
    cmap = plt.get_cmap("viridis")

    for direction, direction_quality, is_valid in zip(
        directions, quality.best_quality, valid, strict=True
    ):
        color = cmap(norm(float(direction_quality))) if is_valid else "#c7c7c7"
        end = anchor + direction * float(direction_length)
        ax.plot(
            [anchor[0], end[0]],
            [anchor[1], end[1]],
            [anchor[2], end[2]],
            color=color,
            linewidth=0.9 if is_valid else 0.45,
            alpha=0.78 if is_valid else 0.22,
            zorder=3 if is_valid else 2,
        )

    if valid_quality.size:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.62, pad=0.02)
        cbar.set_label(
            "每方向最佳质量 = clip(1.1 - best friction, 0, 1)",
            fontproperties=CJK_FONT,
        )

    point_graspness = compute_point_graspness(scores).valid_ratios[point_index]
    ax.set_title(
        (
            f"grasp_label/{object_id:03d}: {object_name or 'object'} "
            f"point {point_index} 的 300 个方向\n"
            f"红点为采样点；颜色表示该方向下 12 x 4 配置的最佳质量；"
            f"点级有效比例 {point_graspness:.3f}"
        ),
        fontsize=13,
        weight="bold",
        fontproperties=CJK_FONT,
        pad=16,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=22, azim=-58)
    ax.grid(True, linestyle="--", color="#d8d2c5", alpha=0.45)
    _set_equal_axes(ax, plot_points)
    fig.text(
        0.08,
        0.05,
        "方向向量使用本仓库确定性 Fibonacci 球面采样生成，用于解释 scores 的 300 个方向维度。",
        fontsize=9.5,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def write_grasp_label_12x4_explanation(
    points: np.ndarray,
    offsets: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    point_index: int | None = None,
    direction_index: int | None = None,
    directions: np.ndarray | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if offsets.ndim != 5 or offsets.shape[-1] != 3:
        raise ValueError(
            f"Expected offsets with shape (N, V, A, D, 3), got {offsets.shape}"
        )
    if scores.ndim != 4:
        raise ValueError(f"Expected scores with shape (N, V, A, D), got {scores.shape}")
    if point_index is None:
        point_index = select_best_graspness_point(scores)
    if point_index < 0 or point_index >= points.shape[0]:
        raise IndexError(
            f"point_index must be in [0, {points.shape[0] - 1}], got {point_index}"
        )
    if direction_index is None:
        direction_index = select_best_direction(scores, point_index)
    if direction_index < 0 or direction_index >= scores.shape[1]:
        raise IndexError(
            f"direction_index must be in [0, {scores.shape[1] - 1}], got {direction_index}"
        )

    view_count = scores.shape[1]
    if directions is None:
        directions = generate_fibonacci_sphere_directions(view_count)
    if directions.shape != (view_count, 3):
        raise ValueError(
            f"Expected directions with shape ({view_count}, 3), got {directions.shape}"
        )

    direction = directions[direction_index]
    config_quality = compute_grasp_config_quality(scores[point_index, direction_index])
    valid = np.isfinite(config_quality)
    best_flat_index = int(np.nanargmax(config_quality)) if valid.any() else 0
    best_angle_index, best_depth_index = np.unravel_index(
        best_flat_index, config_quality.shape
    )
    angle_count, depth_count = config_quality.shape
    angle_offsets = offsets[point_index, direction_index, :, best_depth_index, 0]
    depth_offsets = offsets[point_index, direction_index, best_angle_index, :, 1]
    width_offsets = offsets[point_index, direction_index, best_angle_index, :, 2]

    fig = plt.figure(figsize=(18, 6.4))
    fig.patch.set_facecolor("#fbfbf7")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.45], wspace=0.32)

    angle_ax = fig.add_subplot(grid[0, 0])
    angles = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    angle_ax.add_patch(plt.Circle((0.0, 0.0), 1.0, fill=False, color="#78909c", lw=1.4))
    for index, angle in enumerate(angles):
        start = np.array([0.25 * np.cos(angle), 0.25 * np.sin(angle)])
        end = np.array([0.95 * np.cos(angle), 0.95 * np.sin(angle)])
        color = "#e53935" if index == best_angle_index else "#00897b"
        angle_ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2.0)
        angle_ax.text(
            1.18 * np.cos(angle),
            1.18 * np.sin(angle),
            str(index),
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
        )
    angle_ax.arrow(
        -0.16,
        0.0,
        0.32,
        0.0,
        width=0.015,
        color="#37474f",
        length_includes_head=True,
    )
    angle_ax.arrow(
        0.0,
        -0.16,
        0.0,
        0.32,
        width=0.015,
        color="#37474f",
        length_includes_head=True,
    )
    angle_ax.set_title(
        "12 个 angle: 绕抓取方向的平面内旋转",
        fontsize=12,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    angle_ax.text(
        -1.25,
        -1.45,
        f"红色为当前最佳 angle={best_angle_index}\n"
        f"offset angle 值: {float(angle_offsets[best_angle_index]):.4f}",
        fontsize=9.5,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    angle_ax.set_xlim(-1.42, 1.42)
    angle_ax.set_ylim(-1.58, 1.35)
    angle_ax.set_aspect("equal", adjustable="box")
    angle_ax.axis("off")

    depth_ax = fig.add_subplot(grid[0, 1])
    depth_positions = np.linspace(0.15, 0.95, depth_count)
    depth_values = depth_offsets.astype(np.float32)
    if np.ptp(depth_values) > 1e-6:
        normalized_depth = (depth_values - depth_values.min()) / np.ptp(depth_values)
        depth_positions = 0.15 + normalized_depth * 0.8
    depth_ax.arrow(
        0.0,
        0.0,
        1.08,
        0.0,
        width=0.018,
        color="#37474f",
        length_includes_head=True,
    )
    for index, x_pos in enumerate(depth_positions):
        color = "#e53935" if index == best_depth_index else "#1565c0"
        depth_ax.plot([x_pos, x_pos], [-0.24, 0.24], color=color, lw=2.4)
        depth_ax.scatter([x_pos], [0.0], c=color, s=48, zorder=3)
        depth_ax.text(
            x_pos,
            0.36,
            f"d{index}",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
            fontproperties=CJK_FONT,
        )
        depth_ax.text(
            x_pos,
            -0.42,
            f"{float(depth_offsets[index]):.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#455a64",
        )
    depth_ax.text(
        0.0,
        -0.68,
        "width 随 depth 取值: "
        + ", ".join(f"{float(value):.3f}" for value in width_offsets),
        fontsize=8.5,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    depth_ax.set_title(
        "4 个 depth: 沿 approach direction 的深度/接近距离",
        fontsize=12,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    depth_ax.set_xlim(-0.08, 1.18)
    depth_ax.set_ylim(-0.82, 0.58)
    depth_ax.axis("off")

    heatmap_ax = fig.add_subplot(grid[0, 2])
    masked_quality = np.ma.masked_invalid(config_quality.T)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d7d2c8")
    image = heatmap_ax.imshow(
        masked_quality,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        origin="lower",
        aspect="auto",
    )
    heatmap_ax.scatter(
        [best_angle_index],
        [best_depth_index],
        marker="s",
        s=220,
        facecolors="none",
        edgecolors="#e53935",
        linewidths=2.4,
    )
    heatmap_ax.set_xticks(np.arange(angle_count))
    heatmap_ax.set_yticks(np.arange(depth_count))
    heatmap_ax.set_xlabel("angle index")
    heatmap_ax.set_ylabel("depth index")
    heatmap_ax.set_title(
        "12 x 4 quality heatmap",
        fontsize=12,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    for angle_index in range(angle_count):
        for depth_index in range(depth_count):
            value = config_quality[angle_index, depth_index]
            label = "invalid" if not np.isfinite(value) else f"{float(value):.2f}"
            heatmap_ax.text(
                angle_index,
                depth_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color="#263238" if not np.isfinite(value) or value > 0.55 else "white",
            )
    cbar = fig.colorbar(image, ax=heatmap_ax, shrink=0.82, pad=0.02)
    cbar.set_label("quality = clip(1.1 - friction, 0, 1)", fontproperties=CJK_FONT)

    fig.suptitle(
        (
            f"grasp_label/{object_id:03d}: {object_name or 'object'} "
            f"point {point_index}, direction {direction_index} 的 12 x 4 配置\n"
            f"direction=({float(direction[0]):.3f}, {float(direction[1]):.3f}, {float(direction[2]):.3f})；"
            "灰色表示 score=-1 无效"
        ),
        fontsize=14,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    fig.text(
        0.05,
        0.035,
        "offsets 最后一维为 angle/depth/width；本图固定 point 与 direction，只展开 angle x depth 的离散选择。",
        fontsize=9.5,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def export_grasp_label_visualization(
    grasp_label_path: str | Path,
    output_structure_table_path: str | Path,
    output_points_preview_path: str | Path,
    output_graspness_heatmap_path: str | Path,
    output_300_directions_path: str | Path,
    output_12x4_explanation_path: str | Path,
    object_id: int,
    object_name: str | None = None,
    object_model_path: str | Path | None = None,
    point_index: int | None = None,
    direction_length: float | None = None,
    direction_index: int | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    data = load_grasp_label(grasp_label_path)
    graspness = compute_point_graspness(data.scores)
    object_model_points = (
        load_ply_vertices(object_model_path) if object_model_path is not None else None
    )

    structure = write_grasp_label_structure_table(
        data, output_structure_table_path, object_id=object_id
    )
    points = write_grasp_label_points_preview(
        data.points,
        output_points_preview_path,
        object_id=object_id,
        object_name=object_name,
        object_model_points=object_model_points,
    )
    heatmap = write_grasp_label_graspness_heatmap(
        data.points,
        graspness.valid_ratios,
        output_graspness_heatmap_path,
        object_id=object_id,
        object_name=object_name,
        object_model_points=object_model_points,
    )
    directions = write_grasp_label_300_directions(
        data.points,
        data.scores,
        output_300_directions_path,
        object_id=object_id,
        object_name=object_name,
        object_model_points=object_model_points,
        point_index=point_index,
        direction_length=direction_length,
    )
    configs = write_grasp_label_12x4_explanation(
        data.points,
        data.offsets,
        data.scores,
        output_12x4_explanation_path,
        object_id=object_id,
        object_name=object_name,
        point_index=point_index,
        direction_index=direction_index,
    )
    return structure, points, heatmap, directions, configs
