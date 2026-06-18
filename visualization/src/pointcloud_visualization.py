from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

from visualization.src.sample_paths import SamplePaths


def depth_to_points(
    depth: np.ndarray, intrinsic: np.ndarray, depth_scale: float = 1000.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape {depth.shape}")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 intrinsic matrix, got shape {intrinsic.shape}")

    rows, cols = np.indices(depth.shape)
    valid = depth > 0

    z = depth[valid].astype(np.float32) / depth_scale
    x = (cols[valid].astype(np.float32) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (rows[valid].astype(np.float32) - intrinsic[1, 2]) * z / intrinsic[1, 1]

    points = np.stack([x, y, z], axis=1)
    return points, rows[valid], cols[valid]


def write_ascii_ply(
    output_path: str | Path, points: np.ndarray, colors: np.ndarray
) -> Path:
    output_path = Path(output_path)
    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same number of rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors, strict=True):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    return output_path


def _sample_points(
    points: np.ndarray, colors: np.ndarray, max_points: int = 120000
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors

    indices = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[indices], colors[indices]


def filter_projected_points(
    points: np.ndarray,
    colors: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    label: np.ndarray | None = None,
    erode_iterations: int = 1,
    depth_percentiles: tuple[float, float] = (1.0, 99.0),
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(points.shape[0], dtype=bool)

    if label is not None:
        foreground = label > 0
        if erode_iterations > 0:
            kernel = np.ones((3, 3), dtype=np.uint8)
            eroded = cv2.erode(
                foreground.astype(np.uint8), kernel, iterations=erode_iterations
            ).astype(bool)
            if eroded.any():
                foreground = eroded
        keep &= foreground[rows, cols]

    filtered_points = points[keep]
    filtered_colors = colors[keep]

    if filtered_points.shape[0] >= 20 and depth_percentiles is not None:
        low, high = np.percentile(filtered_points[:, 2], depth_percentiles)
        depth_keep = (filtered_points[:, 2] >= low) & (filtered_points[:, 2] <= high)
        filtered_points = filtered_points[depth_keep]
        filtered_colors = filtered_colors[depth_keep]

    return filtered_points, filtered_colors


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0)
    if radius == 0:
        radius = 0.1

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def write_three_view_preview(
    output_path: str | Path, points: np.ndarray, colors: np.ndarray
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 5))
    views = [
        ("Front", 15, -90),
        ("Side", 15, 0),
        ("Top", 90, -90),
    ]

    for index, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors.astype(np.float32) / 255.0,
            s=0.35,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        _set_equal_axes(ax, points)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def export_pointcloud(
    rgb_path: str | Path,
    depth_path: str | Path,
    intrinsic_path: str | Path,
    output_ply_path: str | Path,
    output_preview_path: str | Path,
    label_path: str | Path | None = None,
    depth_scale: float = 1000.0,
) -> tuple[Path, Path]:
    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)
    intrinsic_path = Path(intrinsic_path)
    output_ply_path = Path(output_ply_path)
    output_preview_path = Path(output_preview_path)

    with Image.open(rgb_path) as rgb_image:
        rgb = np.array(rgb_image.convert("RGB"))
    with Image.open(depth_path) as depth_image:
        depth = np.array(depth_image)

    intrinsic = np.load(intrinsic_path)
    points, rows, cols = depth_to_points(depth, intrinsic, depth_scale=depth_scale)
    colors = rgb[rows, cols]

    if label_path is not None:
        with Image.open(label_path) as label_image:
            label = np.array(label_image)
        if label.shape != depth.shape:
            raise ValueError(
                f"Label shape {label.shape} does not match depth shape {depth.shape}"
            )
        points, colors = filter_projected_points(
            points,
            colors,
            rows,
            cols,
            label=label,
            erode_iterations=1,
            depth_percentiles=(1.0, 99.0),
        )

    points, colors = _sample_points(points, colors)

    write_ascii_ply(output_ply_path, points, colors)
    write_three_view_preview(output_preview_path, points, colors)

    return output_ply_path, output_preview_path


def export_sample_pointcloud(paths: SamplePaths) -> tuple[Path, Path]:
    return export_pointcloud(
        paths.rgb_path,
        paths.depth_path,
        paths.camK_path,
        paths.output_pointcloud_path,
        paths.output_pointcloud_preview_path,
        label_path=paths.label_path,
    )
