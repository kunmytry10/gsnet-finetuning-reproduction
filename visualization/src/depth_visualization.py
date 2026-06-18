from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import numpy as np
from matplotlib import colormaps
from PIL import Image


HAMILTON_CORNERS = np.array(
    [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
    ],
    dtype=np.float64,
)
HAMILTON_SEGMENTS = len(HAMILTON_CORNERS) - 1


def colorize_depth(depth: np.ndarray) -> Image.Image:
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape {depth.shape}")

    depth = depth.astype(np.float32)
    valid_mask = depth > 0
    normalized = np.zeros_like(depth, dtype=np.float32)

    if valid_mask.any():
        valid_depth = depth[valid_mask]
        min_depth = float(valid_depth.min())
        max_depth = float(valid_depth.max())
        if max_depth > min_depth:
            normalized[valid_mask] = (depth[valid_mask] - min_depth) / (
                max_depth - min_depth
            )
        else:
            normalized[valid_mask] = 1.0

    rgb = (colormaps["viridis"](normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~valid_mask] = 0

    return Image.fromarray(rgb, mode="RGB")


def encode_hamilton_depth(depth: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    if dmax <= dmin:
        raise ValueError(f"dmax must be greater than dmin, got {dmin} and {dmax}")

    depth = depth.astype(np.float64)
    t = np.clip((depth - dmin) / (dmax - dmin), 0.0, 1.0)
    s = t * HAMILTON_SEGMENTS
    segment = np.clip(np.floor(s).astype(int), 0, HAMILTON_SEGMENTS - 1)
    fraction = (s - segment)[..., None]
    rgb = HAMILTON_CORNERS[segment] + fraction * (
        HAMILTON_CORNERS[segment + 1] - HAMILTON_CORNERS[segment]
    )
    return np.round(rgb * 255).astype(np.uint8)


def decode_hamilton_depth(rgb8: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    if rgb8.ndim < 1 or rgb8.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with last dimension 3, got {rgb8.shape}")
    if dmax <= dmin:
        raise ValueError(f"dmax must be greater than dmin, got {dmin} and {dmax}")

    pixels = rgb8.astype(np.float64) / 255.0
    flat = pixels.reshape(-1, 3)
    best_t = np.zeros(len(flat), dtype=np.float64)
    best_distance = np.full(len(flat), np.inf, dtype=np.float64)
    for index in range(HAMILTON_SEGMENTS):
        start = HAMILTON_CORNERS[index]
        end = HAMILTON_CORNERS[index + 1]
        edge = end - start
        u = np.clip(((flat - start) @ edge) / (edge @ edge), 0.0, 1.0)
        projected = start + u[:, None] * edge
        distance = np.linalg.norm(flat - projected, axis=1)
        mask = distance < best_distance
        best_distance[mask] = distance[mask]
        best_t[mask] = (index + u[mask]) / HAMILTON_SEGMENTS
    depth = best_t.reshape(rgb8.shape[:-1]) * (dmax - dmin) + dmin
    return depth.astype(np.float32)


def colorize_depth_hamilton(depth: np.ndarray) -> Image.Image:
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape {depth.shape}")

    depth = depth.astype(np.float32)
    valid_mask = depth > 0
    rgb = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid_mask.any():
        valid_depth = depth[valid_mask]
        min_depth = float(valid_depth.min())
        max_depth = float(valid_depth.max())
        if max_depth > min_depth:
            rgb[valid_mask] = encode_hamilton_depth(
                depth[valid_mask],
                dmin=min_depth,
                dmax=max_depth,
            )
        else:
            rgb[valid_mask] = np.array([255, 255, 255], dtype=np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def export_depth_colormap(source_path: str | Path, output_path: str | Path) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Depth image not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        depth = np.array(image)

    colorize_depth(depth).save(output_path)
    return output_path


def export_depth_hamilton_colormap(
    source_path: str | Path, output_path: str | Path
) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Depth image not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        depth = np.array(image)

    colorize_depth_hamilton(depth).save(output_path)
    return output_path
