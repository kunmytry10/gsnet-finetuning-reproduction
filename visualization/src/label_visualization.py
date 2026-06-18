from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _label_color(label_id: int) -> tuple[int, int, int]:
    if label_id == 0:
        return (0, 0, 0)

    return (
        (37 * label_id + 53) % 256,
        (91 * label_id + 29) % 256,
        (149 * label_id + 101) % 256,
    )


def colorize_label(label: np.ndarray) -> Image.Image:
    if label.ndim != 2:
        raise ValueError(f"Expected a 2D label image, got shape {label.shape}")

    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for label_id in np.unique(label):
        rgb[label == label_id] = _label_color(int(label_id))

    return Image.fromarray(rgb, mode="RGB")


def export_label_overlay(
    rgb_path: str | Path,
    label_path: str | Path,
    output_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    rgb_path = Path(rgb_path)
    label_path = Path(label_path)
    output_path = Path(output_path)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label image not found: {label_path}")

    with Image.open(rgb_path) as rgb_image:
        rgb = np.array(rgb_image.convert("RGB"), dtype=np.float32)
    with Image.open(label_path) as label_image:
        label = np.array(label_image)

    color_label = np.array(colorize_label(label), dtype=np.float32)
    object_mask = label > 0

    overlay = rgb.copy()
    overlay[object_mask] = (
        (1.0 - alpha) * rgb[object_mask] + alpha * color_label[object_mask]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(
        output_path
    )
    return output_path
