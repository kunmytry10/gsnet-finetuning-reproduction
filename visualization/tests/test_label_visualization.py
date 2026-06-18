from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.label_visualization import (
    colorize_label,
    export_label_overlay,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_label_input_and_overlay_output_paths():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.label_path == Path(
        "/home/zky-miakho/datas/train_1/scene_0000/kinect/label/0000.png"
    )
    assert paths.output_label_overlay_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/label_overlay.png"
    )


def test_colorize_label_keeps_background_black_and_colors_objects():
    label = np.array(
        [
            [0, 1],
            [6, 15],
        ],
        dtype=np.uint8,
    )

    image = colorize_label(label)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 0)) != (0, 0, 0)
    assert image.getpixel((0, 1)) != image.getpixel((1, 1))


def test_export_label_overlay_writes_rgb_png_with_same_dimensions(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    label_path = tmp_path / "label.png"
    output_path = tmp_path / "label_overlay.png"

    Image.new("RGB", (3, 2), color=(100, 100, 100)).save(rgb_path)
    label = np.array(
        [
            [0, 1, 1],
            [0, 2, 2],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(label).save(label_path)

    result = export_label_overlay(rgb_path, label_path, output_path)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.size == (3, 2)
        assert image.getpixel((0, 0)) == (100, 100, 100)
        assert image.getpixel((1, 0)) != (100, 100, 100)
