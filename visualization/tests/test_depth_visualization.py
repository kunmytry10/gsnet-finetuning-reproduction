from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.depth_visualization import (
    colorize_depth,
    colorize_depth_hamilton,
    decode_hamilton_depth,
    encode_hamilton_depth,
    export_depth_colormap,
    export_depth_hamilton_colormap,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_depth_input_and_output_paths():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.depth_path == Path(
        "/home/zky-miakho/datas/train_1/scene_0000/kinect/depth/0000.png"
    )
    assert paths.output_depth_colormap_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/depth_colormap.png"
    )
    assert paths.output_depth_hamilton_colormap_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/depth_hamilton_colormap.png"
    )


def test_colorize_depth_uses_viridis_with_black_invalid_pixels():
    depth = np.array(
        [
            [0, 100],
            [200, 300],
        ],
        dtype=np.uint16,
    )

    image = colorize_depth(depth)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 1)) == (253, 231, 36)


def test_encode_hamilton_depth_walks_rgb_cube_edges():
    depth = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    encoded = encode_hamilton_depth(depth, dmin=10.0, dmax=30.0)

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [[0, 0, 0], [0, 255, 128], [255, 255, 255]]


def test_decode_hamilton_depth_approximately_recovers_depth():
    depth = np.array([[10.0, 15.0], [20.0, 30.0]], dtype=np.float32)
    encoded = encode_hamilton_depth(depth, dmin=10.0, dmax=30.0)

    decoded = decode_hamilton_depth(encoded, dmin=10.0, dmax=30.0)

    np.testing.assert_allclose(decoded, depth, atol=0.08)


def test_colorize_depth_hamilton_keeps_invalid_pixels_black():
    depth = np.array(
        [
            [0, 100],
            [200, 300],
        ],
        dtype=np.uint16,
    )

    image = colorize_depth_hamilton(depth)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)
    assert image.getpixel((1, 1)) == (255, 255, 255)


def test_export_depth_colormap_writes_rgb_png_with_same_dimensions(tmp_path):
    source = tmp_path / "depth.png"
    output = tmp_path / "depth_colormap.png"
    depth = np.array(
        [
            [0, 100, 200],
            [300, 400, 500],
        ],
        dtype=np.uint16,
    )
    Image.fromarray(depth).save(source)

    result = export_depth_colormap(source, output)

    assert result == output
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (3, 2)


def test_export_depth_hamilton_colormap_writes_rgb_png_with_same_dimensions(tmp_path):
    source = tmp_path / "depth.png"
    output = tmp_path / "depth_hamilton_colormap.png"
    depth = np.array(
        [
            [0, 100, 200],
            [300, 400, 500],
        ],
        dtype=np.uint16,
    )
    Image.fromarray(depth).save(source)

    result = export_depth_hamilton_colormap(source, output)

    assert result == output
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (3, 2)
