from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.pointcloud_visualization import (
    depth_to_points,
    export_pointcloud,
    filter_projected_points,
    write_ascii_ply,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_camera_intrinsics_and_pointcloud_outputs():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.camK_path == Path(
        "/home/zky-miakho/datas/train_1/scene_0000/kinect/camK.npy"
    )
    assert paths.output_pointcloud_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/pointcloud_rgb.ply"
    )
    assert paths.output_pointcloud_preview_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/pointcloud_preview.png"
    )


def test_depth_to_points_projects_valid_depth_pixels_with_intrinsics():
    depth = np.array(
        [
            [1000, 0],
            [2000, 3000],
        ],
        dtype=np.uint16,
    )
    intrinsics = np.array(
        [
            [1000.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    points, pixel_rows, pixel_cols = depth_to_points(depth, intrinsics, depth_scale=1000.0)

    assert points.shape == (3, 3)
    assert pixel_rows.tolist() == [0, 1, 1]
    assert pixel_cols.tolist() == [0, 0, 1]
    np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(points[1], [0.0, 0.002, 2.0])
    np.testing.assert_allclose(points[2], [0.003, 0.003, 3.0])


def test_write_ascii_ply_writes_xyz_rgb_vertices(tmp_path):
    output_path = tmp_path / "cloud.ply"
    points = np.array([[0.0, 0.0, 1.0], [0.1, 0.2, 1.5]], dtype=np.float32)
    colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    result = write_ascii_ply(output_path, points, colors)

    assert result == output_path
    text = output_path.read_text()
    assert "element vertex 2" in text
    assert "property uchar red" in text
    assert "0.000000 0.000000 1.000000 255 0 0" in text


def test_filter_projected_points_removes_label_edges_and_depth_outliers():
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [2.0, 0.0, 3.0],
            [3.0, 0.0, 4.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    colors = np.zeros((5, 3), dtype=np.uint8)
    rows = np.array([0, 1, 2, 3, 4])
    cols = np.array([0, 1, 2, 3, 4])
    label = np.zeros((5, 5), dtype=np.uint8)
    label[1:4, 1:4] = 1

    filtered_points, filtered_colors = filter_projected_points(
        points,
        colors,
        rows,
        cols,
        label=label,
        erode_iterations=1,
        depth_percentiles=(5.0, 95.0),
    )

    assert filtered_points.shape == (1, 3)
    assert filtered_colors.shape == (1, 3)
    np.testing.assert_allclose(filtered_points[0], [2.0, 0.0, 3.0])


def test_export_pointcloud_filters_background_points_using_label_map(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.png"
    label_path = tmp_path / "label.png"
    camk_path = tmp_path / "camK.npy"
    output_ply = tmp_path / "cloud.ply"
    output_preview = tmp_path / "cloud.png"

    Image.new("RGB", (2, 2), color=(200, 100, 50)).save(rgb_path)
    Image.fromarray(
        np.array(
            [
                [1000, 2000],
                [3000, 4000],
            ],
            dtype=np.uint16,
        )
    ).save(depth_path)
    Image.fromarray(
        np.array(
            [
                [0, 1],
                [0, 2],
            ],
            dtype=np.uint8,
        )
    ).save(label_path)
    np.save(
        camk_path,
        np.array(
            [
                [1000.0, 0.0, 0.0],
                [0.0, 1000.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )

    export_pointcloud(
        rgb_path,
        depth_path,
        camk_path,
        output_ply,
        output_preview,
        label_path=label_path,
        depth_scale=1000.0,
    )

    text = output_ply.read_text()
    assert "element vertex 2" in text
    assert "0.002000 0.000000 2.000000" in text
    assert "0.004000 0.004000 4.000000" in text
    with Image.open(output_preview) as image:
        assert image.width > image.height
