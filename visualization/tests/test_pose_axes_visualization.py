from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.pose_axes_visualization import (
    project_points,
    quaternion_to_rotation_matrix,
    transform_points,
    write_pose_axes_overlay,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_pose_axes_overlay_output():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.output_pose_axes_overlay_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/pose_axes_overlay.png"
    )


def test_quaternion_to_rotation_matrix_identity():
    rotation = quaternion_to_rotation_matrix((0.0, 0.0, 0.0, 1.0))

    np.testing.assert_allclose(rotation, np.eye(3), atol=1e-7)


def test_transform_points_applies_inverse_camera_pose():
    points = np.array([[1.0, 2.0, 4.0]])
    camera_pose = np.array(
        [
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 0.0, 0.50],
            [0.0, 0.0, 1.0, 1.00],
            [0.0, 0.0, 0.0, 1.00],
        ]
    )

    transformed = transform_points(points, camera_pose)

    np.testing.assert_allclose(transformed, np.array([[0.75, 1.5, 3.0]]))


def test_project_points_uses_camera_intrinsics():
    points = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 2.0]])
    intrinsic = np.array(
        [
            [100.0, 0.0, 320.0],
            [0.0, 100.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )

    pixels = project_points(points, intrinsic)

    np.testing.assert_allclose(pixels, np.array([[320.0, 240.0], [325.0, 230.0]]))


def test_write_pose_axes_overlay_creates_rgb_image(tmp_path):
    paths = build_sample_paths(DEFAULT_DATA_ROOT)
    output_path = tmp_path / "pose_axes_overlay.png"

    write_pose_axes_overlay(
        rgb_path=paths.rgb_path,
        annotations_path=paths.annotations_path,
        intrinsic_path=paths.camK_path,
        camera_poses_path=paths.camera_poses_path,
        output_path=output_path,
        frame_index=0,
    )

    with Image.open(paths.rgb_path) as rgb_image, Image.open(output_path) as output:
        assert output.size == rgb_image.size
        assert output.mode == "RGB"
