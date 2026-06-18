from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.annotations_visualization import AnnotationObject
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths
from visualization.src.topk_grasp_overlay import (
    GraspCandidate,
    build_gripper_control_points,
    object_points_to_world,
    select_top_grasp_candidates,
    write_topk_grasp_overlay,
)


def _small_scores() -> np.ndarray:
    scores = np.full((2, 2, 2, 2), -1.0, dtype=np.float32)
    scores[0, 0, 0, 0] = 0.8
    scores[0, 1, 1, 0] = 0.1
    scores[1, 0, 0, 1] = 0.4
    return scores


def test_build_sample_paths_includes_topk_grasp_overlay_output():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.output_topk_grasp_overlay_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/topk_3d_grasps_rgb_overlay.png"
    )


def test_select_top_grasp_candidates_returns_valid_scores_by_quality():
    candidates = select_top_grasp_candidates(_small_scores(), topk=2)

    assert [(c.point_index, c.direction_index, c.angle_index, c.depth_index) for c in candidates] == [
        (0, 1, 1, 0),
        (1, 0, 0, 1),
    ]
    np.testing.assert_allclose([c.quality for c in candidates], [1.0, 0.7])


def test_select_top_grasp_candidates_prefers_visible_points_on_quality_ties():
    scores = np.full((2, 1, 1, 1), -1.0, dtype=np.float32)
    scores[0, 0, 0, 0] = 0.1
    scores[1, 0, 0, 0] = 0.1
    point_priorities = np.array([0.0, 1.0], dtype=np.float32)

    candidates = select_top_grasp_candidates(
        scores,
        topk=1,
        point_priorities=point_priorities,
    )

    assert candidates[0].point_index == 1


def test_build_gripper_control_points_responds_to_width_and_depth():
    candidate = GraspCandidate(
        object_id=5,
        point_index=0,
        direction_index=0,
        angle_index=0,
        depth_index=0,
        friction=0.1,
        quality=1.0,
    )
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    offsets = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
    offsets[0, 0, 0, 0] = [0.0, 0.04, 0.08]

    control_points = build_gripper_control_points(
        points,
        offsets,
        candidate,
        direction,
        finger_length=0.03,
    )

    assert control_points.shape == (6, 3)
    np.testing.assert_allclose(control_points[0], [0.0, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(control_points[1] - control_points[2]), 0.08, atol=1e-6)
    np.testing.assert_allclose(control_points[3], [0.0, 0.0, 0.01], atol=1e-6)


def test_object_points_to_world_applies_annotation_pose():
    obj = AnnotationObject(
        obj_id=1,
        obj_name="object.ply",
        obj_path="object.ply",
        position=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    world = object_points_to_world(np.array([[0.1, 0.2, 0.3]], dtype=np.float32), obj)

    np.testing.assert_allclose(world, [[1.1, 2.2, 3.3]], atol=1e-6)


def test_write_topk_grasp_overlay_creates_rgb_image(tmp_path):
    paths = build_sample_paths(DEFAULT_DATA_ROOT)
    output_path = tmp_path / "topk_3d_grasps_rgb_overlay.png"

    result = write_topk_grasp_overlay(
        rgb_path=paths.rgb_path,
        annotations_path=paths.annotations_path,
        intrinsic_path=paths.camK_path,
        camera_poses_path=paths.camera_poses_path,
        grasp_label_root=paths.data_root / "grasp_label",
        output_path=output_path,
        topk_per_object=1,
    )

    assert result == output_path
    with Image.open(paths.rgb_path) as rgb_image, Image.open(output_path) as output:
        assert output.size == rgb_image.size
        assert output.mode == "RGB"
        assert (np.array(output) != np.array(rgb_image.convert("RGB"))).any()
