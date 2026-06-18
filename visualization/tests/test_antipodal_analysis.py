from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.antipodal_analysis import (
    AntipodalContactEstimate,
    compute_antipodal_score,
    estimate_contact_points,
    estimate_surface_normal,
    select_representative_antipodal_candidate,
    write_antipodal_visualizations,
)
from visualization.src.grasp_label_visualization import GraspLabelData
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths
from visualization.src.topk_grasp_overlay import GraspCandidate


def _candidate() -> GraspCandidate:
    return GraspCandidate(
        object_id=14,
        point_index=0,
        direction_index=0,
        angle_index=0,
        depth_index=0,
        friction=0.1,
        quality=1.0,
    )


def test_build_sample_paths_includes_antipodal_outputs():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.output_antipodal_concept_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/antipodal_on_real_object_concept.png"
    )
    assert paths.output_antipodal_positive_negative_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/antipodal_positive_negative_on_object.png"
    )
    assert paths.output_antipodal_top_grasp_analysis_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/antipodal_top_grasp_analysis.png"
    )


def test_estimate_contact_points_uses_width():
    points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    offsets = np.zeros((1, 1, 1, 1, 3), dtype=np.float32)
    offsets[0, 0, 0, 0] = [0.0, 0.04, 0.08]
    direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    estimate = estimate_contact_points(points, offsets, _candidate(), direction)

    assert isinstance(estimate, AntipodalContactEstimate)
    np.testing.assert_allclose(estimate.center, [0.0, 0.0, 0.04], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(estimate.left - estimate.right), 0.08, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(estimate.opening_axis), 1.0, atol=1e-6)


def test_estimate_surface_normal_returns_unit_vector():
    xx, yy = np.meshgrid(np.linspace(-0.03, 0.03, 5), np.linspace(-0.03, 0.03, 5))
    plane = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    normal = estimate_surface_normal(
        plane.astype(np.float32),
        query_point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        orient_toward=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        neighbor_count=12,
    )

    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-6)
    assert normal[2] > 0.99


def test_antipodal_score_higher_for_opposing_normals():
    left = np.array([-0.04, 0.0, 0.0], dtype=np.float32)
    right = np.array([0.04, 0.0, 0.0], dtype=np.float32)
    good = compute_antipodal_score(
        left,
        right,
        left_normal=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        right_normal=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
        friction_coefficient=0.5,
    )
    bad = compute_antipodal_score(
        left,
        right,
        left_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        right_normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        friction_coefficient=0.5,
    )

    assert good > 0.9
    assert bad < 0.2
    assert good > bad


def test_select_representative_candidate_prefers_clear_contact_separation():
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    offsets = np.zeros((2, 1, 1, 1, 3), dtype=np.float32)
    offsets[0, 0, 0, 0] = [0.0, 0.0, 0.01]
    offsets[1, 0, 0, 0] = [0.0, 0.0, 0.08]
    scores = np.full((2, 1, 1, 1), 0.1, dtype=np.float32)
    collision = np.zeros_like(scores, dtype=bool)
    data = GraspLabelData(
        points=points,
        offsets=offsets,
        scores=scores,
        collision=collision,
    )
    object_model_points = np.array(
        [
            [0.0, -0.04, -0.01],
            [0.0, -0.04, 0.01],
            [0.0, -0.005, 0.0],
            [0.0, 0.005, 0.0],
            [0.0, 0.04, -0.01],
            [0.0, 0.04, 0.01],
        ],
        dtype=np.float32,
    )

    candidate = select_representative_antipodal_candidate(
        data,
        object_model_points,
        object_id=14,
        directions=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        top_pool_size=2,
    )

    assert candidate.point_index == 1


def test_write_antipodal_visualizations_writes_rgb_pngs(tmp_path):
    paths = build_sample_paths(DEFAULT_DATA_ROOT, output_root=tmp_path)

    outputs = write_antipodal_visualizations(
        grasp_label_path=paths.grasp_label_path,
        object_model_path=paths.grasp_object_model_path,
        output_concept_path=paths.output_antipodal_concept_path,
        output_positive_negative_path=paths.output_antipodal_positive_negative_path,
        output_top_grasp_analysis_path=paths.output_antipodal_top_grasp_analysis_path,
        object_id=14,
        object_name="peach",
        max_model_points=2500,
    )

    assert outputs == (
        paths.output_antipodal_concept_path,
        paths.output_antipodal_positive_negative_path,
        paths.output_antipodal_top_grasp_analysis_path,
    )
    for output_path in outputs:
        with Image.open(output_path) as image:
            assert image.mode == "RGB"
            assert image.width > 600
            assert image.height > 450
            assert (np.array(image) < 250).any()
