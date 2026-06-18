from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.grasp_label_visualization import (
    compute_direction_quality,
    compute_grasp_config_quality,
    GraspLabelData,
    compute_point_graspness,
    export_grasp_label_visualization,
    generate_fibonacci_sphere_directions,
    load_ply_vertices,
    load_grasp_label,
    select_best_direction,
    select_best_graspness_point,
    write_grasp_label_12x4_explanation,
    write_grasp_label_300_directions,
    write_grasp_label_graspness_heatmap,
    write_grasp_label_points_preview,
    write_grasp_label_structure_table,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def _small_grasp_label_data() -> GraspLabelData:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    offsets = np.zeros((3, 2, 2, 2, 3), dtype=np.float32)
    scores = np.full((3, 2, 2, 2), -1.0, dtype=np.float32)
    scores[0, 0, 0, 0] = 0.1
    scores[0, 0, 1, 0] = 0.4
    scores[1, 1, 0, 0] = 1.0
    collision = np.zeros((3, 2, 2, 2), dtype=bool)
    return GraspLabelData(points=points, offsets=offsets, scores=scores, collision=collision)


def test_build_sample_paths_includes_default_grasp_label_outputs():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.grasp_label_path == Path(
        "/home/zky-miakho/datas/grasp_label/014_labels.npz"
    )
    assert paths.grasp_object_model_path == Path(
        "/home/zky-miakho/datas/models-xt19/models/014/nontextured_simplified.ply"
    )
    assert paths.output_grasp_label_structure_table_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/grasp_label_structure_table.png"
    )
    assert paths.output_grasp_label_points_preview_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/grasp_label_points_preview.png"
    )
    assert paths.output_grasp_label_graspness_heatmap_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/grasp_label_graspness_heatmap.png"
    )
    assert paths.output_grasp_label_300_directions_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/grasp_label_300_directions.png"
    )
    assert paths.output_grasp_label_12x4_explanation_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/grasp_label_12x4_explanation.png"
    )


def test_load_grasp_label_reads_npz_arrays():
    data = load_grasp_label("/home/zky-miakho/datas/grasp_label/014_labels.npz")

    assert data.points.shape == (487, 3)
    assert data.offsets.shape == (487, 300, 12, 4, 3)
    assert data.scores.shape == (487, 300, 12, 4)
    assert data.collision.shape == (487, 300, 12, 4)


def test_load_ply_vertices_reads_object_model_points():
    points = load_ply_vertices(
        "/home/zky-miakho/datas/models-xt19/models/014/nontextured_simplified.ply",
        max_points=2048,
    )

    assert points.shape == (2048, 3)
    assert points.dtype == np.float32
    assert np.isfinite(points).all()


def test_compute_point_graspness_counts_valid_grasps_and_quality():
    data = _small_grasp_label_data()

    graspness = compute_point_graspness(data.scores)

    np.testing.assert_allclose(graspness.valid_counts, [2, 1, 0])
    np.testing.assert_allclose(graspness.best_friction, [0.1, 1.0, np.inf])
    np.testing.assert_allclose(graspness.best_quality, [1.0, 0.1, 0.0], atol=1e-6)
    np.testing.assert_allclose(graspness.valid_ratios, [0.25, 0.125, 0.0])


def test_generate_fibonacci_sphere_directions_returns_unit_vectors():
    directions = generate_fibonacci_sphere_directions(300)

    assert directions.shape == (300, 3)
    assert directions.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-6)


def test_select_best_graspness_point_returns_highest_valid_ratio_index():
    data = _small_grasp_label_data()

    assert select_best_graspness_point(data.scores) == 0


def test_compute_direction_quality_uses_best_valid_score_per_direction():
    scores_for_point = np.full((3, 2, 2), -1.0, dtype=np.float32)
    scores_for_point[0, 0, 0] = 0.8
    scores_for_point[0, 1, 0] = 0.2
    scores_for_point[2, 0, 1] = 1.0

    quality = compute_direction_quality(scores_for_point)

    np.testing.assert_allclose(quality.valid_counts, [2, 0, 1])
    np.testing.assert_allclose(quality.best_friction, [0.2, np.inf, 1.0])
    np.testing.assert_allclose(quality.best_quality, [0.9, 0.0, 0.1], atol=1e-6)


def test_select_best_direction_returns_highest_quality_direction_for_point():
    data = _small_grasp_label_data()

    assert select_best_direction(data.scores, point_index=0) == 0
    assert select_best_direction(data.scores, point_index=1) == 1


def test_compute_grasp_config_quality_converts_12x4_scores_to_quality_grid():
    scores_for_direction = np.array(
        [
            [0.1, -1.0],
            [0.8, 1.0],
        ],
        dtype=np.float32,
    )

    quality = compute_grasp_config_quality(scores_for_direction)

    np.testing.assert_allclose(quality, [[1.0, np.nan], [0.3, 0.1]], atol=1e-6)


def test_write_grasp_label_structure_table_writes_chinese_png(tmp_path):
    data = _small_grasp_label_data()
    output_path = tmp_path / "grasp_label_structure_table.png"

    result = write_grasp_label_structure_table(data, output_path, object_id=14)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height
        assert image.width > 1000


def test_write_grasp_label_points_preview_writes_rgb_png(tmp_path):
    data = _small_grasp_label_data()
    output_path = tmp_path / "grasp_label_points_preview.png"

    result = write_grasp_label_points_preview(data.points, output_path, object_id=14)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height


def test_write_grasp_label_points_preview_can_overlay_object_projection(tmp_path):
    data = _small_grasp_label_data()
    object_model_points = np.array(
        [
            [-0.05, -0.05, -0.04],
            [0.15, -0.05, -0.04],
            [-0.05, 0.15, -0.04],
            [0.15, 0.15, 0.04],
        ],
        dtype=np.float32,
    )
    output_path = tmp_path / "grasp_label_points_preview.png"

    result = write_grasp_label_points_preview(
        data.points,
        output_path,
        object_id=14,
        object_model_points=object_model_points,
        show_object_projection=True,
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        pixels = np.array(image)
        assert (pixels < 250).any()


def test_write_grasp_label_graspness_heatmap_writes_rgb_png(tmp_path):
    data = _small_grasp_label_data()
    graspness = compute_point_graspness(data.scores)
    output_path = tmp_path / "grasp_label_graspness_heatmap.png"

    result = write_grasp_label_graspness_heatmap(
        data.points, graspness.best_quality, output_path, object_id=14
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height


def test_write_grasp_label_300_directions_writes_rgb_png(tmp_path):
    data = _small_grasp_label_data()
    output_path = tmp_path / "grasp_label_300_directions.png"

    result = write_grasp_label_300_directions(
        data.points,
        data.scores,
        output_path,
        object_id=14,
        point_index=0,
        directions=generate_fibonacci_sphere_directions(2),
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height
        pixels = np.array(image)
        assert (pixels < 250).any()


def test_write_grasp_label_12x4_explanation_writes_rgb_png(tmp_path):
    data = _small_grasp_label_data()
    output_path = tmp_path / "grasp_label_12x4_explanation.png"

    result = write_grasp_label_12x4_explanation(
        data.points,
        data.offsets,
        data.scores,
        output_path,
        object_id=14,
        point_index=0,
        direction_index=0,
        directions=generate_fibonacci_sphere_directions(2),
    )

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height
        pixels = np.array(image)
        assert (pixels < 250).any()


def test_export_grasp_label_visualization_writes_all_outputs(tmp_path):
    output_dir = tmp_path / "outputs"

    outputs = export_grasp_label_visualization(
        "/home/zky-miakho/datas/grasp_label/014_labels.npz",
        output_dir / "structure.png",
        output_dir / "points.png",
        output_dir / "graspness.png",
        output_dir / "directions.png",
        output_dir / "configs.png",
        object_id=14,
    )

    assert outputs == (
        output_dir / "structure.png",
        output_dir / "points.png",
        output_dir / "graspness.png",
        output_dir / "directions.png",
        output_dir / "configs.png",
    )
    for output in outputs:
        with Image.open(output) as image:
            assert image.mode == "RGB"
