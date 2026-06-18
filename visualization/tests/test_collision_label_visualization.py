from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.annotations_visualization import AnnotationObject
from visualization.src.collision_label_visualization import (
    CollisionObjectStats,
    collision_free_valid_mask,
    load_scene_collision_labels,
    map_collision_labels_to_objects,
    summarize_collision_labels,
    write_collision_filtered_topk_overlay,
    write_collision_label_collision_rates,
    write_collision_label_structure_table,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def _objects() -> list[AnnotationObject]:
    return [
        AnnotationObject(5, "banana.ply", "banana.ply", (0, 0, 0), (0, 0, 0, 1)),
        AnnotationObject(14, "peach.ply", "peach.ply", (0, 0, 0), (0, 0, 0, 1)),
    ]


def test_build_sample_paths_includes_collision_label_paths():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.collision_label_path == Path(
        "/home/zky-miakho/datas/collision_label/scene_0000/collision_labels.npz"
    )
    assert paths.output_collision_label_structure_table_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/collision_label_structure_table.png"
    )
    assert paths.output_collision_label_collision_rates_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/collision_label_collision_rates.png"
    )
    assert paths.output_collision_filtered_topk_overlay_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/collision_filtered_topk_grasps_rgb_overlay.png"
    )


def test_load_scene_collision_labels_reads_npz_keys():
    labels = load_scene_collision_labels(
        "/home/zky-miakho/datas/collision_label/scene_0000/collision_labels.npz"
    )

    assert sorted(labels) == [f"arr_{index}" for index in range(9)]
    assert labels["arr_0"].shape == (487, 300, 12, 4)
    assert labels["arr_0"].dtype == bool


def test_map_collision_labels_to_objects_uses_annotation_order():
    labels = {"arr_0": np.zeros((1, 2, 3, 4), dtype=bool), "arr_1": np.ones((2, 2, 3, 4), dtype=bool)}

    mapped = map_collision_labels_to_objects(labels, _objects())

    assert list(mapped) == [5, 14]
    assert mapped[5].shape == (1, 2, 3, 4)
    assert mapped[14].all()


def test_collision_free_valid_mask_combines_score_and_collision():
    scores = np.array([[[[0.1, -1.0], [0.5, 0.9]]]], dtype=np.float32)
    collision = np.array([[[[False, False], [True, False]]]], dtype=bool)

    mask = collision_free_valid_mask(scores, collision)

    assert mask.tolist() == [[[[True, False], [False, True]]]]


def test_summarize_collision_labels_counts_collision_and_free_valid_candidates():
    scores_by_object = {
        5: np.array([[[[0.1, -1.0], [0.5, 0.9]]]], dtype=np.float32),
        14: np.array([[[[0.2, 0.3]]]], dtype=np.float32),
    }
    collision_by_object = {
        5: np.array([[[[False, False], [True, False]]]], dtype=bool),
        14: np.array([[[[True, True]]]], dtype=bool),
    }

    stats = summarize_collision_labels(_objects(), collision_by_object, scores_by_object)

    assert stats[0].obj_id == 5
    assert stats[0].shape == (1, 1, 2, 2)
    assert stats[0].valid_count == 3
    assert stats[0].collision_free_valid_count == 2
    assert stats[1].obj_id == 14
    assert stats[1].collision_free_valid_count == 0


def test_write_collision_label_summary_outputs_create_rgb_pngs(tmp_path):
    stats = [
        CollisionObjectStats("arr_0", 5, "banana.ply", (1, 2, 2, 2), 8, 4, 4, 3, 1),
        CollisionObjectStats("arr_1", 14, "peach.ply", (1, 2, 2, 2), 8, 6, 2, 2, 2),
    ]
    table_path = tmp_path / "structure.png"
    rates_path = tmp_path / "rates.png"

    write_collision_label_structure_table(stats, table_path)
    write_collision_label_collision_rates(stats, rates_path)

    for output in [table_path, rates_path]:
        with Image.open(output) as image:
            assert image.mode == "RGB"
            assert image.width > image.height


def test_write_collision_filtered_topk_overlay_creates_rgb_image(tmp_path):
    paths = build_sample_paths(DEFAULT_DATA_ROOT)
    output_path = tmp_path / "collision_filtered_overlay.png"

    result = write_collision_filtered_topk_overlay(paths, output_path, topk_per_object=1)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.size == (2560, 720)
