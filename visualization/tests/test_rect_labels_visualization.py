from pathlib import Path

import numpy as np
from PIL import Image

from visualization.src.rect_labels_visualization import (
    export_rect_labels_visualization,
    parse_rect_labels_array,
    select_balanced_rect_grasps,
    select_top_rect_grasps,
    write_rect_labels_fields_table,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_rect_labels_input_and_outputs():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.rect_labels_path == Path(
        "/home/zky-miakho/datas/rect_labels/scene_0000/kinect/0000.npy"
    )
    assert paths.output_rect_labels_overlay_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/rect_labels_overlay.png"
    )
    assert paths.output_rect_labels_summary_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/rect_labels_summary.md"
    )
    assert paths.output_rect_labels_fields_table_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/rect_labels_fields_table.png"
    )


def test_parse_rect_labels_array_converts_rows_to_grasps():
    rows = np.array(
        [
            [10.0, 20.0, 30.0, 20.0, 8.0, 0.9, 5.0],
            [40.0, 50.0, 40.0, 70.0, 12.0, 0.3, 14.0],
        ],
        dtype=np.float32,
    )

    grasps = parse_rect_labels_array(rows)

    assert len(grasps) == 2
    assert grasps[0].center == (10.0, 20.0)
    assert grasps[0].open_point == (30.0, 20.0)
    assert grasps[0].width == 40.0
    assert grasps[0].height == 8.0
    assert grasps[0].score == 0.9
    assert grasps[0].object_id == 5
    assert round(grasps[1].angle_degrees, 1) == 90.0


def test_rect_grasp_uses_official_center_and_open_point_format():
    rows = np.array(
        [
            [100.0, 120.0, 130.0, 120.0, 20.0, 1.0, 66.0],
        ],
        dtype=np.float32,
    )

    grasp = parse_rect_labels_array(rows)[0]

    assert grasp.center == (100.0, 120.0)
    assert grasp.open_point == (130.0, 120.0)
    assert grasp.width == 60.0
    assert grasp.corners == [
        (130.0, 130.0),
        (70.0, 130.0),
        (70.0, 110.0),
        (130.0, 110.0),
    ]


def test_select_top_rect_grasps_sorts_by_score_descending():
    rows = np.array(
        [
            [10.0, 20.0, 30.0, 20.0, 8.0, 0.1, 5.0],
            [40.0, 50.0, 60.0, 50.0, 8.0, 0.9, 14.0],
            [70.0, 80.0, 90.0, 80.0, 8.0, 0.6, 15.0],
        ],
        dtype=np.float32,
    )
    grasps = parse_rect_labels_array(rows)

    selected = select_top_rect_grasps(grasps, max_grasps=2)

    assert [grasp.score for grasp in selected] == [0.9, 0.6]
    assert [grasp.object_id for grasp in selected] == [14, 15]


def test_select_balanced_rect_grasps_keeps_each_object_represented():
    rows = np.array(
        [
            [10.0, 20.0, 30.0, 20.0, 8.0, 1.0, 5.0],
            [12.0, 20.0, 32.0, 20.0, 8.0, 0.9, 5.0],
            [14.0, 20.0, 34.0, 20.0, 8.0, 0.8, 5.0],
            [40.0, 50.0, 60.0, 50.0, 8.0, 0.4, 14.0],
            [70.0, 80.0, 90.0, 80.0, 8.0, 0.3, 66.0],
        ],
        dtype=np.float32,
    )
    grasps = parse_rect_labels_array(rows)

    selected = select_balanced_rect_grasps(
        grasps, max_grasps=4, per_object_limit=2
    )

    assert [grasp.object_id for grasp in selected] == [5, 5, 14, 66]
    assert [grasp.score for grasp in selected] == [1.0, 0.9, 0.4, 0.3]


def test_export_rect_labels_visualization_writes_overlay_and_summary(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    rect_labels_path = tmp_path / "0000.npy"
    overlay_path = tmp_path / "rect_labels_overlay.png"
    summary_path = tmp_path / "rect_labels_summary.md"
    fields_table_path = tmp_path / "rect_labels_fields_table.png"

    Image.new("RGB", (80, 60), color=(80, 80, 80)).save(rgb_path)
    np.save(
        rect_labels_path,
        np.array(
            [
                [10.0, 20.0, 30.0, 20.0, 8.0, 0.9, 5.0],
                [40.0, 35.0, 60.0, 35.0, 10.0, 0.4, 14.0],
            ],
            dtype=np.float32,
        ),
    )

    overlay, summary, fields_table = export_rect_labels_visualization(
        rgb_path,
        rect_labels_path,
        overlay_path,
        summary_path,
        fields_table_path,
        max_grasps=2,
    )

    assert overlay == overlay_path
    assert summary == summary_path
    assert fields_table == fields_table_path
    with Image.open(overlay_path) as image:
        assert image.mode == "RGB"
        assert image.size == (80, 60)
    with Image.open(fields_table_path) as image:
        assert image.mode == "RGB"
        assert image.width > 800
    summary_text = summary_path.read_text(encoding="utf-8")
    assert summary_text.startswith("# Rect Labels Summary")
    assert "Total rows: 2" in summary_text
    assert "Displayed grasps: 2" in summary_text


def test_write_rect_labels_fields_table_writes_rgb_png(tmp_path):
    output_path = tmp_path / "rect_labels_fields_table.png"

    result = write_rect_labels_fields_table(output_path)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.width > image.height
        assert image.width > 800
