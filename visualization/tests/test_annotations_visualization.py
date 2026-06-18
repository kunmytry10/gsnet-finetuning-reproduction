from pathlib import Path

from PIL import Image

from visualization.src.annotations_visualization import (
    parse_annotations_xml,
    write_annotations_summary,
    write_annotations_table,
    write_annotations_topdown,
)
from visualization.src.sample_paths import DEFAULT_DATA_ROOT, build_sample_paths


def test_build_sample_paths_includes_annotations_outputs():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.annotations_path == Path(
        "/home/zky-miakho/datas/train_1/scene_0000/kinect/annotations/0000.xml"
    )
    assert paths.output_annotations_summary_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/annotations_summary.md"
    )
    assert paths.output_annotations_table_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/annotations_table.png"
    )
    assert paths.output_annotations_topdown_path == Path(
        "visualization/outputs/scene_0000_kinect_0000/annotations_topdown.png"
    )


def test_parse_annotations_xml_reads_object_pose_fields(tmp_path):
    xml_path = tmp_path / "0000.xml"
    xml_path.write_text(
        """<?xml version="1.0" ?>
<scene>
    <obj>
        <obj_id>14</obj_id>
        <obj_name>015_peach.ply</obj_name>
        <obj_path>models/015_peach.ply</obj_path>
        <pos_in_world>-0.1661 0.0837 0.4964</pos_in_world>
        <ori_in_world>0.1943 -0.9781 -0.0634 0.0406</ori_in_world>
    </obj>
</scene>
""",
        encoding="utf-8",
    )

    objects = parse_annotations_xml(xml_path)

    assert len(objects) == 1
    assert objects[0].obj_id == 14
    assert objects[0].obj_name == "015_peach.ply"
    assert objects[0].position == (-0.1661, 0.0837, 0.4964)
    assert objects[0].orientation == (0.1943, -0.9781, -0.0634, 0.0406)


def test_write_annotations_outputs_create_files(tmp_path):
    xml_path = Path("/home/zky-miakho/datas/train_1/scene_0000/kinect/annotations/0000.xml")
    objects = parse_annotations_xml(xml_path)
    summary_path = tmp_path / "annotations_summary.md"
    table_path = tmp_path / "annotations_table.png"
    topdown_path = tmp_path / "annotations_topdown.png"

    write_annotations_summary(objects, summary_path)
    write_annotations_table(objects, table_path)
    write_annotations_topdown(objects, topdown_path)

    assert summary_path.read_text(encoding="utf-8").startswith("# Annotations Summary")
    for path in [table_path, topdown_path]:
        with Image.open(path) as image:
            assert image.width > 0
            assert image.height > 0


def test_annotations_topdown_reserves_space_for_legend(tmp_path):
    xml_path = Path("/home/zky-miakho/datas/train_1/scene_0000/kinect/annotations/0000.xml")
    objects = parse_annotations_xml(xml_path)
    topdown_path = tmp_path / "annotations_topdown.png"

    write_annotations_topdown(objects, topdown_path)

    with Image.open(topdown_path) as image:
        assert image.width > image.height


def test_annotations_table_uses_wide_layout_for_long_names(tmp_path):
    xml_path = Path("/home/zky-miakho/datas/train_1/scene_0000/kinect/annotations/0000.xml")
    objects = parse_annotations_xml(xml_path)
    table_path = tmp_path / "annotations_table.png"

    write_annotations_table(objects, table_path)

    with Image.open(table_path) as image:
        assert image.width / image.height > 2.6
