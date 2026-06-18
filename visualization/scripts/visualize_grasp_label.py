#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from visualization.src.grasp_label_visualization import export_grasp_label_visualization
from visualization.src.sample_paths import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME,
    DEFAULT_GRASP_OBJECT_ID,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCENE,
    DEFAULT_SPLIT,
    build_sample_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export first-pass grasp_label visualizations."
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_DIR", str(DEFAULT_DATA_ROOT)),
        help="Dataset root. Defaults to DATA_DIR or /home/zky-miakho/datas.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument(
        "--object-id",
        type=int,
        default=DEFAULT_GRASP_OBJECT_ID,
        help="Object id to visualize from grasp_label/*.npz.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--point-index",
        type=int,
        default=None,
        help="Surface point index for 300-direction visualization. Defaults to highest graspness point.",
    )
    parser.add_argument(
        "--direction-length",
        type=float,
        default=None,
        help="Direction ray length in object coordinates. Defaults to a fraction of the object span.",
    )
    parser.add_argument(
        "--direction-index",
        type=int,
        default=None,
        help="Direction index for 12x4 explanation. Defaults to best-quality direction for the selected point.",
    )
    return parser.parse_args()


def _find_object_name(annotations_path: Path, object_id: int) -> str | None:
    if not annotations_path.exists():
        return None

    root = ET.parse(annotations_path).getroot()
    for obj in root.findall("obj"):
        if int(obj.findtext("obj_id", default="-1")) == object_id:
            return obj.findtext("obj_name", default=None)
    return None


def main() -> None:
    args = parse_args()
    paths = build_sample_paths(
        data_root=Path(args.data_root),
        split=args.split,
        scene=args.scene,
        camera=args.camera,
        frame=args.frame,
        grasp_object_id=args.object_id,
        output_root=Path(args.output_root),
    )
    object_name = _find_object_name(paths.annotations_path, args.object_id)

    structure, points, heatmap, directions, configs = export_grasp_label_visualization(
        paths.grasp_label_path,
        paths.output_grasp_label_structure_table_path,
        paths.output_grasp_label_points_preview_path,
        paths.output_grasp_label_graspness_heatmap_path,
        paths.output_grasp_label_300_directions_path,
        paths.output_grasp_label_12x4_explanation_path,
        object_id=args.object_id,
        object_name=object_name,
        object_model_path=paths.grasp_object_model_path,
        point_index=args.point_index,
        direction_length=args.direction_length,
        direction_index=args.direction_index,
    )
    print(f"Grasp label source: {paths.grasp_label_path}")
    print(f"Object model source: {paths.grasp_object_model_path}")
    if object_name:
        print(f"Object name: {object_name}")
    print(f"Structure table output: {structure}")
    print(f"Points preview output: {points}")
    print(f"Graspness heatmap output: {heatmap}")
    print(f"300 directions output: {directions}")
    print(f"12x4 explanation output: {configs}")


if __name__ == "__main__":
    main()
