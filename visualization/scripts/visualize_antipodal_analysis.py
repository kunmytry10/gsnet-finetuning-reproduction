#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from visualization.src.antipodal_analysis import export_antipodal_visualization
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
        description="Export antipodal-analysis visualizations on a real object model."
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
        help="Object id to analyze from grasp_label/*.npz.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-model-points",
        type=int,
        default=9000,
        help="Maximum PLY vertices used for plotting and normal estimation.",
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

    concept, positive_negative, top_grasp = export_antipodal_visualization(
        paths,
        object_id=args.object_id,
        object_name=object_name,
        max_model_points=args.max_model_points,
    )
    print(f"Grasp label source: {paths.grasp_label_path}")
    print(f"Object model source: {paths.grasp_object_model_path}")
    if object_name:
        print(f"Object name: {object_name}")
    print(f"Antipodal concept output: {concept}")
    print(f"Positive/negative antipodal output: {positive_negative}")
    print(f"Top grasp antipodal analysis output: {top_grasp}")


if __name__ == "__main__":
    main()
