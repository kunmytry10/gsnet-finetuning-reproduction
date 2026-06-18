#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from visualization.src.rect_labels_visualization import export_rect_labels_visualization
from visualization.src.sample_paths import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCENE,
    DEFAULT_SPLIT,
    build_sample_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a rect_labels grasp-rectangle overlay for a sample frame."
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
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=100,
        help="Maximum number of highest-score grasp rectangles to draw.",
    )
    parser.add_argument(
        "--selection",
        choices=["balanced", "top-score"],
        default="balanced",
        help="Selection mode. balanced draws high-score examples from each object.",
    )
    parser.add_argument(
        "--per-object-limit",
        type=int,
        default=12,
        help="Maximum rectangles drawn per object when --selection balanced.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = build_sample_paths(
        data_root=Path(args.data_root),
        split=args.split,
        scene=args.scene,
        camera=args.camera,
        frame=args.frame,
        output_root=Path(args.output_root),
    )

    overlay, summary, fields_table = export_rect_labels_visualization(
        paths.rgb_path,
        paths.rect_labels_path,
        paths.output_rect_labels_overlay_path,
        paths.output_rect_labels_summary_path,
        paths.output_rect_labels_fields_table_path,
        max_grasps=args.max_grasps,
        selection=args.selection,
        per_object_limit=args.per_object_limit,
    )
    print(f"RGB source: {paths.rgb_path}")
    print(f"Rect labels source: {paths.rect_labels_path}")
    print(f"Rect labels overlay output: {overlay}")
    print(f"Rect labels summary output: {summary}")
    print(f"Rect labels fields table output: {fields_table}")


if __name__ == "__main__":
    main()
