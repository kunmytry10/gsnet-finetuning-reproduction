#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from visualization.src.sample_paths import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SCENE,
    DEFAULT_SPLIT,
    build_sample_paths,
)
from visualization.src.topk_grasp_overlay import export_sample_topk_grasp_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay top-k object-level 3D grasp candidates on RGB."
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
    parser.add_argument("--topk-per-object", type=int, default=3)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--min-quality", type=float, default=None)
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
    output_path = export_sample_topk_grasp_overlay(
        paths,
        topk_per_object=args.topk_per_object,
        frame_index=args.frame_index,
        min_quality=args.min_quality,
    )
    print(f"RGB source: {paths.rgb_path}")
    print(f"Annotations source: {paths.annotations_path}")
    print(f"Grasp label root: {paths.data_root / 'grasp_label'}")
    print(f"Top-k 3D grasp overlay output: {output_path}")


if __name__ == "__main__":
    main()
