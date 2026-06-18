#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from visualization.src.collision_label_visualization import (
    export_collision_label_visualization,
)
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
        description="Export collision_label structure and collision-filtered grasp visualizations."
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
    table, rates, overlay = export_collision_label_visualization(
        paths,
        topk_per_object=args.topk_per_object,
    )
    print(f"Collision label source: {paths.collision_label_path}")
    print(f"Structure table output: {table}")
    print(f"Collision rates output: {rates}")
    print(f"Collision-filtered overlay output: {overlay}")


if __name__ == "__main__":
    main()
