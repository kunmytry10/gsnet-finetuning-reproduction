#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from visualization.src.depth_visualization import (
    export_depth_colormap,
    export_depth_hamilton_colormap,
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
        description="Export a colorized depth image for the fixed AnyGraspNet sample frame."
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
        "--mode",
        choices=("viridis", "hamilton", "both"),
        default="both",
        help="Depth colorization mode. Defaults to exporting both viridis and Hamilton outputs.",
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

    print(f"Depth source: {paths.depth_path}")
    if args.mode in ("viridis", "both"):
        output_path = export_depth_colormap(
            paths.depth_path, paths.output_depth_colormap_path
        )
        print(f"Depth viridis output: {output_path}")
    if args.mode in ("hamilton", "both"):
        hamilton_output_path = export_depth_hamilton_colormap(
            paths.depth_path, paths.output_depth_hamilton_colormap_path
        )
        print(f"Depth Hamilton output: {hamilton_output_path}")


if __name__ == "__main__":
    main()
