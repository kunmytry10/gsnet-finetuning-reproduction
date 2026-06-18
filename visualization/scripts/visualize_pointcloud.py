#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from visualization.src.pointcloud_visualization import export_sample_pointcloud
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
        description="Export a point cloud from the fixed AnyGraspNet sample frame."
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

    ply_path, preview_path = export_sample_pointcloud(paths)
    print(f"RGB source: {paths.rgb_path}")
    print(f"Depth source: {paths.depth_path}")
    print(f"Camera intrinsics: {paths.camK_path}")
    print(f"Point cloud output: {ply_path}")
    print(f"Point cloud preview: {preview_path}")


if __name__ == "__main__":
    main()
