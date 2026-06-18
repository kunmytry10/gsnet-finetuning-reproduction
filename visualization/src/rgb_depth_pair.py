from __future__ import annotations

from pathlib import Path

from PIL import Image

from visualization.src.depth_visualization import export_depth_colormap
from visualization.src.rgb_visualization import export_rgb_image
from visualization.src.sample_paths import SamplePaths


def export_rgb_depth_pair(
    rgb_path: str | Path, depth_colormap_path: str | Path, output_path: str | Path
) -> Path:
    rgb_path = Path(rgb_path)
    depth_colormap_path = Path(depth_colormap_path)
    output_path = Path(output_path)

    with Image.open(rgb_path) as rgb_image:
        rgb = rgb_image.convert("RGB")
        with Image.open(depth_colormap_path) as depth_image:
            depth = depth_image.convert("RGB").resize(rgb.size)

            paired = Image.new("RGB", (rgb.width + depth.width, rgb.height))
            paired.paste(rgb, (0, 0))
            paired.paste(depth, (rgb.width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    paired.save(output_path)
    return output_path


def export_sample_rgb_depth_pair(paths: SamplePaths) -> Path:
    export_rgb_image(paths.rgb_path, paths.output_rgb_path)
    export_depth_colormap(paths.depth_path, paths.output_depth_colormap_path)
    return export_rgb_depth_pair(
        paths.output_rgb_path,
        paths.output_depth_colormap_path,
        paths.output_rgb_depth_pair_path,
    )
