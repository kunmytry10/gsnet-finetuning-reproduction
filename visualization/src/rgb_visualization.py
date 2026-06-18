from __future__ import annotations

from pathlib import Path

from PIL import Image


def export_rgb_image(source_path: str | Path, output_path: str | Path) -> Path:
    source_path = Path(source_path)
    output_path = Path(output_path)

    if not source_path.exists():
        raise FileNotFoundError(f"RGB image not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image.convert("RGB").save(output_path)

    return output_path
