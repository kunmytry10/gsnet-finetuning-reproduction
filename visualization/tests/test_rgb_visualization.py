from pathlib import Path

from PIL import Image

from visualization.src.rgb_visualization import export_rgb_image
from visualization.src.sample_paths import (
    DEFAULT_CAMERA,
    DEFAULT_DATA_ROOT,
    DEFAULT_FRAME,
    DEFAULT_SCENE,
    build_sample_paths,
)


def test_build_sample_paths_uses_default_scene_frame_and_camera():
    paths = build_sample_paths(DEFAULT_DATA_ROOT)

    assert paths.data_root == Path("/home/zky-miakho/datas")
    assert paths.scene == DEFAULT_SCENE
    assert paths.camera == DEFAULT_CAMERA
    assert paths.frame == DEFAULT_FRAME
    assert paths.rgb_path == Path(
        "/home/zky-miakho/datas/train_1/scene_0000/kinect/rgb/0000.png"
    )
    assert paths.output_dir == Path("visualization/outputs/scene_0000_kinect_0000")


def test_export_rgb_image_writes_png_with_same_dimensions(tmp_path):
    source = tmp_path / "source.png"
    output = tmp_path / "rgb.png"
    Image.new("RGB", (8, 6), color=(12, 34, 56)).save(source)

    result = export_rgb_image(source, output)

    assert result == output
    with Image.open(output) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 6)
