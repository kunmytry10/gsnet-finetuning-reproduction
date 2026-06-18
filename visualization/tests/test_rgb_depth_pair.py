from PIL import Image

from visualization.src.rgb_depth_pair import export_rgb_depth_pair


def test_export_rgb_depth_pair_writes_side_by_side_image(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth_colormap.png"
    output_path = tmp_path / "rgb_depth_pair.png"
    Image.new("RGB", (4, 3), color=(255, 0, 0)).save(rgb_path)
    Image.new("RGB", (4, 3), color=(0, 255, 0)).save(depth_path)

    result = export_rgb_depth_pair(rgb_path, depth_path, output_path)

    assert result == output_path
    with Image.open(output_path) as image:
        assert image.mode == "RGB"
        assert image.size == (8, 3)
        assert image.getpixel((0, 0)) == (255, 0, 0)
        assert image.getpixel((7, 0)) == (0, 255, 0)
