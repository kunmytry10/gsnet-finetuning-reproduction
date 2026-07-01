from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from gsnet_reproduction.scripts.run_single_frame_inference import (
    _composite_grippers_over_cloud,
    _crop_render_to_content,
    _mesh_unique_edges,
    assign_grasp_center_labels,
    balance_grasps_by_label,
    build_grasp_control_points,
    build_frame_paths,
    build_mode_output_paths,
    build_output_dir,
    build_render_output_paths,
    canonical_frame,
    canonical_scene,
    project_camera_points,
    sample_point_cloud,
    save_report_render_contact_sheet,
    select_visualization_points,
)


class SingleFrameInferenceHelpersTest(unittest.TestCase):
    def test_canonical_scene_and_frame_accept_prefixed_or_numeric_values(self) -> None:
        self.assertEqual(canonical_scene("scene_0000"), "scene_0000")
        self.assertEqual(canonical_scene("0"), "scene_0000")
        self.assertEqual(canonical_scene("42"), "scene_0042")
        self.assertEqual(canonical_frame("0000"), "0000")
        self.assertEqual(canonical_frame("7"), "0007")

    def test_build_frame_paths_targets_raw_train_split_layout(self) -> None:
        paths = build_frame_paths(
            dataset_root=Path("/dataset"),
            split="train_1",
            scene="0",
            camera="kinect",
            frame="7",
        )

        self.assertEqual(paths.scene, "scene_0000")
        self.assertEqual(paths.frame, "0007")
        self.assertEqual(
            paths.depth_path,
            Path("/dataset/train_1/scene_0000/kinect/depth/0007.png"),
        )
        self.assertEqual(
            paths.meta_path,
            Path("/dataset/train_1/scene_0000/kinect/meta/0007.mat"),
        )

    def test_build_output_dir_uses_stable_scene_camera_frame_name(self) -> None:
        self.assertEqual(
            build_output_dir(Path("out"), "scene_0000", "kinect", "0000"),
            Path("out/single_frame_scene_0000_kinect_0000"),
        )

    def test_sample_point_cloud_downsamples_without_replacement(self) -> None:
        points = np.arange(18, dtype=np.float32).reshape(6, 3)
        sampled, indices = sample_point_cloud(points, 4, np.random.default_rng(123))

        self.assertEqual(sampled.shape, (4, 3))
        self.assertEqual(indices.shape, (4,))
        self.assertEqual(len(set(indices.tolist())), 4)
        np.testing.assert_array_equal(sampled, points[indices])

    def test_sample_point_cloud_upsamples_and_keeps_every_original_point(self) -> None:
        points = np.arange(6, dtype=np.float32).reshape(2, 3)
        sampled, indices = sample_point_cloud(points, 5, np.random.default_rng(123))

        self.assertEqual(sampled.shape, (5, 3))
        self.assertEqual(indices.shape, (5,))
        self.assertTrue({0, 1}.issubset(set(indices.tolist())))
        np.testing.assert_array_equal(sampled, points[indices])

    def test_sample_point_cloud_rejects_empty_input(self) -> None:
        points = np.empty((0, 3), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "no points"):
            sample_point_cloud(points, 10, np.random.default_rng(123))

    def test_select_visualization_points_keeps_dense_cloud_when_under_limit(self) -> None:
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        colors = np.arange(12, dtype=np.uint8).reshape(4, 3)

        sampled, sampled_colors, indices = select_visualization_points(
            points,
            colors,
            max_points=10,
            rng=np.random.default_rng(123),
        )

        np.testing.assert_array_equal(sampled, points)
        np.testing.assert_array_equal(sampled_colors, colors)
        np.testing.assert_array_equal(indices, np.arange(4))

    def test_select_visualization_points_downsamples_with_matching_colors(self) -> None:
        points = np.arange(30, dtype=np.float32).reshape(10, 3)
        colors = np.arange(30, dtype=np.uint8).reshape(10, 3)

        sampled, sampled_colors, indices = select_visualization_points(
            points,
            colors,
            max_points=4,
            rng=np.random.default_rng(123),
        )

        self.assertEqual(sampled.shape, (4, 3))
        self.assertEqual(sampled_colors.shape, (4, 3))
        self.assertEqual(len(set(indices.tolist())), 4)
        np.testing.assert_array_equal(sampled, points[indices])
        np.testing.assert_array_equal(sampled_colors, colors[indices])

    def test_build_render_output_paths_keeps_only_top_view(self) -> None:
        self.assertEqual(
            build_render_output_paths(Path("out")),
            {
                "top": Path("out/grasp_result_top.png"),
            },
        )

    def test_project_camera_points_uses_pinhole_intrinsics(self) -> None:
        points = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 2.0]], dtype=np.float32)
        intrinsic = np.array(
            [[100.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        pixels = project_camera_points(points, intrinsic)

        np.testing.assert_allclose(
            pixels,
            np.array([[320.0, 240.0], [325.0, 220.0]], dtype=np.float32),
        )

    def test_build_grasp_control_points_matches_graspnet_array_layout(self) -> None:
        grasp = np.array(
            [
                0.9,
                0.10,
                0.02,
                0.03,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
            ],
            dtype=np.float32,
        )

        points = build_grasp_control_points(grasp)

        np.testing.assert_allclose(points[0], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(points[1], [0.03, 0.05, 1.0])
        np.testing.assert_allclose(points[2], [0.03, -0.05, 1.0])
        np.testing.assert_allclose(points[3], [-0.02, 0.05, 1.0])
        np.testing.assert_allclose(points[4], [-0.02, -0.05, 1.0])
        np.testing.assert_allclose(points[5], [-0.06, 0.0, 1.0])

    def test_composite_grippers_uses_background_color_not_absolute_white(self) -> None:
        cloud = np.zeros((2, 2, 3), dtype=np.uint8)
        cloud[:, :] = [10, 120, 200]
        grippers = np.zeros((2, 2, 3), dtype=np.uint8)
        grippers[:, :] = [235, 235, 235]
        grippers[1, 1] = [180, 0, 180]

        composite = _composite_grippers_over_cloud(cloud, grippers)

        np.testing.assert_array_equal(composite[0, 0], cloud[0, 0])
        np.testing.assert_array_equal(composite[1, 1], grippers[1, 1])

    def test_crop_render_to_content_removes_large_empty_border(self) -> None:
        image = np.full((100, 120, 3), 255, dtype=np.uint8)
        image[40:60, 45:75] = [20, 100, 180]

        cropped = _crop_render_to_content(image, padding_px=5)

        self.assertLess(cropped.shape[0], image.shape[0])
        self.assertLess(cropped.shape[1], image.shape[1])
        np.testing.assert_array_equal(cropped[5:25, 5:35], image[40:60, 45:75])

    def test_save_report_render_contact_sheet_concatenates_pngs(self) -> None:
        from PIL import Image

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            left = tmp / "left.png"
            right = tmp / "right.png"
            output = tmp / "sheet.png"

            Image.fromarray(np.full((20, 30, 3), [255, 0, 0], dtype=np.uint8)).save(left)
            Image.fromarray(np.full((10, 40, 3), [0, 255, 0], dtype=np.uint8)).save(right)

            saved = save_report_render_contact_sheet([left, right], output)

            self.assertEqual(saved, output)
            sheet = np.array(Image.open(output).convert("RGB"))
            self.assertEqual(sheet.shape, (20, 70, 3))
            np.testing.assert_array_equal(sheet[0, 0], np.array([255, 0, 0], dtype=np.uint8))
            np.testing.assert_array_equal(sheet[5, 50], np.array([0, 255, 0], dtype=np.uint8))

    def test_mesh_unique_edges_deduplicates_triangle_edges(self) -> None:
        triangles = np.array([[0, 1, 2], [2, 1, 3]], dtype=np.int32)

        edges = _mesh_unique_edges(triangles)

        np.testing.assert_array_equal(
            edges,
            np.array(
                [
                    [0, 1],
                    [0, 2],
                    [1, 2],
                    [1, 3],
                    [2, 3],
                ],
                dtype=np.int32,
            ),
        )

    def test_build_mode_output_paths_names_default_collision_and_balanced_outputs(self) -> None:
        self.assertEqual(
            build_mode_output_paths(Path("out"), "top30").grasps_path,
            Path("out/top30_grasps.npy"),
        )
        self.assertEqual(
            build_mode_output_paths(Path("out"), "collision").rgb_overlay_path,
            Path("out/grasp_result_collision_rgb_overlay.png"),
        )
        self.assertEqual(
            build_mode_output_paths(Path("out"), "balanced").top_render_path,
            Path("out/grasp_result_balanced_top.png"),
        )

    def test_assign_grasp_center_labels_projects_centers_to_label_image(self) -> None:
        grasps = np.zeros((4, 17), dtype=np.float32)
        grasps[:, 4:13] = np.eye(3, dtype=np.float32).reshape(1, 9)
        grasps[:, 13:16] = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        intrinsic = np.array(
            [[10.0, 0.0, 1.0], [0.0, 10.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        label = np.zeros((3, 3), dtype=np.uint8)
        label[1, 1] = 4
        label[1, 2] = 7

        assigned = assign_grasp_center_labels(grasps, intrinsic, label)

        np.testing.assert_array_equal(assigned, np.array([4, 7, -1, -1]))

    def test_balance_grasps_by_label_spreads_quota_then_fills_by_score(self) -> None:
        scores = np.array([0.99, 0.95, 0.94, 0.80, 0.70, 0.60], dtype=np.float32)
        grasps = np.zeros((6, 17), dtype=np.float32)
        grasps[:, 0] = scores
        labels = np.array([1, 1, 1, 2, 2, 3], dtype=np.int32)

        indices = balance_grasps_by_label(grasps, labels, topk=5)

        np.testing.assert_array_equal(indices, np.array([0, 3, 5, 1, 2]))

    def test_balance_grasps_by_label_falls_back_to_global_topk_without_labels(self) -> None:
        grasps = np.zeros((4, 17), dtype=np.float32)
        labels = np.array([-1, 0, -1, 0], dtype=np.int32)

        indices = balance_grasps_by_label(grasps, labels, topk=3)

        np.testing.assert_array_equal(indices, np.array([0, 1, 2]))


if __name__ == "__main__":
    unittest.main()
