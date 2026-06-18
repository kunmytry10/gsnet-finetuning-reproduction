from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/home/zky-miakho/datas")
DEFAULT_SPLIT = "train_1"
DEFAULT_SCENE = "scene_0000"
DEFAULT_CAMERA = "kinect"
DEFAULT_FRAME = "0000"
DEFAULT_GRASP_OBJECT_ID = 14
DEFAULT_OUTPUT_ROOT = Path("visualization/outputs")


@dataclass(frozen=True)
class SamplePaths:
    data_root: Path
    split: str
    scene: str
    camera: str
    frame: str
    rgb_path: Path
    depth_path: Path
    label_path: Path
    rect_labels_path: Path
    collision_label_path: Path
    grasp_label_path: Path
    grasp_object_model_path: Path
    annotations_path: Path
    camK_path: Path
    camera_poses_path: Path
    output_dir: Path
    output_rgb_path: Path
    output_depth_colormap_path: Path
    output_depth_hamilton_colormap_path: Path
    output_rgb_depth_pair_path: Path
    output_label_overlay_path: Path
    output_rect_labels_overlay_path: Path
    output_rect_labels_summary_path: Path
    output_rect_labels_fields_table_path: Path
    output_collision_label_structure_table_path: Path
    output_collision_label_collision_rates_path: Path
    output_collision_filtered_topk_overlay_path: Path
    output_grasp_label_structure_table_path: Path
    output_grasp_label_points_preview_path: Path
    output_grasp_label_graspness_heatmap_path: Path
    output_grasp_label_300_directions_path: Path
    output_grasp_label_12x4_explanation_path: Path
    output_pointcloud_path: Path
    output_pointcloud_preview_path: Path
    output_annotations_summary_path: Path
    output_annotations_table_path: Path
    output_annotations_topdown_path: Path
    output_pose_axes_overlay_path: Path
    output_topk_grasp_overlay_path: Path
    output_antipodal_concept_path: Path
    output_antipodal_positive_negative_path: Path
    output_antipodal_top_grasp_analysis_path: Path


def build_sample_paths(
    data_root: str | Path = DEFAULT_DATA_ROOT,
    split: str = DEFAULT_SPLIT,
    scene: str = DEFAULT_SCENE,
    camera: str = DEFAULT_CAMERA,
    frame: str = DEFAULT_FRAME,
    grasp_object_id: int = DEFAULT_GRASP_OBJECT_ID,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> SamplePaths:
    data_root = Path(data_root)
    output_dir = Path(output_root) / f"{scene}_{camera}_{frame}"
    frame_png = f"{frame}.png"

    return SamplePaths(
        data_root=data_root,
        split=split,
        scene=scene,
        camera=camera,
        frame=frame,
        rgb_path=data_root / split / scene / camera / "rgb" / frame_png,
        depth_path=data_root / split / scene / camera / "depth" / frame_png,
        label_path=data_root / split / scene / camera / "label" / frame_png,
        rect_labels_path=data_root
        / "rect_labels"
        / scene
        / camera
        / f"{frame}.npy",
        collision_label_path=data_root
        / "collision_label"
        / scene
        / "collision_labels.npz",
        grasp_label_path=data_root
        / "grasp_label"
        / f"{grasp_object_id:03d}_labels.npz",
        grasp_object_model_path=data_root
        / "models-xt19"
        / "models"
        / f"{grasp_object_id:03d}"
        / "nontextured_simplified.ply",
        annotations_path=data_root
        / split
        / scene
        / camera
        / "annotations"
        / f"{frame}.xml",
        camK_path=data_root / split / scene / camera / "camK.npy",
        camera_poses_path=data_root / split / scene / camera / "camera_poses.npy",
        output_dir=output_dir,
        output_rgb_path=output_dir / "rgb.png",
        output_depth_colormap_path=output_dir / "depth_colormap.png",
        output_depth_hamilton_colormap_path=output_dir / "depth_hamilton_colormap.png",
        output_rgb_depth_pair_path=output_dir / "rgb_depth_pair.png",
        output_label_overlay_path=output_dir / "label_overlay.png",
        output_rect_labels_overlay_path=output_dir / "rect_labels_overlay.png",
        output_rect_labels_summary_path=output_dir / "rect_labels_summary.md",
        output_rect_labels_fields_table_path=output_dir / "rect_labels_fields_table.png",
        output_collision_label_structure_table_path=output_dir
        / "collision_label_structure_table.png",
        output_collision_label_collision_rates_path=output_dir
        / "collision_label_collision_rates.png",
        output_collision_filtered_topk_overlay_path=output_dir
        / "collision_filtered_topk_grasps_rgb_overlay.png",
        output_grasp_label_structure_table_path=output_dir
        / "grasp_label_structure_table.png",
        output_grasp_label_points_preview_path=output_dir
        / "grasp_label_points_preview.png",
        output_grasp_label_graspness_heatmap_path=output_dir
        / "grasp_label_graspness_heatmap.png",
        output_grasp_label_300_directions_path=output_dir
        / "grasp_label_300_directions.png",
        output_grasp_label_12x4_explanation_path=output_dir
        / "grasp_label_12x4_explanation.png",
        output_pointcloud_path=output_dir / "pointcloud_rgb.ply",
        output_pointcloud_preview_path=output_dir / "pointcloud_preview.png",
        output_annotations_summary_path=output_dir / "annotations_summary.md",
        output_annotations_table_path=output_dir / "annotations_table.png",
        output_annotations_topdown_path=output_dir / "annotations_topdown.png",
        output_pose_axes_overlay_path=output_dir / "pose_axes_overlay.png",
        output_topk_grasp_overlay_path=output_dir / "topk_3d_grasps_rgb_overlay.png",
        output_antipodal_concept_path=output_dir
        / "antipodal_on_real_object_concept.png",
        output_antipodal_positive_negative_path=output_dir
        / "antipodal_positive_negative_on_object.png",
        output_antipodal_top_grasp_analysis_path=output_dir
        / "antipodal_top_grasp_analysis.png",
    )
