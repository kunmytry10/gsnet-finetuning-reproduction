from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import cv2
import numpy as np
from matplotlib import font_manager
from matplotlib import pyplot as plt
from PIL import Image

from visualization.src.annotations_visualization import (
    AnnotationObject,
    parse_annotations_xml,
)
from visualization.src.grasp_label_visualization import load_grasp_label
from visualization.src.topk_grasp_overlay import (
    _draw_legend,
    _draw_projected_gripper,
    _object_color,
    _point_visibility_priorities,
    _project_gripper,
    build_gripper_control_points,
    generate_fibonacci_sphere_directions,
    select_top_grasp_candidates,
)
from visualization.src.sample_paths import SamplePaths


CJK_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
CJK_FONT = (
    font_manager.FontProperties(fname=str(CJK_FONT_PATH))
    if CJK_FONT_PATH.exists()
    else font_manager.FontProperties()
)


@dataclass(frozen=True)
class CollisionObjectStats:
    key: str
    obj_id: int
    obj_name: str
    shape: tuple[int, ...]
    total_count: int
    collision_count: int
    collision_free_count: int
    valid_count: int
    collision_free_valid_count: int

    @property
    def collision_ratio(self) -> float:
        return self.collision_count / self.total_count if self.total_count else 0.0

    @property
    def collision_free_ratio(self) -> float:
        return (
            self.collision_free_count / self.total_count if self.total_count else 0.0
        )

    @property
    def collision_free_valid_ratio(self) -> float:
        return (
            self.collision_free_valid_count / self.valid_count
            if self.valid_count
            else 0.0
        )


def load_scene_collision_labels(collision_label_path: str | Path) -> dict[str, np.ndarray]:
    collision_label_path = Path(collision_label_path)
    if not collision_label_path.exists():
        raise FileNotFoundError(f"Collision label file not found: {collision_label_path}")

    data = np.load(collision_label_path)
    return {key: data[key].astype(bool) for key in data.files}


def map_collision_labels_to_objects(
    collision_labels: dict[str, np.ndarray],
    objects: list[AnnotationObject],
) -> dict[int, np.ndarray]:
    mapped: dict[int, np.ndarray] = {}
    for index, obj in enumerate(objects):
        key = f"arr_{index}"
        if key not in collision_labels:
            raise KeyError(f"Missing collision label key {key}")
        mapped[obj.obj_id] = collision_labels[key]
    return mapped


def collision_free_valid_mask(scores: np.ndarray, collision: np.ndarray) -> np.ndarray:
    if scores.shape != collision.shape:
        raise ValueError(
            f"scores and collision shapes must match, got {scores.shape} and {collision.shape}"
        )
    return (scores >= 0) & (~collision)


def summarize_collision_labels(
    objects: list[AnnotationObject],
    collision_by_object: dict[int, np.ndarray],
    scores_by_object: dict[int, np.ndarray],
) -> list[CollisionObjectStats]:
    stats: list[CollisionObjectStats] = []
    for index, obj in enumerate(objects):
        collision = collision_by_object[obj.obj_id]
        scores = scores_by_object[obj.obj_id]
        if collision.shape != scores.shape:
            raise ValueError(
                f"Shape mismatch for obj_id {obj.obj_id}: collision {collision.shape}, scores {scores.shape}"
            )
        valid = scores >= 0
        free_valid = collision_free_valid_mask(scores, collision)
        stats.append(
            CollisionObjectStats(
                key=f"arr_{index}",
                obj_id=obj.obj_id,
                obj_name=obj.obj_name,
                shape=tuple(collision.shape),
                total_count=int(collision.size),
                collision_count=int(collision.sum()),
                collision_free_count=int((~collision).sum()),
                valid_count=int(valid.sum()),
                collision_free_valid_count=int(free_valid.sum()),
            )
        )
    return stats


def _save_rgb_figure(fig, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(output_path) as image:
        image.convert("RGB").save(output_path)
    return output_path


def write_collision_label_structure_table(
    stats: list[CollisionObjectStats], output_path: str | Path
) -> Path:
    rows = [
        [
            stat.key,
            str(stat.obj_id),
            stat.obj_name.removesuffix(".ply"),
            str(stat.shape),
            f"{stat.collision_ratio:.3f}",
            f"{stat.collision_free_valid_ratio:.3f}",
        ]
        for stat in stats
    ]
    fig, ax = plt.subplots(figsize=(17.0, max(4.2, 0.42 * len(rows) + 1.2)))
    fig.patch.set_facecolor("#fbfbf7")
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "key",
            "obj_id",
            "object",
            "shape",
            "collision ratio",
            "free valid / valid",
        ],
        cellLoc="left",
        colWidths=[0.08, 0.08, 0.25, 0.28, 0.14, 0.16],
        bbox=[0.02, 0.03, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6d1c4")
        cell.set_linewidth(0.8)
        cell.PAD = 0.045
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold", fontproperties=CJK_FONT)
        elif row % 2 == 0:
            cell.set_facecolor("#f1efe7")
            cell.set_text_props(fontproperties=CJK_FONT)
        else:
            cell.set_facecolor("#fffdfa")
            cell.set_text_props(fontproperties=CJK_FONT)
    ax.set_title(
        "collision_label/scene_0000/collision_labels.npz 结构总览",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
        pad=10,
    )
    ax.text(
        0.02,
        0.90,
        "arr_i 对应 annotations 中第 i 个物体；True 表示该候选在当前场景中碰撞。",
        transform=ax.transAxes,
        fontsize=10,
        color="#455a64",
        fontproperties=CJK_FONT,
    )
    return _save_rgb_figure(fig, output_path)


def write_collision_label_collision_rates(
    stats: list[CollisionObjectStats], output_path: str | Path
) -> Path:
    names = [f"{stat.obj_id}:{stat.obj_name.removesuffix('.ply')}" for stat in stats]
    collision = [stat.collision_ratio for stat in stats]
    free_valid = [stat.collision_free_valid_ratio for stat in stats]
    x = np.arange(len(stats))
    width = 0.38

    fig, ax = plt.subplots(figsize=(16.0, 6.2))
    fig.patch.set_facecolor("#fbfbf7")
    ax.set_facecolor("#fffdfa")
    ax.bar(x - width / 2, collision, width, color="#d84315", label="collision / all")
    ax.bar(x + width / 2, free_valid, width, color="#00897b", label="free valid / valid")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ratio")
    ax.set_title(
        "每个物体的场景碰撞过滤比例",
        fontsize=15,
        weight="bold",
        fontproperties=CJK_FONT,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", color="#d8d2c5", alpha=0.75)
    ax.legend()
    plt.tight_layout()
    return _save_rgb_figure(fig, output_path)


def _draw_candidates_for_objects(
    image_bgr: np.ndarray,
    objects: list[AnnotationObject],
    paths: SamplePaths,
    topk_per_object: int,
    collision_by_object: dict[int, np.ndarray] | None,
) -> None:
    intrinsic = np.load(paths.camK_path)
    camera_pose = np.load(paths.camera_poses_path)[0]
    label_image = np.array(Image.open(paths.label_path))
    height, width = image_bgr.shape[:2]
    legend: list[tuple[int, str, tuple[int, int, int], int]] = []
    for object_index, obj in enumerate(objects):
        data = load_grasp_label(paths.data_root / "grasp_label" / f"{obj.obj_id:03d}_labels.npz")
        point_priorities = _point_visibility_priorities(
            data.points, obj, camera_pose, intrinsic, label_image
        )
        mask = None
        if collision_by_object is not None:
            mask = ~collision_by_object[obj.obj_id]
        candidates = select_top_grasp_candidates(
            data.scores,
            topk=topk_per_object,
            object_id=obj.obj_id,
            point_priorities=point_priorities,
            candidate_mask=mask,
        )
        if not candidates:
            legend.append((obj.obj_id, obj.obj_name, _object_color(object_index), 0))
            continue
        directions = generate_fibonacci_sphere_directions(data.scores.shape[1])
        color_bgr = _object_color(object_index)
        drawn = 0
        for candidate in candidates:
            control_points = build_gripper_control_points(
                data.points,
                data.offsets,
                candidate,
                directions[candidate.direction_index],
            )
            pixels = _project_gripper(control_points, obj, camera_pose, intrinsic)
            if pixels is None:
                continue
            if not np.all(
                (pixels[:, 0] >= -width * 0.2)
                & (pixels[:, 0] <= width * 1.2)
                & (pixels[:, 1] >= -height * 0.2)
                & (pixels[:, 1] <= height * 1.2)
            ):
                continue
            _draw_projected_gripper(image_bgr, pixels, color_bgr, candidate.quality)
            drawn += 1
        legend.append((obj.obj_id, obj.obj_name, color_bgr, drawn))
    _draw_legend(image_bgr, legend)


def write_collision_filtered_topk_overlay(
    paths: SamplePaths,
    output_path: str | Path,
    topk_per_object: int = 3,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(paths.rgb_path) as image:
        rgb = np.array(image.convert("RGB"))
    objects = parse_annotations_xml(paths.annotations_path)
    collision_labels = load_scene_collision_labels(paths.collision_label_path)
    collision_by_object = map_collision_labels_to_objects(collision_labels, objects)

    left = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    right = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    _draw_candidates_for_objects(left, objects, paths, topk_per_object, None)
    _draw_candidates_for_objects(
        right, objects, paths, topk_per_object, collision_by_object
    )
    cv2.putText(left, "score-only top-k", (24, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(right, "collision-free top-k", (24, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    combined_bgr = np.concatenate([left, right], axis=1)
    combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(combined_rgb).save(output_path)
    return output_path


def export_collision_label_visualization(
    paths: SamplePaths,
    topk_per_object: int = 3,
) -> tuple[Path, Path, Path]:
    objects = parse_annotations_xml(paths.annotations_path)
    collision_labels = load_scene_collision_labels(paths.collision_label_path)
    collision_by_object = map_collision_labels_to_objects(collision_labels, objects)
    scores_by_object = {
        obj.obj_id: load_grasp_label(
            paths.data_root / "grasp_label" / f"{obj.obj_id:03d}_labels.npz"
        ).scores
        for obj in objects
    }
    stats = summarize_collision_labels(objects, collision_by_object, scores_by_object)
    table = write_collision_label_structure_table(
        stats, paths.output_collision_label_structure_table_path
    )
    rates = write_collision_label_collision_rates(
        stats, paths.output_collision_label_collision_rates_path
    )
    overlay = write_collision_filtered_topk_overlay(
        paths,
        paths.output_collision_filtered_topk_overlay_path,
        topk_per_object=topk_per_object,
    )
    return table, rates, overlay
