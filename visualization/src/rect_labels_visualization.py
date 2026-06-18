from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class RectGrasp:
    center: tuple[float, float]
    open_point: tuple[float, float]
    height: float
    score: float
    object_id: int

    @property
    def width(self) -> float:
        center_x, center_y = self.center
        open_x, open_y = self.open_point
        return 2.0 * math.hypot(open_x - center_x, open_y - center_y)

    @property
    def angle_degrees(self) -> float:
        center_x, center_y = self.center
        open_x, open_y = self.open_point
        return math.degrees(math.atan2(open_y - center_y, open_x - center_x))

    @property
    def corners(self) -> list[tuple[float, float]]:
        center_x, center_y = self.center
        open_x, open_y = self.open_point
        axis_x = open_x - center_x
        axis_y = open_y - center_y
        axis_norm = math.hypot(axis_x, axis_y)
        if axis_norm == 0.0:
            return []

        normal_x = -axis_y / axis_norm * self.height / 2.0
        normal_y = axis_x / axis_norm * self.height / 2.0
        return [
            (center_x + normal_x + axis_x, center_y + normal_y + axis_y),
            (center_x + normal_x - axis_x, center_y + normal_y - axis_y),
            (center_x - normal_x - axis_x, center_y - normal_y - axis_y),
            (center_x - normal_x + axis_x, center_y - normal_y + axis_y),
        ]


def parse_rect_labels_array(rect_labels: np.ndarray) -> list[RectGrasp]:
    if rect_labels.ndim != 2 or rect_labels.shape[1] != 7:
        raise ValueError(f"Expected rect_labels with shape (N, 7), got {rect_labels.shape}")

    grasps: list[RectGrasp] = []
    for row in rect_labels:
        center_x, center_y, open_x, open_y, height, score, object_id = row.tolist()
        grasps.append(
            RectGrasp(
                center=(float(center_x), float(center_y)),
                open_point=(float(open_x), float(open_y)),
                height=float(height),
                score=round(float(score), 4),
                object_id=int(round(float(object_id))),
            )
        )
    return grasps


def load_rect_labels(rect_labels_path: str | Path) -> list[RectGrasp]:
    rect_labels_path = Path(rect_labels_path)
    if not rect_labels_path.exists():
        raise FileNotFoundError(f"Rect labels file not found: {rect_labels_path}")

    return parse_rect_labels_array(np.load(rect_labels_path))


def select_top_rect_grasps(
    grasps: list[RectGrasp], max_grasps: int = 100
) -> list[RectGrasp]:
    if max_grasps <= 0:
        return []

    return sorted(grasps, key=lambda grasp: grasp.score, reverse=True)[:max_grasps]


def select_balanced_rect_grasps(
    grasps: list[RectGrasp],
    max_grasps: int = 100,
    per_object_limit: int = 12,
) -> list[RectGrasp]:
    if max_grasps <= 0 or per_object_limit <= 0:
        return []

    selected: list[RectGrasp] = []
    for object_id in sorted({grasp.object_id for grasp in grasps}):
        object_grasps = [grasp for grasp in grasps if grasp.object_id == object_id]
        selected.extend(
            sorted(object_grasps, key=lambda grasp: grasp.score, reverse=True)[
                :per_object_limit
            ]
        )

    return selected[:max_grasps]


def _score_color(score: float) -> tuple[int, int, int, int]:
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        t = score / 0.5
        return (255, int(180 + 55 * t), 40, 210)

    t = (score - 0.5) / 0.5
    return (int(255 - 215 * t), int(235 - 25 * t), int(40 + 120 * t), 230)


def _rectangle_corners(grasp: RectGrasp) -> list[tuple[float, float]]:
    return grasp.corners


def _clip_point(
    point: tuple[float, float], image_size: tuple[int, int]
) -> tuple[float, float]:
    width, height = image_size
    x, y = point
    return (max(0.0, min(float(width - 1), x)), max(0.0, min(float(height - 1), y)))


def _draw_rect_grasps(
    rgb_image: Image.Image, grasps: list[RectGrasp]
) -> Image.Image:
    base = rgb_image.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, mode="RGBA")

    for grasp in grasps:
        corners = _rectangle_corners(grasp)
        if not corners:
            continue

        corners = [_clip_point(point, base.size) for point in corners]
        open_point = _clip_point(grasp.open_point, base.size)
        center = _clip_point(grasp.center, base.size)
        color = _score_color(grasp.score)
        fill = (color[0], color[1], color[2], 34)

        draw.polygon(corners, fill=fill)
        draw.line(corners + [corners[0]], fill=color, width=3)
        draw.line([center, open_point], fill=(255, 255, 255, 200), width=1)
        radius = 3
        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            fill=(255, 255, 255, 230),
            outline=color,
            width=1,
        )

    return Image.alpha_composite(base, overlay).convert("RGB")


def write_rect_labels_summary(
    grasps: list[RectGrasp],
    displayed: list[RectGrasp],
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    object_counts = Counter(grasp.object_id for grasp in grasps)
    scores = [grasp.score for grasp in grasps]
    widths = [grasp.width for grasp in grasps]
    heights = [grasp.height for grasp in grasps]

    lines = [
        "# Rect Labels Summary",
        "",
        "Row format: `center_x, center_y, open_x, open_y, height, score, object_id`.",
        "",
        f"Total rows: {len(grasps)}",
        f"Displayed grasps: {len(displayed)}",
    ]
    if scores:
        lines.extend(
            [
                f"Score range: {min(scores):.1f} to {max(scores):.1f}",
                f"Width range: {min(widths):.2f} px to {max(widths):.2f} px",
                f"Height range: {min(heights):.2f} px to {max(heights):.2f} px",
                "",
                "| object_id | row_count |",
                "|---:|---:|",
            ]
        )
        for object_id, count in sorted(object_counts.items()):
            lines.append(f"| {object_id} | {count} |")

        lines.extend(
            [
                "",
                "| rank | score | object_id | center_x | center_y | angle_deg |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for rank, grasp in enumerate(displayed[:20], start=1):
            center_x, center_y = grasp.center
            lines.append(
                f"| {rank} | {grasp.score:.1f} | {grasp.object_id} | "
                f"{center_x:.1f} | {center_y:.1f} | {grasp.angle_degrees:.1f} |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_rect_labels_fields_table(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        [
            "center_x",
            "grasp center x",
            "pixel",
            "Center point of the 2D grasp rectangle.",
        ],
        [
            "center_y",
            "grasp center y",
            "pixel",
            "Center point of the 2D grasp rectangle.",
        ],
        [
            "open_x",
            "open point x",
            "pixel",
            "Point on the gripper opening direction.",
        ],
        [
            "open_y",
            "open point y",
            "pixel",
            "Point on the gripper opening direction.",
        ],
        [
            "height",
            "rectangle height",
            "pixel",
            "Short side length perpendicular to the opening axis.",
        ],
        [
            "score",
            "grasp quality",
            "0.1-1.0",
            "Higher values mark better 2D grasp candidates.",
        ],
        [
            "object_id",
            "object id",
            "integer",
            "Object id matched to the scene annotation.",
        ],
    ]

    fig, ax = plt.subplots(figsize=(14.5, 4.7))
    fig.patch.set_facecolor("#fbfbf7")
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["field", "meaning", "unit/range", "notes"],
        cellLoc="left",
        colWidths=[0.14, 0.20, 0.12, 0.50],
        bbox=[0.02, 0.03, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6d1c4")
        cell.set_linewidth(0.8)
        cell.PAD = 0.05
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold", ha="left")
        elif row % 2 == 0:
            cell.set_facecolor("#f1efe7")
        else:
            cell.set_facecolor("#fffdfa")
        if col in {0, 2} and row > 0:
            cell.set_text_props(weight="bold", color="#263238")

    ax.set_title(
        "rect_labels row format: center_x, center_y, open_x, open_y, height, score, object_id",
        pad=10,
        fontsize=13,
        weight="bold",
    )
    ax.text(
        0.02,
        0.91,
        "Width is derived as 2 x distance(center, open point).",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#455a64",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    with Image.open(output_path) as image:
        image.convert("RGB").save(output_path)
    return output_path


def export_rect_labels_visualization(
    rgb_path: str | Path,
    rect_labels_path: str | Path,
    output_overlay_path: str | Path,
    output_summary_path: str | Path,
    output_fields_table_path: str | Path,
    max_grasps: int = 100,
    selection: str = "balanced",
    per_object_limit: int = 12,
) -> tuple[Path, Path, Path]:
    rgb_path = Path(rgb_path)
    output_overlay_path = Path(output_overlay_path)

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")

    grasps = load_rect_labels(rect_labels_path)
    if selection == "balanced":
        displayed = select_balanced_rect_grasps(
            grasps, max_grasps=max_grasps, per_object_limit=per_object_limit
        )
    elif selection == "top-score":
        displayed = select_top_rect_grasps(grasps, max_grasps=max_grasps)
    else:
        raise ValueError(f"Unknown rect label selection mode: {selection}")

    output_overlay_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(rgb_path) as image:
        overlay = _draw_rect_grasps(image, displayed)
    overlay.save(output_overlay_path)

    summary = write_rect_labels_summary(grasps, displayed, output_summary_path)
    fields_table = write_rect_labels_fields_table(output_fields_table_path)
    return output_overlay_path, summary, fields_table
