from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/anygraspnet-matplotlib")

from matplotlib import pyplot as plt

from visualization.src.sample_paths import SamplePaths


@dataclass(frozen=True)
class AnnotationObject:
    obj_id: int
    obj_name: str
    obj_path: str
    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


def _parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split())


def _clean_mesh_name(name: str) -> str:
    return name.removesuffix(".ply")


def parse_annotations_xml(xml_path: str | Path) -> list[AnnotationObject]:
    xml_path = Path(xml_path)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects: list[AnnotationObject] = []

    for obj in root.findall("obj"):
        objects.append(
            AnnotationObject(
                obj_id=int(obj.findtext("obj_id", default="0")),
                obj_name=obj.findtext("obj_name", default=""),
                obj_path=obj.findtext("obj_path", default=""),
                position=_parse_floats(obj.findtext("pos_in_world", default="0 0 0")),  # type: ignore[arg-type]
                orientation=_parse_floats(
                    obj.findtext("ori_in_world", default="0 0 0 1")
                ),  # type: ignore[arg-type]
            )
        )

    return objects


def write_annotations_summary(
    objects: list[AnnotationObject], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Annotations Summary",
        "",
        "| obj_id | obj_name | x | y | z | qx | qy | qz | qw |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for obj in objects:
        x, y, z = obj.position
        qx, qy, qz, qw = obj.orientation
        lines.append(
            f"| {obj.obj_id} | {obj.obj_name} | {x:.4f} | {y:.4f} | {z:.4f} | "
            f"{qx:.4f} | {qy:.4f} | {qz:.4f} | {qw:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def write_annotations_table(
    objects: list[AnnotationObject], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for obj in objects:
        x, y, z = obj.position
        qx, qy, qz, qw = obj.orientation
        rows.append(
            [
                str(obj.obj_id),
                _clean_mesh_name(obj.obj_name),
                f"{x:.3f}",
                f"{y:.3f}",
                f"{z:.3f}",
                f"{qx:.2f}",
                f"{qy:.2f}",
                f"{qz:.2f}",
                f"{qw:.2f}",
            ]
        )

    fig, ax = plt.subplots(figsize=(15.0, max(2.8, 0.32 * len(rows) + 1.0)))
    fig.patch.set_facecolor("#fbfbf7")
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["id", "name", "x", "y", "z", "qx", "qy", "qz", "qw"],
        cellLoc="center",
        colWidths=[0.06, 0.22, 0.09, 0.09, 0.09, 0.085, 0.085, 0.085, 0.085],
        bbox=[0.02, 0.02, 0.96, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)

    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#d6d1c4")
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor("#263238")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1efe7")
        else:
            cell.set_facecolor("#fffdfa")

    ax.set_title(
        "Object Poses from annotations/0000.xml",
        pad=10,
        fontsize=14,
        weight="bold",
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def write_annotations_topdown(
    objects: list[AnnotationObject], output_path: str | Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xs = [obj.position[0] for obj in objects]
    ys = [obj.position[1] for obj in objects]

    fig, (ax, legend_ax) = plt.subplots(
        ncols=2,
        figsize=(10.8, 6.2),
        gridspec_kw={"width_ratios": [3.3, 1.25]},
    )
    fig.patch.set_facecolor("#fbfbf7")
    ax.set_facecolor("#fffdfa")

    colors = plt.get_cmap("tab10").colors
    point_colors = [colors[index % len(colors)] for index, _ in enumerate(objects)]
    ax.scatter(
        xs,
        ys,
        s=260,
        c=point_colors,
        edgecolors="white",
        linewidths=1.8,
        zorder=3,
    )
    for obj, x, y in zip(objects, xs, ys, strict=True):
        ax.text(
            x,
            y,
            str(obj.obj_id),
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            zorder=4,
        )

    x_pad = max(0.035, (max(xs) - min(xs)) * 0.12)
    y_pad = max(0.035, (max(ys) - min(ys)) * 0.18)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.axhline(0, color="#9e9e9e", linewidth=0.9, zorder=1)
    ax.axvline(0, color="#9e9e9e", linewidth=0.9, zorder=1)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("Top-down Object Positions", fontsize=14, weight="bold")
    ax.grid(True, linestyle="--", color="#d8d2c5", alpha=0.75, zorder=0)
    ax.set_aspect("equal", adjustable="box")

    legend_ax.axis("off")
    legend_ax.set_title("obj_id -> mesh", loc="left", fontsize=11, weight="bold")
    for index, obj in enumerate(objects):
        y = 0.94 - index * 0.095
        legend_ax.scatter(
            [0.04],
            [y],
            s=95,
            color=point_colors[index],
            edgecolors="white",
            linewidths=1.2,
            transform=legend_ax.transAxes,
        )
        legend_ax.text(
            0.10,
            y,
            f"{obj.obj_id}: {_clean_mesh_name(obj.obj_name)}",
            transform=legend_ax.transAxes,
            va="center",
            fontsize=8.8,
            color="#263238",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def export_sample_annotations(paths: SamplePaths) -> tuple[Path, Path, Path]:
    objects = parse_annotations_xml(paths.annotations_path)
    summary = write_annotations_summary(objects, paths.output_annotations_summary_path)
    table = write_annotations_table(objects, paths.output_annotations_table_path)
    topdown = write_annotations_topdown(objects, paths.output_annotations_topdown_path)
    return summary, table, topdown
