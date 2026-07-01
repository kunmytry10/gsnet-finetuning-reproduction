#!/usr/bin/env python3
"""Fine-tune GSNet and run a small automatic AP probe after selected epochs."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
GSNET_ROOT = REPO_ROOT / "external" / "graspness_unofficial"
SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "graspnet"
DEFAULT_CHECKPOINT = REPO_ROOT / "weights" / "kinect" / "minkuresunet_kinect.tar"
DEFAULT_LOG_ROOT = REPO_ROOT / "gsnet_reproduction" / "outputs" / "finetune_runs"
DEFAULT_RUN_NAME = "kinect_finetune_probe"

for path in (
    GSNET_ROOT,
    GSNET_ROOT / "pointnet2",
    GSNET_ROOT / "utils",
    GSNET_ROOT / "models",
    GSNET_ROOT / "dataset",
    SCRIPT_DIR,
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from dataset.graspnet_dataset import (  # noqa: E402
    GraspNetDataset,
    load_grasp_labels,
    minkowski_collate_fn,
)
from models.graspnet import GraspNet  # noqa: E402
from models.loss import get_loss  # noqa: E402
from eval_utils import parse_frame_spec, parse_scene_spec, write_json  # noqa: E402


@dataclass
class ProbeResult:
    epoch: int
    checkpoint_path: str
    metrics_percent: dict[str, float]
    elapsed_sec: float
    frame_count: int
    json_path: str


@dataclass
class FullEvalResult:
    checkpoint_path: str
    dump_dir: str
    cache_dir: str
    metrics_percent: dict[str, float]
    ap_path: str
    summary_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--camera", default="kinect", choices=["kinect", "realsense"])
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--num-point", type=int, default=15000)
    parser.add_argument("--seed-feat-dim", type=int, default=512)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-log-interval", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--val-scenes", default="100-102")
    parser.add_argument("--val-frames", default="0-31")
    parser.add_argument("--val-top-k", type=int, default=50)
    parser.add_argument("--val-split-dir", default="scenes")
    parser.add_argument("--val-num-points", type=int, default=15000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-init",
        action="store_true",
        default=False,
        help="Do not initialize from the official checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime_library_env() -> None:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return
    lib_dir = os.path.join(conda_prefix, "lib")
    lib64_dir = os.path.join(conda_prefix, "lib64")
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [lib_dir, lib64_dir]
    if current_ld_library_path:
        ld_parts.append(current_ld_library_path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    preload_parts = []
    for name in ("libstdc++.so.6", "libgcc_s.so.1"):
        candidate = os.path.join(lib_dir, name)
        if os.path.exists(candidate):
            preload_parts.append(candidate)
    current_preload = os.environ.get("LD_PRELOAD", "")
    if current_preload:
        preload_parts.append(current_preload)
    if preload_parts:
        os.environ["LD_PRELOAD"] = ":".join(preload_parts)


def my_worker_init_fn(worker_id: int) -> None:
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def log_string(log_fout: Any, out_str: str) -> None:
    log_fout.write(out_str + "\n")
    log_fout.flush()
    print(out_str, flush=True)


def get_current_lr(base_lr: float, epoch: int) -> float:
    return base_lr * (0.95 ** epoch)


def adjust_learning_rate(optimizer: optim.Optimizer, base_lr: float, epoch: int) -> float:
    lr = get_current_lr(base_lr, epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_checkpoint_path(log_dir: Path, run_name: str, epoch_index_1based: int) -> Path:
    return log_dir / f"{run_name}_epoch{epoch_index_1based:02d}.tar"


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def run_probe(
    *,
    epoch_index_1based: int,
    checkpoint_path: Path,
    dataset_root: Path,
    split_dir: str,
    camera: str,
    output_dir: Path,
    eval_num_points: int,
    voxel_size: float,
    seed_feat_dim: int,
    top_k: int,
    device: str,
    seed: int,
    scenes: list[int],
    frames: list[int],
) -> ProbeResult:
    probe_tag = f"epoch_{epoch_index_1based:02d}"
    probe_output_dir = output_dir / probe_tag
    probe_json_path = probe_output_dir / "summary.json"
    probe_output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath_parts = [
        str(GSNET_ROOT / "pointnet2"),
        str(GSNET_ROOT / "knn"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join([part for part in pythonpath_parts if part])

    probe_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_test_seen_probe.py"),
        "--dataset-root",
        str(dataset_root),
        "--split-dir",
        split_dir,
        "--scenes",
        ",".join(str(scene) for scene in scenes),
        "--frames",
        ",".join(str(frame) for frame in frames),
        "--camera",
        camera,
        "--official-checkpoint",
        str(checkpoint_path),
        "--candidate-checkpoint",
        str(checkpoint_path),
        "--candidate-name",
        probe_tag,
        "--output-dir",
        str(probe_output_dir),
        "--output-json",
        str(probe_json_path),
        "--eval-num-points",
        str(eval_num_points),
        "--voxel-size",
        str(voxel_size),
        "--seed-feat-dim",
        str(seed_feat_dim),
        "--top-k",
        str(top_k),
        "--device",
        device,
        "--seed",
        str(seed),
    ]

    started = time.perf_counter()
    subprocess.run(probe_cmd, check=True, env=env, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - started
    if not probe_json_path.exists():
        raise FileNotFoundError(f"missing probe summary: {probe_json_path}")
    probe_payload = json.loads(probe_json_path.read_text(encoding="utf-8"))
    summary = probe_payload["candidate"]["summary"]
    json_path = output_dir / f"{probe_tag}_summary.json"
    write_json(
        json_path,
        {
            "epoch": epoch_index_1based,
            "checkpoint_path": str(checkpoint_path),
            "summary": summary,
            "source_summary_json": str(probe_json_path),
            "frames": probe_payload["candidate"]["frames"],
        },
    )
    return ProbeResult(
        epoch=epoch_index_1based,
        checkpoint_path=str(checkpoint_path),
        metrics_percent=summary["metrics_percent"],
        elapsed_sec=elapsed,
        frame_count=len(probe_payload["candidate"]["frames"]),
        json_path=str(json_path),
    )


def run_full_test_seen_eval(
    *,
    checkpoint_path: Path,
    dataset_root: Path,
    camera: str,
    output_dir: Path,
    eval_num_points: int,
    voxel_size: float,
    seed_feat_dim: int,
    device: str,
    seed: int,
) -> FullEvalResult:
    dump_dir = output_dir / "dumps"
    cache_dir = output_dir / "scene_eval_test_seen_kinect"
    dump_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    pythonpath_parts = [
        str(GSNET_ROOT / "pointnet2"),
        str(GSNET_ROOT / "knn"),
        env.get("PYTHONPATH", ""),
    ]
    env["PYTHONPATH"] = ":".join([part for part in pythonpath_parts if part])

    infer_cmd = [
        sys.executable,
        str(GSNET_ROOT / "test.py"),
        "--camera",
        camera,
        "--split",
        "test_seen",
        "--dataset_root",
        str(dataset_root),
        "--checkpoint_path",
        str(checkpoint_path),
        "--dump_dir",
        str(dump_dir),
        "--batch_size",
        "1",
        "--infer",
    ]
    subprocess.run(infer_cmd, check=True, env=env, cwd=str(GSNET_ROOT))

    eval_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "eval_dumps_by_scene.py"),
        "--dataset_root",
        str(dataset_root),
        "--dump_dir",
        str(dump_dir),
        "--camera",
        camera,
        "--split",
        "test_seen",
        "--cache_dir",
        str(cache_dir),
    ]
    subprocess.run(eval_cmd, check=True, env=env, cwd=str(REPO_ROOT))

    summary_path = dump_dir / "ap_kinect_summary.json"
    ap_path = dump_dir / "ap_kinect.npy"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing full eval summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary["metrics_percent"]
    return FullEvalResult(
        checkpoint_path=str(checkpoint_path),
        dump_dir=str(dump_dir),
        cache_dir=str(cache_dir),
        metrics_percent=metrics,
        ap_path=str(ap_path),
        summary_path=str(summary_path),
    )


def main() -> int:
    args = parse_args()
    configure_runtime_library_env()
    set_seed(args.seed)

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    run_root = Path(args.log_root).expanduser().resolve() / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "train_logs"
    tb_train_dir = run_root / "tensorboard" / "train"
    tb_val_dir = run_root / "tensorboard" / "val"
    probe_dir = run_root / "val_probe"
    meta_json = run_root / "run_state.json"
    best_json = run_root / "best_probe.json"

    train_writer = SummaryWriter(str(tb_train_dir))
    val_writer = SummaryWriter(str(tb_val_dir))
    log_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    log_fout = open(log_dir / "log_train.txt", "a", encoding="utf-8")
    log_string(log_fout, str(vars(args)))

    grasp_labels = load_grasp_labels(str(dataset_root))
    train_dataset = GraspNetDataset(
        str(dataset_root),
        grasp_labels=grasp_labels,
        camera=args.camera,
        split="train",
        num_points=args.num_point,
        voxel_size=args.voxel_size,
        remove_outlier=True,
        augment=True,
        load_label=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=my_worker_init_fn,
        collate_fn=minkowski_collate_fn,
    )
    log_string(log_fout, f"train dataset length: {len(train_dataset)}")
    log_string(log_fout, f"train dataloader length: {len(train_dataloader)}")

    net = GraspNet(seed_feat_dim=args.seed_feat_dim, is_training=True)
    device = resolve_device(args.device)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    start_epoch = 0
    source_checkpoint = None if args.random_init else checkpoint_path
    if args.resume and source_checkpoint is not None and source_checkpoint.is_file():
        checkpoint = torch.load(str(source_checkpoint), map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint["epoch"])
        log_string(
            log_fout,
            f"-> resumed checkpoint {source_checkpoint} (epoch: {start_epoch})",
        )
    elif (not args.resume) and source_checkpoint is not None and source_checkpoint.is_file():
        checkpoint = torch.load(str(source_checkpoint), map_location=device)
        net.load_state_dict(checkpoint["model_state_dict"])
        log_string(
            log_fout,
            f"-> initialized model from checkpoint {source_checkpoint} (epoch: {checkpoint.get('epoch')})",
        )
    elif source_checkpoint is not None:
        raise FileNotFoundError(f"checkpoint not found: {source_checkpoint}")

    probe_scenes = parse_scene_spec(args.val_scenes)
    probe_frames = parse_frame_spec(args.val_frames)
    probe_history: list[dict[str, Any]] = []
    best_probe: ProbeResult | None = None

    def dump_run_state() -> None:
        payload = {
            "run_name": args.run_name,
            "dataset_root": str(dataset_root),
            "camera": args.camera,
            "run_root": str(run_root),
            "train_log_dir": str(log_dir),
            "tensorboard_train_dir": str(tb_train_dir),
            "tensorboard_val_dir": str(tb_val_dir),
            "probe_dir": str(probe_dir),
            "checkpoint_init": None if source_checkpoint is None else str(source_checkpoint),
            "probe_history": probe_history,
        }
        save_json(meta_json, payload)

    dump_run_state()

    for epoch in range(start_epoch, args.max_epoch):
        epoch_index_1based = epoch + 1
        lr = adjust_learning_rate(optimizer, args.learning_rate, epoch)
        log_string(log_fout, f"**** EPOCH {epoch:03d} ****")
        log_string(log_fout, f"Current learning rate: {lr:.8f}")
        net.train()
        stat_dict: dict[str, float] = {}
        batch_interval = args.train_log_interval

        for batch_idx, batch_data_label in enumerate(train_dataloader):
            for key in batch_data_label:
                if "list" in key:
                    for i in range(len(batch_data_label[key])):
                        for j in range(len(batch_data_label[key][i])):
                            batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
                else:
                    batch_data_label[key] = batch_data_label[key].to(device)

            end_points = net(batch_data_label)
            loss, end_points = get_loss(end_points)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            for key, value in end_points.items():
                if (
                    "loss" in key
                    or "acc" in key
                    or "prec" in key
                    or "recall" in key
                    or "count" in key
                ):
                    stat_dict[key] = stat_dict.get(key, 0.0) + float(value.item())

            if (batch_idx + 1) % batch_interval == 0:
                global_step = (epoch * len(train_dataloader) + batch_idx) * args.batch_size
                log_string(log_fout, f" ----epoch: {epoch:03d}  ---- batch: {batch_idx + 1:03d} ----")
                for key in sorted(stat_dict.keys()):
                    mean_value = stat_dict[key] / batch_interval
                    train_writer.add_scalar(key, mean_value, global_step)
                    log_string(log_fout, f"mean {key}: {mean_value:f}")
                    stat_dict[key] = 0.0

        checkpoint_to_eval: Path | None = None
        if epoch_index_1based % args.save_every == 0:
            checkpoint_to_eval = build_checkpoint_path(log_dir, args.run_name, epoch_index_1based)
            torch.save(
                {
                    "epoch": epoch_index_1based,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_state_dict": net.state_dict(),
                },
                checkpoint_to_eval,
            )
            log_string(log_fout, f"saved checkpoint: {checkpoint_to_eval}")

        if epoch_index_1based % args.val_every == 0:
            if checkpoint_to_eval is None:
                checkpoint_to_eval = build_checkpoint_path(log_dir, args.run_name, epoch_index_1based)
                torch.save(
                    {
                        "epoch": epoch_index_1based,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_state_dict": net.state_dict(),
                    },
                    checkpoint_to_eval,
                )
                log_string(log_fout, f"saved checkpoint for probe: {checkpoint_to_eval}")

            log_string(
                log_fout,
                "starting probe "
                f"epoch={epoch_index_1based} scenes={args.val_scenes} frames={args.val_frames}",
            )
            probe = run_probe(
                epoch_index_1based=epoch_index_1based,
                checkpoint_path=checkpoint_to_eval,
                dataset_root=dataset_root,
                split_dir=args.val_split_dir,
                camera=args.camera,
                output_dir=probe_dir,
                eval_num_points=args.val_num_points,
                voxel_size=args.voxel_size,
                seed_feat_dim=args.seed_feat_dim,
                top_k=args.val_top_k,
                device=args.device,
                seed=args.seed,
                scenes=probe_scenes,
                frames=probe_frames,
            )
            probe_entry = {
                "epoch": probe.epoch,
                "checkpoint_path": probe.checkpoint_path,
                "metrics_percent": probe.metrics_percent,
                "elapsed_sec": probe.elapsed_sec,
                "frame_count": probe.frame_count,
                "json_path": probe.json_path,
            }
            probe_history.append(probe_entry)
            val_writer.add_scalar("AP", probe.metrics_percent["AP"], epoch_index_1based)
            val_writer.add_scalar("AP0.8", probe.metrics_percent["AP0.8"], epoch_index_1based)
            val_writer.add_scalar("AP0.4", probe.metrics_percent["AP0.4"], epoch_index_1based)
            log_string(
                log_fout,
                "[probe] epoch={epoch} AP={AP:.4f} AP0.8={AP08:.4f} AP0.4={AP04:.4f} sec={sec:.1f}".format(
                    epoch=epoch_index_1based,
                    AP=probe.metrics_percent["AP"],
                    AP08=probe.metrics_percent["AP0.8"],
                    AP04=probe.metrics_percent["AP0.4"],
                    sec=probe.elapsed_sec,
                ),
            )
            if best_probe is None or probe.metrics_percent["AP"] > best_probe.metrics_percent["AP"]:
                best_probe = probe
                shutil.copy2(checkpoint_to_eval, run_root / "best_AP_checkpoint.tar")
                save_json(
                    best_json,
                    {
                        "epoch": best_probe.epoch,
                        "checkpoint_path": best_probe.checkpoint_path,
                        "metrics_percent": best_probe.metrics_percent,
                        "elapsed_sec": best_probe.elapsed_sec,
                        "frame_count": best_probe.frame_count,
                        "json_path": best_probe.json_path,
                    },
                )
                log_string(log_fout, f"[probe] updated best checkpoint: {run_root / 'best_AP_checkpoint.tar'}")
            dump_run_state()

    if best_probe is not None:
        best_checkpoint_path = run_root / "best_AP_checkpoint.tar"
        log_string(log_fout, f"starting full test_seen eval with best checkpoint: {best_checkpoint_path}")
        full_eval_dir = run_root / "full_test_seen_eval"
        full_eval = run_full_test_seen_eval(
            checkpoint_path=best_checkpoint_path,
            dataset_root=dataset_root,
            camera=args.camera,
            output_dir=full_eval_dir,
            eval_num_points=args.val_num_points,
            voxel_size=args.voxel_size,
            seed_feat_dim=args.seed_feat_dim,
            device=args.device,
            seed=args.seed,
        )
        save_json(
            run_root / "final_report.json",
            {
                "status": "finished",
                "best_probe": {
                    "epoch": best_probe.epoch,
                    "checkpoint_path": best_probe.checkpoint_path,
                    "metrics_percent": best_probe.metrics_percent,
                    "elapsed_sec": best_probe.elapsed_sec,
                    "frame_count": best_probe.frame_count,
                    "json_path": best_probe.json_path,
                },
                "full_test_seen_eval": {
                    "checkpoint_path": full_eval.checkpoint_path,
                    "dump_dir": full_eval.dump_dir,
                    "cache_dir": full_eval.cache_dir,
                    "metrics_percent": full_eval.metrics_percent,
                    "ap_path": full_eval.ap_path,
                    "summary_path": full_eval.summary_path,
                },
            },
        )
        log_string(
            log_fout,
            "[full-eval] AP={AP:.4f} AP0.8={AP08:.4f} AP0.4={AP04:.4f}".format(
                AP=full_eval.metrics_percent["AP"],
                AP08=full_eval.metrics_percent["AP0.8"],
                AP04=full_eval.metrics_percent["AP0.4"],
            ),
        )

    train_writer.close()
    val_writer.close()
    log_fout.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
