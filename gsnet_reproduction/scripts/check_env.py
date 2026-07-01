#!/usr/bin/env python3
"""Check the active Python/CUDA environment for GSNet reproduction."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys


MODULES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("PIL", "Pillow"),
    ("tqdm", "tqdm"),
    ("tensorboard", "tensorboard"),
    ("open3d", "open3d"),
    ("torch", "torch"),
    ("MinkowskiEngine", "MinkowskiEngine"),
    ("graspnetAPI", "graspnetAPI"),
    ("pointnet2._ext", "pointnet2._ext"),
    ("knn_pytorch.knn_pytorch", "knn_pytorch.knn_pytorch"),
]


def module_version(module: object) -> str:
    return str(getattr(module, "__version__", "version unknown"))


def check_module(import_name: str, display_name: str) -> bool:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        print(f"[MISS] {display_name}: {exc.__class__.__name__}: {exc}")
        return False
    print(f"[OK] {display_name}: {module_version(module)}")
    return True


def run_command(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
        )
    except FileNotFoundError:
        return 127, "command not found"
    except subprocess.TimeoutExpired:
        return 124, "command timed out"
    return result.returncode, result.stdout.strip()


def check_torch_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False

    ok = bool(torch.cuda.is_available())
    print(f"[INFO] torch version: {torch.__version__}")
    print(f"[INFO] torch cuda build: {getattr(torch.version, 'cuda', None)}")
    print(f"[INFO] torch.cuda.is_available(): {ok}")
    if ok:
        print(f"[INFO] cuda device count: {torch.cuda.device_count()}")
        print(f"[INFO] cuda device 0: {torch.cuda.get_device_name(0)}")
    return ok


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"CONDA_DEFAULT_ENV: {os.environ.get('CONDA_DEFAULT_ENV', '<unset>')}")

    print("\nNVIDIA driver")
    nvidia_code, output = run_command(["nvidia-smi"])
    if nvidia_code == 0:
        first_line = output.splitlines()[0] if output else "nvidia-smi returned no output"
        print(f"[OK] nvidia-smi: {first_line}")
    else:
        print(f"[MISS] nvidia-smi failed: {output}")

    nvcc_code, output = run_command(["nvcc", "--version"])
    if nvcc_code == 0:
        last_line = output.splitlines()[-1] if output else "nvcc returned no output"
        print(f"[OK] nvcc: {last_line}")
    else:
        print(f"[WARN] nvcc unavailable: {output}")

    print("\nPython packages")
    package_ok = True
    for import_name, display_name in MODULES:
        package_ok = check_module(import_name, display_name) and package_ok

    print("\nPyTorch CUDA")
    cuda_ok = check_torch_cuda()

    if package_ok and cuda_ok and nvidia_code == 0:
        print("[PASS] environment looks ready for GPU GSNet work")
        return 0

    print("[FAIL] environment is not fully ready for GSNet training")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
