# 补丁说明

这个目录保存对第三方源码的本地兼容补丁。第三方源码目录 `external/` 被 `.gitignore` 忽略，因此不能只改 `external/graspness_unofficial/` 或临时构建目录，否则推到 GitHub 后别人无法复现。

## graspness_unofficial_torch2_cuda.patch

用途：让 `external/graspness_unofficial/knn` 能在当前环境中按 CUDA 扩展编译。

解决的问题：

- `knn/setup.py` 原来只在 `torch.cuda.is_available()` 为 true 时编译 CUDA；如果宿主驱动临时不可见，会误编成 CPU 扩展。补丁加入 `FORCE_CUDA=1`。
- `knn` 原来依赖旧 PyTorch 的 `THC/THC.h`、`THCudaMalloc`、`THCudaFree`；PyTorch 2.x 中这些接口已移除。补丁改为 `ATen/cuda/CUDAContext.h` 和 `c10::cuda::CUDACachingAllocator`。
- `Tensor.data<T>()` 改为 `Tensor.data_ptr<T>()`。

应用方式：

```bash
cd external/graspness_unofficial
git apply ../../gsnet_reproduction/patches/graspness_unofficial_torch2_cuda.patch
```

## minkowski_engine_0.5.4_cuda128_py312.patch

用途：让 `MinkowskiEngine==0.5.4` 在当前环境中编译并 import。

当前环境：

```text
Python 3.12.13
PyTorch 2.11.0+cu128
CUDA toolkit 12.8.93
GCC/G++ 12.4.0
TORCH_CUDA_ARCH_LIST=12.0
```

解决的问题：

- CUDA 12.8 的 NVTX 头文件冲突。
- CUDA 12.8 / Thrust 需要显式包含 `sort`、`reduce`、`execution_policy` 等头文件。
- NVCC 12.8 下 `std::shared_ptr = std::unique_ptr` 触发 `std::__to_address` 二义性。
- Python 3.12 移除了 `collections.Sequence`。

应用方式示例：

```bash
cd /tmp/gsnet_me_build
# 解压 MinkowskiEngine-0.5.4.tar.gz 后：
patch -p2 -d MinkowskiEngine-0.5.4 < /path/to/repo/gsnet_reproduction/patches/minkowski_engine_0.5.4_cuda128_py312.patch
```

注意：这个补丁是基于 PyPI 源码包 `MinkowskiEngine-0.5.4.tar.gz` 导出的。

## graspness_unofficial_eval_split.patch

用途：让上游 `test.py` 在保持默认 `test_seen` 行为不变的前提下，增加
`--split {test_seen,test_similar,test_novel}`，用于复现 README 表格里的 Seen、
Similar 和 Novel 三个测试集结果。

解决的问题：

- 上游 `test.py` 推理阶段硬编码 `GraspNetDataset(..., split='test_seen')`。
- 上游 `test.py` 评估阶段硬编码 `GraspNetEval(..., split='test_seen')` 和
  `eval_seen()`。
- README 报告了三个 split 的 AP，但原脚本只能直接跑 Seen。

应用方式：

```bash
cd /path/to/repo
git apply gsnet_reproduction/patches/graspness_unofficial_eval_split.patch
```

默认不传 `--split` 时仍然使用 `test_seen`，因此原 README 的 Seen 命令兼容不变。
