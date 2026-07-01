# GSNet 微调复现

中文 | [English](README.md)

本仓库提供 **Graspness Discovery in Clutters for Fast and Accurate Grasp Detection** (ICCV 2021) 的 GSNet 微调、评估和可视化流程。

本项目基于上游 `graspness_unofficial` 实现，额外整理了：

- 数据预处理命令
- 从官方 Kinect checkpoint 出发的微调训练
- 训练期间周期验证
- 完整 `test_seen` AP 评估
- top1 / top5 / top50 抓取结果的 RGB 可视化

## 环境要求

已验证环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA toolkit | 12.8 |
| MinkowskiEngine | 0.5.4 |
| Open3D | 0.19.0 |
| TensorBoard | 2.20.0 |

需要的包和扩展：

- NumPy
- SciPy
- Pillow
- tqdm
- OpenCV
- Open3D
- TensorBoard
- MinkowskiEngine
- pointnet2 CUDA operators
- knn CUDA operator
- graspnetAPI
- grasp_nms

自定义 CUDA 扩展需要针对当前 GPU 架构重新编译。如果遇到：

```text
no kernel image is available for execution on the device
```

通常需要重新编译 `pointnet2` 或 `knn`，并设置正确的 `TORCH_CUDA_ARCH_LIST`。

## 安装

克隆本仓库：

```bash
git clone <this-repo>
cd gsnet-finetuning-reproduction
```

将上游 GSNet 代码放到：

```text
external/graspness_unofficial
```

期望目录结构：

```text
gsnet-finetuning-reproduction/
  Makefile
  README.md
  README_CN.md
  external/
    graspness_unofficial/
  gsnet_reproduction/
    assets/report/
    patches/
    scripts/
    tests/
```

依赖安装和 CUDA 扩展编译参考上游 GSNet 要求。现代 Python / PyTorch / CUDA 环境下可能需要使用：

```text
gsnet_reproduction/patches/
```

中的兼容补丁。

## 数据和权重

下载 GraspNet 数据集和官方 Kinect checkpoint。Makefile 默认路径为：

```text
data/graspnet
weights/kinect/minkuresunet_kinect.tar
```

也可以通过变量覆盖：

```bash
make train \
  DATASET_ROOT=/path/to/graspnet \
  OFFICIAL_CKPT=/path/to/minkuresunet_kinect.tar
```

## 输入和输出格式

### 数据集输入

流程默认使用 GraspNet 风格的数据目录：

```text
graspnet/
  scenes/
    scene_0100/
      kinect/
        rgb/
        depth/
        label/
        meta/
        annotations/
        camera_poses.npy
        cam0_wrt_table.npy
  grasp_label/
  collision_label/
  models/
  dex_models/
  graspness/                  # 预处理生成
```

单帧推理需要一帧 RGB-D 及其元数据：

- RGB 图：`rgb/XXXX.png`
- 深度图：`depth/XXXX.png`
- 分割标签：`label/XXXX.png`
- 相机内参：`meta/XXXX.mat`
- 相机 / 桌面位姿：`camera_poses.npy`、`cam0_wrt_table.npy`

脚本会把输入帧转换成模型输入：

```text
point_clouds: float32 [N, 3]，相机坐标系下的 XYZ 点，单位为米
coors:        float32 [N, 3]，point_clouds / voxel_size
feats:        float32 [N, 3]，全 1 特征
```

默认 `N=15000`。

### Checkpoint 输入

训练和评估使用 PyTorch `.tar` checkpoint，包含：

```text
model_state_dict
optimizer_state_dict
epoch
```

其中 `optimizer_state_dict` 只在续训时需要。

### 预测输出

推理输出 GraspNet 格式的抓取数组，保存为 `.npy`。每一行 17 个数：

```text
[score, width, height, depth, rotation(9), translation(3), object_id]
```

该结果可以直接用 `graspnetAPI.graspnet_eval.GraspGroup` 读取。

### 主要输出文件

训练输出：

```text
gsnet_reproduction/outputs/finetune_runs/<RUN_NAME>/
  train_logs/
    <RUN_NAME>_epochXX.tar
    log_train.txt
  tensorboard/
  val_probe/
  best_AP_checkpoint.tar
  best_probe.json
  final_report.json
```

完整评估输出：

```text
gsnet_reproduction/outputs/test_seen_epochXX_full/
  scene_0100/...               # 预测 dump
  scene_eval_test_seen_kinect/  # scene 级 AP 缓存
  ap_kinect.npy
  ap_kinect_summary.json
```

可视化输出：

```text
report_pngs/top1_rgb_overlay.png
report_pngs/top5_rgb_overlay.png
report_pngs/top50_rgb_overlay.png
summary.json
```

## Makefile 命令

查看所有命令：

```bash
make help
```

检查环境：

```bash
make check-env
```

检查数据集：

```bash
make check-dataset
```

## Point-level Graspness 生成

从 GraspNet 数据集中生成 point-level graspness 标签。

简化数据标签：

```bash
make preprocess-simplify
```

生成 graspness：

```bash
make preprocess-graspness
```

这两个命令会调用上游脚本：

```text
external/graspness_unofficial/dataset/
```

## 训练

从官方 Kinect checkpoint 开始微调：

```bash
make train
```

默认训练设置：

| 参数 | 数值 |
| --- | --- |
| camera | `kinect` |
| max epoch | `10` |
| batch size | `4` |
| learning rate | `1e-4` |
| num point | `15000` |
| voxel size | `0.005` |
| validation scenes | `100-102` |
| validation frames | `0-31` |

查看训练日志：

```bash
make tail-train
```

打开 TensorBoard：

```bash
make tensorboard
```

然后访问：

```text
http://127.0.0.1:6007
```

## 测试

评估官方 checkpoint：

```bash
make eval-official
```

评估某个微调 checkpoint：

```bash
make eval-epoch EPOCH=10
```

查看已保存的 AP 汇总：

```bash
make show-result EPOCH=10
```

列出所有 AP 汇总文件：

```bash
make list-results
```

完整评估拆成两步：

1. `test.py --infer` 生成预测 dump
2. `eval_dumps_by_scene.py` 按 scene 计算 AP

## 模型权重

官方 Kinect 权重默认路径：

```text
weights/kinect/minkuresunet_kinect.tar
```

微调 checkpoint 会保存到：

```text
gsnet_reproduction/outputs/finetune_runs/<RUN_NAME>/train_logs/
```

周期验证选出的最佳 checkpoint：

```text
gsnet_reproduction/outputs/finetune_runs/<RUN_NAME>/best_AP_checkpoint.tar
```

大型权重和输出目录默认不进入 Git。

## 结果

Kinect camera 上完整 `test_seen` 评估结果：

| Checkpoint | AP | AP0.8 | AP0.4 |
| --- | ---: | ---: | ---: |
| official | 35.5829 | 42.9651 | 26.7221 |
| epoch01 | 60.4761 | 72.8625 | 50.6743 |
| epoch02 | 60.9058 | 73.3467 | 51.1578 |
| epoch03 | 60.7956 | 73.1478 | 51.2388 |
| epoch05 | 61.9253 | 74.2925 | 52.5971 |
| epoch06 | 62.6110 | 74.9938 | 53.3483 |
| epoch07 | 61.8484 | 74.1433 | 52.6821 |
| epoch08 | 62.2937 | 74.5109 | 53.3766 |
| epoch09 | 62.2355 | 74.5615 | 53.1543 |
| epoch10 | **62.8227** | **74.9629** | **53.9890** |

最佳 checkpoint：

```text
epoch10
AP=62.8227
AP0.8=74.9629
AP0.4=53.9890
```

## 可视化

生成官方 checkpoint 的 RGB 抓取 overlay：

```bash
make vis-official
```

生成微调 checkpoint 的 RGB 抓取 overlay：

```bash
make vis-epoch EPOCH=10
```

默认输出：

```text
report_pngs/top1_rgb_overlay.png
report_pngs/top5_rgb_overlay.png
report_pngs/top50_rgb_overlay.png
```

夹爪颜色表示预测抓取得分：

- 蓝色：较低分
- 红色：较高分

## 展示图片

建议保留三类结果图：

- 训练曲线
- 不同 checkpoint 的 AP 对比图
- official 与微调后权重的 RGB 抓取可视化对比图

## 注意事项

- 不要提交数据集、checkpoint、预测 dump 或 TensorBoard 日志。
- 更换 GPU 后，优先重新编译自定义 CUDA 扩展。

## 致谢

本流程基于上游 GSNet / `graspness_unofficial` 实现和 GraspNetAPI。
