# adjustTM 模型训练交接手册

## 0. 交接目标

本手册用于将 `adjustTM` 可控亮度 Tone Mapping 实验交接给模型训练执行人员。

训练人员的任务不是继续修改模型结构，而是：

1. 验证真实 luminance-only baseline checkpoint 能被当前代码正确加载；
2. 使用同一数据、同一 baseline、同一划分和同一训练协议，完成四种控制方法训练；
3. 保存完整的配置、日志、最佳模型、最终模型和评测结果；
4. 汇总四种方法在亮度拟合、控制单调性、零点稳定性、色度漂移和效率上的差异。

四种待训练方法：

```text
param_residual
parallel_adapter
film
dual_lora
```

代码分支：

```text
feature/adjustTM-brightness-control
```

训练代码目录：

```text
adjustTM/
```

---

## 1. 实验边界：不要改变的内容

本轮实验的目标是比较不同的 **Gain/GTM 控制注入方式**，不是重新设计 ISP 模型。

以下设置必须保持不变：

- 所有方法使用同一个 baseline checkpoint；
- baseline 中 Gain、GTM 和 LTM 原始权重全部冻结；
- 仅训练 Gain/GTM 中新增的控制参数；
- LTM 保留在前向路径中，但不增加控制模块、不更新权重；
- 不使用 CbCr LUT、3D LUT、颜色网络或 learned GammaNet；
- 输出使用固定标准 sRGB OETF；
- 四种方法使用同一个 scene split；
- 四种方法使用相同 seed、epoch、image size、batch size、optimizer 和 loss 权重；
- batch size 必须是 18 的正整数倍；
- 主对比实验不启用 early stopping，四种方法统一训练 30 epochs；
- 模型选择使用各自 validation `log_luma_mae` 最优的 `control_best.pth`；
- 不使用 `--allow-parameter-mismatch`，除非明确进行非公平容量消融。

允许根据机器条件调整的参数只有：

- `--num-workers`；
- 是否启用 `--amp`；
- 数据、checkpoint 和输出目录；
- 在显存不足时统一降低四种方法的 `--image-size`。

任何调整必须同时应用于四种方法，并记录在交付说明中。

---

## 2. 数据要求

### 2.1 数据目录

输入和九级 GT 必须组织为：

```text
DATA_ROOT/
├── input_linear/
│   ├── scene_0001.png
│   ├── scene_0002.png
│   └── ...
└── gt_levels/
    ├── a_m100/
    │   ├── scene_0001.png
    │   └── ...
    ├── a_m075/
    ├── a_m050/
    ├── a_m025/
    ├── a_000/
    ├── a_p025/
    ├── a_p050/
    ├── a_p075/
    └── a_p100/
```

九级控制值为：

| 文件夹 | alpha |
|---|---:|
| `a_m100` | -1.00 |
| `a_m075` | -0.75 |
| `a_m050` | -0.50 |
| `a_m025` | -0.25 |
| `a_000` | 0.00 |
| `a_p025` | +0.25 |
| `a_p050` | +0.50 |
| `a_p075` | +0.75 |
| `a_p100` | +1.00 |

### 2.2 输入图像约束

`input_linear/` 中的图像必须满足：

- PNG；
- 3 通道 RGB；
- `uint16`；
- 线性 RGB；
- 已经完成白平衡和 CCM；
- 代码直接执行 `float32 / 65535`，不再读取 metadata，不再执行 AWB/CCM。

### 2.3 GT 约束

GT 图像可以是 `uint8` 或 `uint16` PNG，表示 sRGB 输出。

每个 scene 必须满足：

- 输入和九个 GT 文件名完全相同；
- 输入和九个 GT 空间尺寸完全相同；
- 不允许任一 level 缺文件；
- 不允许某个 GT level 出现额外文件。

当前完整数据预期为：

```text
844 scenes × 9 levels = 7596 GT images
```

---

## 3. 环境准备

### 3.1 获取代码

```bash
git clone https://github.com/baolinv0/modular_neural_isp.git
cd modular_neural_isp
git checkout feature/adjustTM-brightness-control
```

记录实际 commit：

```bash
git rev-parse HEAD
```

### 3.2 创建环境

推荐 Python 3.11：

```bash
conda create -n adjusttm python=3.11 -y
conda activate adjusttm
pip install -r requirements.txt
```

如果训练服务器已有与 CUDA 匹配的 PyTorch，请不要盲目覆盖。至少确认：

```bash
python - <<'PY'
import torch
import cv2
import numpy as np

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("opencv:", cv2.__version__)
print("numpy:", np.__version__)
PY
```

### 3.3 运行单元测试

```bash
PYTHONPATH=. python -m pytest adjustTM/tests -q
```

交接时该分支 GitHub Actions 的参考结果为：

```text
22 passed
```

本地测试未通过时，不要开始正式训练。

---

## 4. Baseline checkpoint 前置 Gate

真实 luminance-only baseline checkpoint 不在仓库内，因此正式训练前必须先验证 checkpoint 兼容性。

设置路径：

```bash
export BASELINE=/absolute/path/to/luminance_only_baseline.pth
export SAMPLE_INPUT=/absolute/path/to/input_linear/scene_0001.png
```

执行四方法联合 smoke test：

```bash
python -m adjustTM.smoke \
  --baseline-checkpoint "$BASELINE" \
  --input "$SAMPLE_INPUT" \
  --image-size 128 \
  --device cuda \
  --output adjustTM/smoke_results.json
```

该命令会对四种方法依次检查：

- baseline checkpoint 能否加载；
- Gain/GTM/LTM key 和 shape 是否兼容；
- `alpha=0` 是否严格回到 baseline；
- `alpha=-1` 和 `alpha=+1` 是否可前向；
- 控制参数是否获得梯度；
- optimizer step 后控制参数是否发生变化；
- baseline 参数是否保持不变；
- control checkpoint 是否可保存并重新加载。

### 4.1 必须满足的 Gate

开始正式训练前必须确认：

```text
所有四种方法 smoke test 均完成
alpha-zero drift = 0
baseline unchanged = true
control parameters updated = true
checkpoint round-trip = true
```

### 4.2 checkpoint 加载失败时

常见原因：

- checkpoint 不是当前 Gain/GTM/LTM 架构；
- checkpoint key 前缀不同；
- Gain/GTM/LTM 通道数不同；
- checkpoint 实际包含另一版 LTM；
- 保存内容不是 `state_dict`、`model_state_dict` 或 `model`。

不要使用 `strict=False` 绕过 Gain/GTM/LTM 的 missing key。应先输出：

```python
checkpoint = torch.load(PATH, map_location="cpu")
print(checkpoint.keys())
```

然后确认实际架构和 key 命名。只有已删除的 `_gamma_net`、`_lut_net`、`_3d_lut` 额外 key 可以被忽略。

---

## 5. 创建统一 scene split

四种方法必须使用同一个 scene-disjoint train/validation split。

设定目录：

```bash
export INPUT_DIR=/absolute/path/to/input_linear
export GT_ROOT=/absolute/path/to/gt_levels
export RUN_ROOT=/absolute/path/to/adjusttm_runs
export SPLIT_FILE=$RUN_ROOT/shared/split_seed_42.json

mkdir -p "$RUN_ROOT/shared"
```

预先生成 split，避免四个并行进程同时创建文件：

```bash
python - <<'PY'
import os
from adjustTM.dataset import discover_scene_names
from adjustTM.manifest import create_or_load_split_manifest

input_dir = os.environ["INPUT_DIR"]
split_file = os.environ["SPLIT_FILE"]

scenes = discover_scene_names(input_dir)
split = create_or_load_split_manifest(
    scenes,
    split_file,
    val_fraction=0.1,
    seed=42,
)

print("total:", len(scenes))
print("train:", len(split["train"]))
print("val:", len(split["val"]))
print("split:", split_file)
PY
```

对于 844 个 scene，0.1 validation 通常得到：

```text
train: 760 scenes
val:    84 scenes
```

训练样本数量：

```text
760 × C(9,2) = 27,360 same-scene pair samples / epoch
```

验证样本数量：

```text
84 × 9 = 756 scene-level samples
```

### 5.1 重要要求

四种方法最终 `config.json` 中必须具有：

- 相同的 `baseline_sha256`；
- 相同的 `split_manifest`；
- 相同的 train/val scene 列表；
- 相同的 seed；
- 相同的训练超参数。

---

## 6. 正式训练配置

推荐主实验配置：

```text
image_size              = 512
batch_size              = 18
val_batch_size          = 8
epochs                   = 30
learning_rate            = 1e-4
weight_decay             = 1e-6
seed                     = 42
lambda_grad              = 0.2
lambda_mono              = 0.1
lambda_zero              = 0.5
margin_per_alpha         = 0.01
gradient_clip            = 1.0
parameter_target         = 1040
parameter_tolerance      = ±10%
early_stopping           = disabled
```

### 6.1 batch size 约束

`LevelBalancedBatchSampler` 要求：

```text
batch_size ∈ {18, 36, 54, ...}
```

推荐使用 18。每个 18-sample batch 中，九个 level 各出现 4 次，因此 batch 内 level 边际严格均衡。

不要使用：

```text
batch_size = 8 / 9 / 16 / 24 / 32
```

---

## 7. 单卡顺序训练

下面命令依次训练四种方法：

```bash
METHODS=(
  param_residual
  parallel_adapter
  film
  dual_lora
)

for METHOD in "${METHODS[@]}"; do
  python -m adjustTM.train \
    --input-dir "$INPUT_DIR" \
    --gt-root "$GT_ROOT" \
    --baseline-checkpoint "$BASELINE" \
    --control-method "$METHOD" \
    --output-dir "$RUN_ROOT" \
    --split-manifest "$SPLIT_FILE" \
    --manifest-dir "$RUN_ROOT/manifests/$METHOD" \
    --image-size 512 \
    --batch-size 18 \
    --val-batch-size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --weight-decay 1e-6 \
    --num-workers 4 \
    --seed 42 \
    --lambda-grad 0.2 \
    --lambda-mono 0.1 \
    --lambda-zero 0.5 \
    --margin-per-alpha 0.01 \
    --grad-clip 1.0 \
    --save-every 5 \
    --amp \
    2>&1 | tee "$RUN_ROOT/${METHOD}_console.log"
done
```

没有 CUDA 时删除 `--amp`，并将 `--device cpu` 显式传入。

---

## 8. 四卡并行训练

代码当前不是 DDP。推荐一张 GPU 运行一种方法，四个独立进程并行。

```bash
mkdir -p "$RUN_ROOT/logs"

CUDA_VISIBLE_DEVICES=0 python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method param_residual \
  --output-dir "$RUN_ROOT" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/manifests/param_residual" \
  --image-size 512 --batch-size 18 --val-batch-size 8 \
  --epochs 30 --lr 1e-4 --weight-decay 1e-6 \
  --num-workers 4 --seed 42 --grad-clip 1.0 --save-every 5 --amp \
  > "$RUN_ROOT/logs/param_residual.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method parallel_adapter \
  --output-dir "$RUN_ROOT" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/manifests/parallel_adapter" \
  --image-size 512 --batch-size 18 --val-batch-size 8 \
  --epochs 30 --lr 1e-4 --weight-decay 1e-6 \
  --num-workers 4 --seed 42 --grad-clip 1.0 --save-every 5 --amp \
  > "$RUN_ROOT/logs/parallel_adapter.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method film \
  --output-dir "$RUN_ROOT" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/manifests/film" \
  --image-size 512 --batch-size 18 --val-batch-size 8 \
  --epochs 30 --lr 1e-4 --weight-decay 1e-6 \
  --num-workers 4 --seed 42 --grad-clip 1.0 --save-every 5 --amp \
  > "$RUN_ROOT/logs/film.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method dual_lora \
  --output-dir "$RUN_ROOT" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/manifests/dual_lora" \
  --image-size 512 --batch-size 18 --val-batch-size 8 \
  --epochs 30 --lr 1e-4 --weight-decay 1e-6 \
  --num-workers 4 --seed 42 --grad-clip 1.0 --save-every 5 --amp \
  > "$RUN_ROOT/logs/dual_lora.log" 2>&1 &

wait
```

并行训练时必须确认：

- 四个任务使用相同 `$SPLIT_FILE`；
- `--output-dir` 相同没有问题，各方法会写入独立子目录；
- `--manifest-dir` 使用方法独立目录，避免并发写同一个 sample-index 文件；
- 不要为不同方法修改 seed 或 loss 权重。

---

## 9. 快速训练链路检查

正式 30 epochs 前，建议每种方法先执行一个极小 dry run：

```bash
python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method param_residual \
  --output-dir "$RUN_ROOT/dry_run" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/dry_run/manifests/param_residual" \
  --image-size 128 \
  --batch-size 18 \
  --val-batch-size 4 \
  --epochs 1 \
  --max-train-batches 2 \
  --max-val-batches 1 \
  --num-workers 0 \
  --seed 42 \
  --device cuda \
  --amp
```

分别替换 `--control-method`，确保四种方法均能：

- 完成至少两个 optimizer steps；
- 写出 `config.json`；
- 写出 `train.jsonl`；
- 写出 `control_last.pth`；
- 写出 `control_best.pth`。

---

## 10. 训练中监控

每种方法目录：

```text
RUN_ROOT/
├── param_residual/
├── parallel_adapter/
├── film/
└── dual_lora/
```

每个目录应包含：

```text
config.json
train.jsonl
control_last.pth
control_best.pth
control_epoch_005.pth
control_epoch_010.pth
...
```

### 10.1 日志重点

`train.jsonl` 每个 epoch 应包含：

```text
train_total
train_log_luminance
train_gradient
train_monotonic
train_zero_anchor
val_log_luma_mae
val_luma_psnr
val_luma_ssim
val_chroma_rg_mae
selection_metric
best_metric
learning_rate
seconds
```

### 10.2 正常性检查

训练过程中检查：

- loss 为有限值；
- 没有 `Non-finite loss`；
- 没有 `Frozen baseline received parameter gradients`；
- `train_zero_anchor` 接近 0；
- validation 指标不是从首个 epoch 后持续恶化；
- 四种方法的每 epoch step 数一致；
- 参数量 gate 没有失败。

新增控制参数的默认数量约为：

| 方法 | 训练参数量 |
|---|---:|
| `param_residual` | 1064 |
| `parallel_adapter` | 1096 |
| `film` | 976 |
| `dual_lora` | 1024 |

---

## 11. 中断与恢复训练

恢复必须使用同一方法、同一 baseline、同一数据和同一 split。

例如：

```bash
python -m adjustTM.train \
  --input-dir "$INPUT_DIR" \
  --gt-root "$GT_ROOT" \
  --baseline-checkpoint "$BASELINE" \
  --control-method film \
  --output-dir "$RUN_ROOT" \
  --split-manifest "$SPLIT_FILE" \
  --manifest-dir "$RUN_ROOT/manifests/film" \
  --image-size 512 \
  --batch-size 18 \
  --val-batch-size 8 \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-6 \
  --num-workers 4 \
  --seed 42 \
  --grad-clip 1.0 \
  --save-every 5 \
  --amp \
  --resume "$RUN_ROOT/film/control_last.pth"
```

resume checkpoint 会恢复：

- control parameters；
- optimizer；
- scheduler；
- AMP scaler；
- epoch；
- best validation metric；
- Python、NumPy、Torch 和 CUDA RNG state。

不要使用另一个 control method 的 checkpoint 进行 resume。

---

## 12. 正式评测

优先评测：

```text
control_best.pth
```

评测命令：

```bash
python -m adjustTM.evaluate \
  --input-dir /absolute/path/to/test/input_linear \
  --gt-root /absolute/path/to/test/gt_levels \
  --baseline-checkpoint "$BASELINE" \
  --control-checkpoint "$RUN_ROOT/film/control_best.pth" \
  --control-method film \
  --image-size 512 \
  --batch-size 8 \
  --num-workers 4 \
  --dense-steps 41 \
  --latency-warmup 5 \
  --latency-runs 20 \
  --device cuda \
  --output "$RUN_ROOT/results/film.json"
```

对四种方法分别执行。

### 12.1 测试数据要求

`adjustTM.evaluate` 会评测传入目录中的全部 scene。

因此：

- 最佳方案是使用独立的 scene-disjoint test 目录；
- 不要把训练集和验证集混在同一个目录后直接作为最终测试；
- 如果暂时只有 844 个 scene，则主模型选择依据训练脚本内部 validation 指标，最终 test 结果应等待独立测试集。

### 12.2 输出指标

评测 JSON 包括：

- per-level log-luminance MAE；
- luminance PSNR；
- luminance SSIM；
- 对 GT 的 chroma RG MAE；
- 相对 alpha=0 的 chroma drift；
- clipping ratio；
- deep-shadow ratio；
- 9-level control-curve MAE；
- adjacent-level step error；
- endpoint range error；
- Spearman correlation；
- dense-alpha monotonic violation rate；
- dense-alpha scene pass rate；
- alpha=0 maximum baseline drift；
- Gain factor 范围；
- 三个 GTM 参数范围；
- 单张推理延迟。

---

## 13. 结果汇总模板

最终至少提交以下表格：

| Method | Trainable Params | Best Epoch | Val LogY MAE ↓ | Val Y-PSNR ↑ | Val Y-SSIM ↑ | Val Chroma MAE ↓ | Test Curve MAE ↓ | Monotonic Scene Pass ↑ | Alpha-0 Drift ↓ | Latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| param_residual | | | | | | | | | | |
| parallel_adapter | | | | | | | | | | |
| film | | | | | | | | | | |
| dual_lora | | | | | | | | | | |

同时报告：

- GPU 型号；
- CUDA 和 PyTorch 版本；
- Git commit；
- baseline checkpoint 路径与 SHA-256；
- 数据 scene 数；
- split manifest；
- 总训练时长；
- 是否使用 AMP；
- 是否降低 image size；
- 是否出现中断或 resume；
- 任何与本手册不同的配置。

---

## 14. 最终交付清单

训练人员必须提交以下内容，不要只提交 `.pth`：

### 14.1 代码与环境

- [ ] Git commit SHA；
- [ ] Python/PyTorch/CUDA/GPU 信息；
- [ ] 执行过的完整命令；
- [ ] 单元测试结果。

### 14.2 数据与公平性

- [ ] `split_seed_42.json`；
- [ ] 每种方法的 `train_sample_index.json`；
- [ ] 每种方法的 `val_sample_index.json`；
- [ ] 数据 scene 数和缺失文件检查结果；
- [ ] 四种方法 `baseline_sha256` 一致证明。

### 14.3 每种方法

- [ ] `config.json`；
- [ ] `train.jsonl`；
- [ ] console log；
- [ ] `control_best.pth`；
- [ ] `control_last.pth`；
- [ ] 必要的周期 checkpoint；
- [ ] 最终 evaluation JSON。

### 14.4 汇总

- [ ] 四方法指标对比表；
- [ ] 最优方法结论；
- [ ] 失败案例；
- [ ] 极端 alpha 输出样例；
- [ ] alpha=0、-1、+1 可视化；
- [ ] 是否观察到 clipping、shadow crush、色度漂移或非单调。

---

## 15. 常见故障

### 15.1 `batch_size must be a positive multiple of 18`

使用：

```text
18、36、54……
```

显存不足时优先使用 18，不要改成 8 或 16。

### 15.2 CUDA OOM

按顺序处理：

1. 开启 `--amp`；
2. 确认 batch size 已是最小合法值 18；
3. 将四种方法统一从 512 降到 384 或 256；
4. 减少 `--num-workers` 不会降低 GPU 显存，但可能缓解主机内存。

不要只为某一种方法降低分辨率。

### 15.3 baseline checkpoint missing/unexpected keys

先停止训练，不要绕过。检查实际 checkpoint 的架构、key 前缀和 shape。

### 15.4 `--amp requires a CUDA device`

CPU 训练时删除 `--amp`。

### 15.5 某方法参数预算失败

默认四种配置应位于 1040 ±10%。出现失败通常意味着代码结构或 constants 被改动。

不要直接使用：

```text
--allow-parameter-mismatch
```

除非当前任务明确是参数容量消融。

### 15.6 alpha=0 drift 不为 0

这是结构性错误，不是可接受的数值偏差。停止正式训练并检查：

- control gating；
- checkpoint 加载；
- baseline train/eval 状态；
- control checkpoint 是否来自同一种方法。

### 15.7 训练速度慢

LTM 虽然冻结，但仍然在前向路径中运行，并会对 Gain/GTM 输出被动响应。不要为了提速直接删除 LTM，否则实验定义发生改变。

---

## 16. 训练完成判定

只有满足以下条件，才能认为模型训练交付完成：

1. 四种方法使用同一个 baseline SHA-256；
2. 四种方法使用同一个 split manifest；
3. 四种方法完成相同 epoch 和训练协议；
4. 四种方法均生成 `control_best.pth` 和 `control_last.pth`；
5. 无 baseline 参数更新；
6. alpha=0 drift 为 0；
7. 无 NaN/Inf；
8. 完成统一测试集评测；
9. 提交完整配置、日志、manifest、checkpoint 和结果表；
10. 明确记录任何偏离本手册的操作。

在真实 checkpoint smoke test 未通过前，不得开始长时间正式训练。
