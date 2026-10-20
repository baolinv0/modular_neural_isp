# 数据与实验复现手册

2026-10-05。配套 [数据合成设计](DATA_SYNTHESIS_DESIGN.md) 和 [实现字段合同](DATA_PIPELINE.md)。本手册面向取得仓库和已发布产物的工程师：先恢复已跑结果，再验证新数据链，最后接入自己的线性 HDR 素材。

## 0. 先选对路径

| 路径 | 输入与结果 | 是否重训 | 证据边界 |
|---|---|---|---|
| A：恢复旧实验包 | 40 个解析场景、24 个 selected checkpoint、日志/图片 | 不需要 | 旧 **RGB 模拟**实验，不含原生 Bayer 候选包 |
| B：从零生成 Bayer demo | 12 场景、16×16、1 噪声种子、8 组各 1 epoch | 是，小规模检查 | 验证数据→训练→RAW 推理接口，不是质量 benchmark |
| C：自己的线性 HDR | 显式 float 素材→v1 scene→v2 Bayer 候选包 | 按需要 | 格式可接入不代表参考、时间采样或设备已标定 |

不要将 A 的 24 个模型描述为 Bayer 训练权重。B/C 保存的 checkpoint 才绑定 `acquisition` profile；旧 RGB checkpoint 加载后 `algorithm.acquisition is None` 是正确行为。

以下 Bash 命令在仓库根目录运行。生成、解压、训练都使用**新的输出目录**。生成器拒绝已有目录；训练器仅对已存在的 `results.json` 拒绝继续，未完成的目录可能被覆盖，所以不要把复现当成断点续训。

## 1. 获取分支与环境

```bash
git clone --branch feature/capture-tm-c-20261005 --single-branch \
  https://github.com/baolinv0/modular_neural_isp.git
cd modular_neural_isp
git rev-parse HEAD

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r capture_tm/requirements.txt numpy==1.26.4 pytest
```

记录 `git rev-parse HEAD`。本手册的算法审查基线为 `6090197`，物理管线来自 `f726419`；文档所在提交还校正了旧产物的分片清单。不要回退到有错误清单的旧提交后直接运行恢复命令。

已验证运行环境：Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4。`capture_tm/requirements.txt` 是宽松依赖范围，单独安装它不能保证这些版本。CPU 命令不需要 CUDA；当前 CLI 训练没有 `--device cuda` 开关，不能仅安装 CUDA wheel 就称完成 GPU 训练。

需要 EXR/TIFF 时才安装可选解码器：

```bash
python -m pip install -r capture_tm/requirements-hdr.txt OpenEXR==3.3.3
```

原始权重必须存在，即使 selected checkpoint 已含完整模型状态，loader 也会先从原权重构造网络：

```bash
test -f photofinishing/models/photofinishing_s24-style-0.pth
python -c "import sys,torch,numpy; print(sys.version); print(torch.__version__, numpy.__version__)"
```

原作者模型与代码依其仓库许可使用。本手册不把合成产物当成来自真实 Apple/Samsung 的数据。

## 2. 路径 A：恢复 40 场景、24 个模型及日志

### 2.1 重组与完整性检查

```bash
python artifacts/reconstruct_archive.py
python -m zipfile -t artifacts/ae_tm_single_hdr_experiments_20261005.zip
test ! -e runs/capture_tm/recovered-rgb-study && \
  python -m zipfile -e artifacts/ae_tm_single_hdr_experiments_20261005.zip \
  runs/capture_tm/recovered-rgb-study
```

预期：重组打印 `41759233 bytes`，校验通过；zip 检查打印 `Done testing`。SHA-256 为 `9206a7a89f669493debce2712aa2f4004688b322227a1c1d01c7a7146ae32051`。重组脚本会覆盖同名重组 ZIP；不要把自己的不同文件放在该路径。原分片不会被删除。

本次修正仅给 manifest 的 70 个 `chunks[*].path` 补上漏写的 `.zip`，使之对应实际 `*.zip.part-000.bin` 等文件。分片内容、总大小和原 ZIP 哈希没有改变。

解压目录直接包含：

| 相对路径 | 内容 |
|---|---|
| `data/manifest.json` 与数组 | 40 场景，24 train / 8 val / 8 test，32×32，数据种子 2026 |
| `apple/seed_0/A11/selected.pt` 等 | Apple 三种子×四组，共 12 个所选模型 |
| `samsung/seed_0/S11/selected.pt` 等 | Samsung 三种子×四组，共 12 个所选模型 |
| 每组 `training.json`、`evaluation.json`、PNG | 训练、逐场景测试和目标/成片对照 |
| `apple/results.json`、`samsung/results.json` | 分组均值、场景结果及配对统计 |
| `runtime.json`、`*_deployment_parity.json` | 实际环境和推理一致性记录 |

不含候选缓存和 `last.pt`；缓存可重建，`selected.pt` 是验证集选定的模型。源 manifest 路径可迁移，部分历史日志保留原运行绝对路径，不应将其当成现在的加载路径。

### 2.2 不重训，直接载入旧模型推理

```bash
python - <<'PY'
from pathlib import Path
import torch
from capture_tm.pipeline_sources import iter_source_scenes
from capture_tm.joint_algorithm import JointCaptureAlgorithm

torch.set_num_threads(1)
root = Path('runs/capture_tm/recovered-rgb-study')
scene = next(s for _, s in iter_source_scenes(root / 'data/manifest.json')
             if s.split == 'test')
for scheme, group in [('apple', 'A11'), ('samsung', 'S11')]:
    model = JointCaptureAlgorithm.from_checkpoint(
        root / scheme / 'seed_0' / group / 'selected.pt')
    assert model.acquisition is None  # 这些历史模型是 RGB 模拟域
    result = model.run_simulated(scene, seed=7, noisy=True)
    assert torch.isfinite(result['output']).all()
    print(group, scene.scene_id, result['request']['selected_index'],
          len(result['captures']), tuple(result['output'].shape))
PY
```

本次核验的预期：`joint_0032`；A11 候选 11、1 次采集；S11 候选 9、3 次采集；输出均 `[3,32,32]`。这些结果包含旧研究的失败行为，不是推荐的相机曝光策略。

### 2.3 可选：重跑旧 RGB 实验

```bash
test ! -e runs/capture_tm/repeated-rgb-study && \
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/recovered-rgb-study/data/manifest.json \
  --output runs/capture_tm/repeated-rgb-study --scheme both \
  --epochs 10 --warmup 10 --seeds 0,1,2 --noise-seeds 0,1 \
  --threads 1 --prepare-threads 8 --candidate-chunk-size 4
```

也可用 `python -m capture_tm.joint_data --output 新目录 --scenes 40 --size 32 --seed 2026` 重建解析场景。已有结果见 [JOINT_RESULTS.md](JOINT_RESULTS.md)。本轮文档核验没有重新执行这项完整 24 组训练；旧结果不可和下一节的 Bayer 小实验混为同一输入条件。

## 3. 路径 B：生成 Bayer 数据并跑通八组

### 3.1 生成与独立 RAW 重算

```bash
python -m capture_tm.pipeline_cli demo \
  --output runs/capture_tm/raw-demo-repro --scenes 12 --size 16 \
  --scheme both --noise-seeds 0 --seed 2026 --threads 1

python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/raw-demo-repro/acquisition/manifest.json \
  --output runs/capture_tm/raw-demo-repro/validation.json --threads 1
```

预期输出：4 train / 4 val / 4 test；Apple 12 个候选、Samsung 24 个计划；`valid=true`、`raw_composition_verified=true`。该验证从归档 DN 重建去马赛克、固定融合与前端，不重新采集，不证明设备真实性，也不验证源图是否真的干净。

`sources/manifest.json` 是 v1 线性源；`acquisition/manifest.json` 是 v2 采集包。**下一步训练必须传后者**，否则会走旧 RGB 采集路径。

### 3.2 八组小规模训练

```bash
test ! -e runs/capture_tm/raw-study-repro && \
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/raw-demo-repro/acquisition/manifest.json \
  --output runs/capture_tm/raw-study-repro --scheme both \
  --epochs 1 --warmup 1 --seeds 0 --noise-seeds 0 \
  --threads 1 --prepare-threads 1 --candidate-chunk-size 4
```

每个活动模块正式更新 4 次，另有共同 AE 预热。输出 root `results.json`、两方案 `results.json`、各组 `selected.pt/last.pt`、训练/评测 JSON 和图片。四组为规则/学习 AE × 冻结/学习 TM，HDR F0 始终固定。此配置只用于验证执行。

旧 Bayer 核验记录见 [DATA_PIPELINE_VALIDATION.md](DATA_PIPELINE_VALIDATION.md)，本手册命令的独立再执行见 [REPRODUCE_DATA_VALIDATION.md](REPRODUCE_DATA_VALIDATION.md)。不是用一个 epoch 的数字证明收敛或画质提升。

### 3.3 用本次训练权重执行实际 Bayer 模拟采集

```bash
python - <<'PY'
from pathlib import Path
import torch
from capture_tm.pipeline_sources import iter_source_scenes
from capture_tm.joint_algorithm import JointCaptureAlgorithm

torch.set_num_threads(1)
demo = Path('runs/capture_tm/raw-demo-repro')
study = Path('runs/capture_tm/raw-study-repro')
scene = next(s for _, s in iter_source_scenes(demo / 'sources/manifest.json')
             if s.split == 'test')
for scheme, group in [('apple', 'A11'), ('samsung', 'S11')]:
    model = JointCaptureAlgorithm.from_checkpoint(
        study / scheme / 'seed_0' / group / 'selected.pt')
    assert model.acquisition is not None
    result = model.run_simulated(scene, seed=37, noisy=True)
    assert torch.isfinite(result['output']).all()
    print(group, result['request']['selected_index'],
          [tuple(c.raw_dn.shape) for c in result['captures']],
          tuple(result['output'].shape))
PY
```

预期一帧或三帧 `[1,16,16]` 原生 DN，输出 `[3,16,16]`。用于真实驱动时，`select(previews,state)` 接收 `[3,3,H,W]` 的已观测预览和 `[3,3]` 生效状态；按 request 采集后以 `finish(actual_captures)` 使用实际读回参数。不能把请求曝光当成已经生效的曝光，也不能把 RGB checkpoint 静默当 Bayer checkpoint。

### 3.4 查看包内字段

```bash
python - <<'PY'
from pathlib import Path
import json, torch
p = Path('runs/capture_tm/raw-demo-repro/acquisition/manifest.json')
m = json.loads(p.read_text())
for name, entry in m['schemes'].items():
    row = entry['records'][0]
    d = torch.load(p.parent / row['path'], map_location='cpu', weights_only=True)
    print(name, d['scene_id'], tuple(d['raw_dn'].shape),
          tuple(d['images'].shape), tuple(d['target'].shape))
PY
```

Apple RAW `[1,12,1,1,16,16]`；Samsung RAW `[1,24,3,1,16,16]`。这些维度依次是噪声重复、计划、帧、通道、高、宽。metadata 保存实际曝光和时序；`images` 已经是固定前端/融合输入，不要在 TM 外额外做第二次曝光补偿。

### 3.5 扩大数据时的配置示例（不是本轮已执行研究）

```bash
python -m capture_tm.pipeline_cli demo \
  --output runs/capture_tm/raw-demo-40 --scenes 40 --size 32 \
  --scheme both --noise-seeds 0,1 --seed 2026 --threads 1
python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/raw-demo-40/acquisition/manifest.json --threads 1
test ! -e runs/capture_tm/raw-study-40 && \
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/raw-demo-40/acquisition/manifest.json \
  --output runs/capture_tm/raw-study-40 --scheme both \
  --epochs 10 --warmup 10 --seeds 0,1,2 --noise-seeds 0,1 \
  --threads 1 --prepare-threads 8 --candidate-chunk-size 4
```

扩大解析 demo 仍不等于真实内容数据集。源 H/W 必须能被 `spatial_downsample` 整除，native H/W 必须为偶数、经面积缩小后至少满足 TM 的 12 像素下限；demo 生成器要求 `size>=16`、场景数至少 12 且为 4 的倍数。候选数量目前固定为 12/24，没有可直接使用的 `--num-candidates` 参数。

先以小分辨率/单场景测存储和内存，再放大。当前每个场景会保存所有候选及 RAW/前端中间张量，空间随 R×K×F×H×W 增长。流式读取避免把整个数据集装入内存，但不能消除单场景候选包的峰值；`candidate-chunk-size` 主要限制 TM 准备/训练渲染，不会缩小已导出的 RAW 包。

## 4. 路径 C：接入自己的线性 HDR 素材

模板为 [source_recipe.example.json](../../configs/capture_tm/source_recipe.example.json)。其中 `.npy` 是待替换路径，且仅一个 train 示例，**不能原样执行完整训练**。需要自己提供素材，并加入非空 train/val/test，保持原 episode 及依赖不跨 split。

### 4.1 最小输入合同

| 项 | 必填/约束 |
|---|---|
| `sensor` | 整包共享；没有实测时保留 `assumed_engineering`，不可只改状态冒充标定 |
| `scene_id` / `source_id` / `split` | scene 唯一；原 episode/crop/增强共享 source；split 为 train/val/test |
| `input_path` 或 `frame_paths` | 二选一；所有相对路径相对 recipe 所在目录，不是终端 cwd |
| `input_layout` | 明确 CHW/HWC/TCHW/THWC；frame_paths 显式提供 T 轴 |
| `input_domain` | `sensor_linear_relative_radiance` 或 `linear_xyz`；后者另需 `xyz_to_sensor` |
| `frame_times_s` | 显式有限且递增；单帧也写 `[0.0]`；动态时覆盖预览/全部行曝光及参考时刻 |
| `radiance_scale` | 整个 episode 一次正倍率；非逐帧、非逐候选归一化 |
| `clean_reference_verified` | 依据外部证据声明；布尔值本身不执行参考质量验证 |
| `subject_mask_path` | 可选 float NPY/NPZ，[1,H,W]、[0,1]；主体指标需非空有效 mask |
| `dependency_ids` / `provenance` | 原始依赖、源曝光/光学/裁切/噪声等限制，供分组与证据审查 |

NPY/NPZ 必须是 float，不能把 uint16 自动除白点后假定为 HDR 真值。NPZ 帧键为 `frames`；EXR 须单 part、非 deep、恰好完整分辨率 R/G/B 三通道，带额外 alpha/depth 的文件需要先明确导出。XYZ 不在此入口再次 inverse OETF；RawGen 的编码 XYZ 走旧专用 importer，不能作为已经 linear 的 XYZ 直接输入。

模板的稀疏 timestamps 只是语法示例，不是高速运动合格采样率。源视频自身长曝光模糊、插帧鬼影、去噪和裁切不会被模拟器修复。普通线性 sRGB 也不是 camera RGB：必须给出明确颜色映射，不能仅改 `input_domain` 标签。

### 4.2 导入、采集、验证、训练

创建包含真实本地路径的 `my_sources.json` 后运行：

```bash
python -m capture_tm.pipeline_cli import \
  --recipe my_sources.json --output runs/capture_tm/imported-scenes-repro
python -m capture_tm.pipeline_cli generate \
  --manifest runs/capture_tm/imported-scenes-repro/manifest.json \
  --acquisition-profile configs/capture_tm/acquisition_bayer.json \
  --output runs/capture_tm/imported-acquisition-repro \
  --scheme both --noise-seeds 0,1 --seed 2026 --render-ev 0 --threads 1
python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/imported-acquisition-repro/manifest.json \
  --output runs/capture_tm/imported-acquisition-repro/validation.json --threads 1
test ! -e runs/capture_tm/imported-study-repro && \
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/imported-acquisition-repro/manifest.json \
  --output runs/capture_tm/imported-study-repro --scheme both \
  --epochs 10 --warmup 10 --seeds 0,1,2 --noise-seeds 0,1 --render-ev 0 \
  --threads 1 --prepare-threads 8 --candidate-chunk-size 4
```

这些命令是用户素材的执行模板；本次未取得该素材，不能称已验证其质量或成功跑过训练。导入成功 fixture 与缺失格式边界见 [DATA_PIPELINE_VALIDATION.md](DATA_PIPELINE_VALIDATION.md)。

改变传感器、PSF、rolling、readout、候选、噪声重复或目标 EV 后，重新生成新数据包。训练 `--noise-seeds` 列表及 `--render-ev` 必须与包完全相同。当前 profile 的 rolling 默认 0；想验证非零 rolling 要显式修改配置并重新生成，不能把默认 demo 当作 rolling 实验。

`--no-noise` 只用于诊断最终采集，预览仍有噪声，ADC 仍量化；它不是整个管线的无噪声/无限位深模式。

## 5. 验证命令与故障排查

```bash
python -m pytest -q tests/test_capture_tm_acquisition.py \
  tests/test_capture_tm_pipeline.py tests/test_capture_tm_pipeline_sources.py \
  tests/test_capture_tm_learned_tone.py
```

本轮专项验证为 111 passed；全仓可用 `python -m pytest -q`，此前完整运行记录为 314 passed，不能用本轮专项重跑冒充完整回归。

| 现象 | 先检查 |
|---|---|
| 恢复脚本找不到分片 | 是否使用修正后的清单，实际文件名含 `.zip.part-`；是否完整 clone 了 70 个二进制分片 |
| ZIP 校验失败 | 停止解压；对照清单大小，重新获取受损分片，不跳过哈希校验 |
| `output already exists` | 选新路径，不删除旧实验来复用名称 |
| 动态时间 support 不足 | 检查历史预览与所有 rolling 行曝光，不补假未来边界帧 |
| `no exposure duration remains` / overlap | rolling＋readout 占满采集槽；修正设备时序，不能忽略约束 |
| 编码/布局/XYZ 失败 | 检查 float、明确 layout 与颜色矩阵；不把解码失败变为 uint8 归一化 |
| 训练提示 split 缺失 | 模板只有 train；增加独立源的 val/test，不复制同场景伪造 split |
| noise seeds / render EV 不匹配 | 使用归档参数，或重新生成；不能训练时悄悄换采集数据 |
| 权重找不到 | 在仓库根目录运行；保留原 style-0 权重，必要时明确传入 `--weights`/加载器 `weights=` |
| RAW 推理没有 `raw_dn` | 检查 checkpoint 是否来自 v2 包；路径 A 的 RGB 权重没有原生 Bayer 输出 |
| 主体指标看似正常但无主体 | absent/全零 mask 会退化为全图项；检查 mask 与固定参考坐标 |

所有“通过”都应带上具体证据范围：文件可恢复、schema/RAW 重算一致、训练可执行、物理模型可解释、源素材可信和真实设备收益，是不同层级，不能相互替代。
