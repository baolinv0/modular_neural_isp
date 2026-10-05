# 可训练的局部 Tone Mapping 算法库

`tm_library` 在原有 Modular Neural ISP 的 photofinishing 管线中替换局部 TM 算子，提供五种可微分研究候选、显式配对数据加载、训练、续训、评估和推理。Gain、GTM、chroma 和 gamma 网络沿用原实现；`baseline` 保留原局部网络。原来的 `photofinishing`、`main` 和 GUI 使用方式见各自文档。

这些候选受公开算法范式启发，不是任何厂商算法或专利的精确复现。合成数据上的运行、梯度和机制测试仅证明实现能够执行；不证明真实相机画质更好，也不代表移动 GPU、NPU 或硬件 ISP 已完成集成。

## CPU 快速开始

以下命令均在仓库根目录执行。推荐 Python 3.10 或 3.11；安装 CUDA 版本 PyTorch 时按本机环境选择对应 wheel。

```bash
python -m venv .venv-tm
source .venv-tm/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-tm-library.txt

python -m tm_library.synthetic --output /tmp/tm-fixture \
  --num-train 8 --num-val 2 --size 64 --seed 123

python -m tm_library.train --config configs/tm_library/gain_residual.yaml \
  --train-manifest /tmp/tm-fixture/train.jsonl \
  --val-manifest /tmp/tm-fixture/val.jsonl \
  --output-dir /tmp/tm-run --epochs 2

python -m tm_library.evaluate --checkpoint /tmp/tm-run/best.pt \
  --manifest /tmp/tm-fixture/val.jsonl --output /tmp/tm-evaluation.json

python -m tm_library.infer --checkpoint /tmp/tm-run/best.pt \
  --manifest /tmp/tm-fixture/val.jsonl --output /tmp/tm-rendered --save-maps

python -m pytest tests/tm_library -q
```

生成器使用不同随机流产生 train/val 样本，保存 float32 输入、对齐 sRGB 目标、软语义图和置信度。目标来自已知标量亮度变换；这套小样本适合验证数据与命令闭环，不能作为自然图像质量基准。正式实验应自行准备真实、合法且对齐的数据，不在导入或测试期间下载数据集或外部预训练模型。

## 五种算子与共享管线

共享顺序为：归一化线性 sRGB → 原 Gain → 原 GTM → 候选局部算子 → 原 chroma → 原 gamma → 显示 sRGB。候选可以同时观察增益后的 `gain` 和 GTM 后的 `base`；`gain` 的值可能大于 1。各候选的输出是 chroma 之前的线性 RGB。

| 候选 / `algorithm` | 实际机制 | 主要参数与诊断图 |
| --- | --- | --- |
| 原基线 / `baseline` | 保留原始局部 TM 网络，使用原训练模式前向路径 | 原管线中间结果；只允许 S0 |
| A / `region_curves` | 正增量构造端点为 0/1 的单调曲线库；图像和可选语义特征预测逐像素归一化权重；每个专家在 RGB 三通道使用同一标量曲线，再加权融合 | `num_experts`、`curve_bins`；`curves`、`weights` |
| B / `spatial_grid` | 预测三维 bilateral grid 和学习到的 guidance；按空间坐标与 guidance 三线性切片得到 gate、三个 tone-curve 参数、local gain；混合原 GTM 与局部重映射 | `grid_size`、`grid_depth`；`grid`、`guidance`、`coefficients` |
| C / `gain_residual` | 预测低分辨率、有界 log2 EV 残差，边缘感知上采样后以同一个标量乘 RGB；零残差初始化 | `max_ev`、`filter_radius`、`filter_eps`；EV 与 gain 图 |
| D / `exposure_fusion` | 从同一张增益后图像合成固定 EV 曝光，使用 `x/(1+x)` 压缩；构建 RGB Laplacian 金字塔和权重 Gaussian 金字塔，逐层归一化融合并重建 | `exposures`、`pyramid_levels`；`exposures`、`weights`、各尺度权重 |
| E / `base_detail` | 在 log 亮度中用可配置 guided filter 分解 base/detail；分别预测 base 对比度/平移和独立 detail gain，再重建亮度和 RGB | `filter_radius`、`filter_eps`、`max_ev`；`base`、`detail`、`base_toned`、`detail_gain` |

A 的曲线查表域为 `[0,1]`，超出该范围的增益后高光在曲线白端点饱和。它没有预设“皮肤必须提亮”或“天空必须压暗”的专家身份。B 的 grid 存储的是参数 logits，切片后再约束参数；gate 可趋近原 GTM 或局部结果。C 不添加无约束 RGB 残差。D 确实执行多尺度融合；合成曝光没有增加传感器捕获的信息，也没有提供独立噪声样本，金字塔重建仍可能产生越界值。E 的 detail gain 实际改变输出；固定边缘感知分解不等同于学习到的语义分解。最终管线会裁剪输出，评估中的裁剪指标应结合中间线性结果理解。

五种算子均能用于全场景或人像。实际风格和区域行为由训练数据、语义开关、损失及控制参数决定；算法名称本身不保证人像肤色、逆光保护或全场景泛化，评估结论只能覆盖所用数据。

当前 PyTorch 计算图按上述依赖顺序执行，算子的条件特征依赖 `gain` / `base`。本库没有实现硬件 ISP global 分支与 NPU 语义分支的并行调度，也没有接入 DirectLink SDK 或硬件融合链路。S2 需要外部提供已对齐的软语义图；S1 辅助头用于训练约束，不是可直接部署的生产分割器。

## S0 / S1 / S2：语义的三个使用方式

| 模式 | `semantic_mode` | 渲染时使用语义图 | 训练中的语义监督 |
| --- | --- | --- | --- |
| S0 | `none` | 不使用；即使传入也忽略 | 无语义辅助头 |
| S1 | `train_only` | 不使用；推理不依赖分割图 | 在算子共享特征上训练辅助语义头，仅在训练时计算 logits |
| S2 | `explicit` | 使用外部软语义图，经置信度门控 | 可选区域损失；没有 S1 辅助头 |

默认语义通道依次是 person、skin、sky；它们是 `[0,1]` 概率，可重叠，不是 RGB 图或互斥类别 ID。默认 `semantic_channels: 3`，修改数量时必须同步修改标注含义。

没有标注的样本返回零 `semantics`、零 `confidence` 和零 `semantic_valid`，因此不会被当成所有类别均为负例。已标注背景可以是全零语义图，同时保持有效标记为 1。可用 `semantic_valid` 文件标记部分已标注像素/通道；辅助 BCE 只计算有效位置。S2 中零置信度屏蔽语义条件，图像特征驱动的增强仍然有效，因而不等价于关闭整个 TM。S0/S1 的推理输出不应随输入 mask 改变。

## 数据格式与颜色约定

manifest 为 JSONL（每行一条记录）或 JSON 数组；路径相对 manifest 所在目录解析。每个样本显式指定 `id`、`input`、`input_encoding`。监督训练与评估还必须提供 `target` 和 `target_aligned: true`。

下面是一条 JSONL 记录，展示一个对齐配对；文件需要由使用者提供：

```json
{"id":"scene001_frame001","input":"linear/scene001_frame001.npy","input_layout":"HWC","input_encoding":"linear_srgb","target":"targets/scene001_frame001.png","target_aligned":true,"semantics":"masks/scene001_frame001.npy","semantics_layout":"HWC","confidence":"confidence/scene001_frame001.npy","semantic_valid":"validity/scene001_frame001.npy","semantic_valid_layout":"HWC","scene":"backlit_portrait_001","camera":"camera_A","reference":"unaligned_reference/scene001.jpg"}
```

| 字段/编码 | 约定 |
| --- | --- |
| `linear_srgb` | 已在 sRGB 原色空间中的线性 RGB，归一化到 `[0,1]`；不能把有 gamma 的 JPEG 直接声明为线性 |
| `srgb` | 显示 sRGB `[0,1]`，加载器执行标准逆传递函数转为线性 sRGB |
| `camera_rgb` | 已去马赛克的相机 RGB `[0,1]`；需要 `metadata.cam_illum` 和 `metadata.ccm`，执行与原管线一致的 green-normalized WB + CCM 并裁剪到 `[0,1]` |
| `target` | 对齐的显示 sRGB `[0,1]`，加载器不会再对 target 逆 gamma |
| `semantics` | HWC 或 CHW 软概率，通道顺序由实验约定；空间尺寸必须匹配 input |
| `confidence` | HW 或单通道图，范围 `[0,1]`；有标注且省略时默认为 1 |
| `semantic_valid` | HW、单通道或 S 通道有效性图，范围 `[0,1]`；有标注且省略时默认为 1 |
| `scene` / `camera` | 用于分组报告，也用于使用者组织按场景/相机隔离的数据划分 |
| `reference` | 可记录未对齐风格参考；加载、损失和推理均不会将其当作像素 GT |

相机元数据可内联，也可放在单独 JSON 文件并由 `metadata` 指向：

```json
{"cam_illum":[2.0,1.0,1.5],"ccm":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}
```

该示例只说明字段形状，不能替代真实相机标定。`camera_rgb` 不接受 Bayer mosaic 或 DNG；黑电平、白电平、去马赛克等前端操作应在进入该库前正确完成。库的输入也不是任意无界 HDR radiance。

输入/目标支持 NPY、NPZ、PNG、JPEG、TIFF 等。PNG16 按 65535 归一化，PNG8 按 255 归一化，不先降成 8 位；浮点 NPY/NPZ/TIFF 保留浮点精度后转 float32。NPZ 的固定数据键为 `image`。NPY/NPZ 可为 HWC 或 CHW；当首尾维都可能是通道维时，必须写 `input_layout`、`target_layout` 或对应 `<field>_layout`，避免猜测。栅格图采用 HWC，RGB 不是 OpenCV 的 BGR。

加载器拒绝重复 ID、缺失文件、错误通道、未对齐 target 声明、尺寸不一致、非有限值和超出 `[0,1]` 的数据。resize、翻转和旋转同步应用于 input、target、语义图、置信度及有效性图。`target_aligned: true` 是使用者对几何对齐的声明，不会自动配准。未对齐参考图不能用于逐像素 L1/PSNR；需要另行设计风格训练方法。

矩形图随机旋转只取 0°/180°，避免同一 batch 中高宽互换；正方形图支持四个直角方向。`batch_size > 1` 时，输入须同尺寸或显式设置 `image_size`，形状检查不消耗增强随机状态。

只需推理时可以省略 target，例如：

```json
{"id":"capture001","input":"linear/capture001.npy","input_layout":"HWC","input_encoding":"linear_srgb","scene":"new_scene","camera":"camera_A"}
```

## 从目录构建 manifest

以下命令适用于已经去马赛克且已准备对齐目标的 S24 风格数据：

```bash
python -m tm_library.prepare_data \
  --input-dir data/camera_rgb --target-dir data/aligned_srgb \
  --metadata-dir data/metadata --input-encoding camera_rgb \
  --input-layout HWC --target-aligned \
  --scene portrait_session_001 --camera camera_A \
  --output data/train.jsonl
```

`--semantics-dir`、`--confidence-dir` 和对应 `--semantics-layout` 可选。构建器严格匹配大小写一致的文件 stem，包括元数据与标注，不会独立排序后按位置配对；重复或不匹配 stem 会报错。它不会把 reference 目录自动变为 target。train/val 应按采集场景、连拍组或主体隔离，避免同一图像/同一场景变体泄漏；跨相机实验还应固定相机划分。

## 配置与训练

`configs/tm_library/` 提供 `baseline.yaml` 与五个以算法名命名的候选配置，另有 `gain_residual_train_only.yaml`（S1）和 `gain_residual_explicit.yaml`（S2）示例。核心 YAML 结构如下（YAML 中的相对路径基于该配置文件所在目录；命令行路径覆盖项基于当前工作目录）：

```yaml
model:
  algorithm: gain_residual
  semantic_mode: none
  semantic_channels: 3
  width: 16
  analysis_size: 64
  max_ev: 1.0
  freeze_backbone: false
  use_3d_lut: false
data:
  train_manifest: /tmp/tm-fixture/train.jsonl
  val_manifest: /tmp/tm-fixture/val.jsonl
  image_size: 64
  augment: true
training:
  epochs: 2
  batch_size: 2
  learning_rate: 0.001
  weight_decay: 0.0001
  seed: 123
  device: cpu
  num_workers: 0
  gradient_clip: 1.0
  scheduler_step: 10
  scheduler_gamma: 0.5
  cpu_threads: 1
loss:
  rgb: 1.0
  log_luma: 0.1
  gradient: 0.1
  region: 0.0
  semantic: 0.1
output_dir: /tmp/tm-run
# 可选：从仓库中的原 photofinishing checkpoint 初始化共享网络
# upstream_checkpoint: ../../photofinishing/models/photofinishing_s24-style-0.pth
```

训练使用 AdamW、有限损失检查、梯度裁剪和 scheduler。RGB L1 比较对齐显示 sRGB，另有 log 亮度、梯度及可选区域平衡项；`semantic` 是 S1 的有效位置 BCE 权重。语义 mask 可以参与监督损失而不进入 S0/S1 渲染器。`freeze_backbone: true` 冻结原网络，仅训练候选/辅助头；候选模式下未使用的原 LTM 本来就冻结。建议先以小样本确认数据域和梯度，再开展真实数据训练。

`rgb`、`log_luma`、`gradient` 至少一个权重必须为正；不能只依赖区域或语义损失，否则未标注样本会失去图像监督，验证集可能出现无意义的零分。

训练保存最后一轮 `last.pt`、验证集最佳 `best.pt` 和 `history.json`（逐轮损失及划分检查结果）。最佳模型只由 val 决定，不应使用 test 选择 checkpoint。续训时 `epochs` 表示总训练轮数：

```bash
python -m tm_library.train --config configs/tm_library/gain_residual.yaml \
  --resume /tmp/tm-run/last.pt \
  --train-manifest /tmp/tm-fixture/train.jsonl \
  --val-manifest /tmp/tm-fixture/val.jsonl \
  --output-dir /tmp/tm-run --epochs 4
```

库 checkpoint 包含模型配置、模型权重以及训练状态；续训恢复 optimizer、scheduler 和随机状态。除总轮数和输出目录外，续训配置必须与 checkpoint 一致；当前版本要求 `num_workers: 0`，以保证轮次边界恢复不遗漏预取 worker 的随机状态。CPU 在相同输入与运行环境下提供轮次边界精确恢复验证；CUDA 的 `grid_sample` backward 可能不确定，训练器采用确定性算法的 `warn_only=True`，可能发出警告而继续执行，不声明所有候选都能在 CUDA 上精确续训。保留 manifest 和实际运行参数以便复现实验。不要加载来源不可信的 checkpoint。

## 原 checkpoint 与 Python API

原始 checkpoint 是原 `PhotofinishingModule` 的 bare state dict。通过 `load_upstream` 严格加载，不会静默跳过不匹配层；`use_3d_lut` 必须与 checkpoint 对应（文件名带 `-lut` 的模型使用 `true`）。候选仍需要训练自己的局部算子；原权重不代表候选已经训练好。

```python
import torch
from tm_library.config import ModelConfig
from tm_library.model import TMModel

torch.set_num_threads(1)
model = TMModel(ModelConfig(algorithm="gain_residual", semantic_mode="none"))
model.load_upstream("photofinishing/models/photofinishing_s24-style-0.pth")
model.eval()
with torch.no_grad():
    result = model(torch.rand(1, 3, 64, 64))
print(result["output"].shape, result["maps"].keys())

torch.save(model.checkpoint(), "/tmp/tm-model.pt")
restored = TMModel.from_checkpoint("/tmp/tm-model.pt").eval()
```

`output` 是显示 sRGB，`linear` 是候选处理后、chroma 前的线性 RGB，`maps` 是候选诊断图；S1 在训练时另有 `semantic_logits`。输入为 float BCHW RGB `[0,1]`。`.to(device)` 同时移动上游色彩矩阵。`baseline` + S0 在原网络支持的空间尺寸上与原训练模式路径保持一致；为了处理小于原 reflection padding 要求的输入，包装器只对极小图执行 replicate padding 再裁回。原文件和 checkpoint 没有因此改写。

## 评估、推理与证据边界

评估逐样本计算再汇总，并按 scene、camera 和有效语义区域报告。`--output` 可以是 JSON 文件，或目录（写入 `metrics.json`）。指标包括 PSNR、SSIM、亮度/区域误差、裁剪、延迟和参数量。PSNR 使用单位范围 sRGB，MSE 下限 `1e-12`，因此完美匹配记为 120 dB；SSIM 使用 RGB Gaussian 11×11、sigma 1.5、replicate 边界与总体协方差。亮度误差是 Rec.709 加权的 sRGB code values，不是物理线性亮度；裁剪比例统计最终输出中 `<=0` / `>=1` 的通道值，不能单独反推出裁剪前高光损失。参数量包含包装器中保存的原 LTM 等未执行参数，区别于仅运行算子的参数量。

不同数据域、分辨率、设备或计时范围的数字不可直接横比。延迟以 batch=1 测量模型前向，排除一次 warm-up，设备同步；解码、CPU→设备传输、指标和保存均不计入，也不是完整 RAW ISP 延迟。报告包含实际分辨率、设备和计时说明。实现细节见 `metrics.py` / `evaluate.py`。

推理从 checkpoint 恢复配置，不需要 GT，默认输出 8 位 sRGB PNG 和 `metadata.json`；输入精度保留不代表 PNG 导出也是高精度。加入 `--save-maps` 保存 float NPZ 诊断图，保留 batch 维，未做显示归一化。文件名使用经过清理的 ID 加哈希，原 ID 与文件映射记录在 metadata 中。S2 若要使用语义，manifest 应包含可信且同几何的外部软 mask；S0/S1 无需分割网络。该库没有内置在线分割模型或下载逻辑。

CI 在 CPU 上运行 `tests/tm_library`，覆盖定义性机制、梯度、奇数/极小/黑白输入、数据精度与颜色、语义门控、原基线一致性、优化和 checkpoint/CLI 闭环。实际审查与运行证据由 [review 记录](../docs/tm_library_review.md) 汇总，不能用合成验证替代真实相机 IQ 测试。

尚未证明的范围包括：真实数据上优于原 LTM、跨相机泛化、肤色偏好与主观评价、视频时序稳定、真实多曝光去鬼影、低光去噪收益、移动端功耗/内存/吞吐，以及硬件 ISP 的量化、颜色标定和在线集成。需要分别开展固定划分消融（baseline/A–E、S0/S1/S2）、真实高分辨率测量和部署验证。

## 公开范式来源

这些链接说明相关研究背景；实现中的具体参数化、训练目标和语义控制以本库代码为准，不把论文或上游结果转述为本库实验结果。

- [Modular Neural Image Signal Processing](https://arxiv.org/abs/2512.08564)：本仓库共享 Gain/GTM/chroma/gamma 管线与原局部基线。
- [Deep Bilateral Learning for Real-Time Image Enhancement / HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/)：B 的低分辨率 bilateral grid 与 slicing 范式；本库切片的是五个局部 tone controls，并非原论文的 affine RGB 变换复现。
- [Exposure Fusion — Mertens、Kautz、Van Reeth](https://jankautz.com/publications/exposure_fusion.pdf)：D 的多分辨率融合范式；本库使用单图合成曝光和学习权重，区别于真实 bracketed captures 的融合。
- [Fast Bilateral Filtering for the Display of High-Dynamic-Range Images — Durand、Dorsey](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf)：E 的边缘感知 base/detail 思路；本库采用 log 亮度 guided filter 和分别学习的控制。

使用原仓库代码和权重时仍须遵守 [原许可证](../LICENSE.md)，并按原 README 引用上游论文。
