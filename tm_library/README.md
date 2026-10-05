# 可训练的局部 Tone Mapping 算法库 v2

`tm_library` 在原有 Modular Neural ISP 的 photofinishing 管线中提供五种可微分研究候选，并支持 A/C 对选定基底进行校正、分阶段预测/渲染、用户预训练分割器接入、配对数据训练、续训、评估和推理。Gain、GTM、chroma 和 gamma 网络沿用原实现；`baseline` 保留原局部网络。原来的 `photofinishing`、`main` 和 GUI 使用方式见各自文档。

本版本使用 `architecture_version: 2` 和 `tm_library_v2` checkpoint。A/B/E 的结构已改变，v1 库 checkpoint 会明确报错，需要重新训练或另行编写并验证迁移；原 photofinishing bare state dict 仍可严格加载。

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
  --num-train 12 --num-val 6 --size 64 --seed 123 --task semantic

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

生成器使用不同随机流和互不重叠的 scene/subject/burst ID 产生 train/val 样本，保存 float32 输入、对齐 sRGB 目标、软语义图和置信度。默认 `--task global` 保留已知标量亮度变换 fixture；`--task semantic` 依次循环 `portrait_backlight`、`two_people_unequal`、`side_light`、`no_person_window`、`night_silhouette`、`already_good`，每个划分至少六张可覆盖全部场景。语义目标按角色与照明施加有界标量校正，变化 skin RGB，不设置固定肤色或亮度；后三种场景保护输入。它们只是非物理的软件 fixture，不能作为自然图像质量基准。正式实验应自行准备真实、合法且对齐的数据，不在导入或测试期间下载数据集或外部预训练模型。

## 五种算子与共享管线

共享顺序为：归一化线性 sRGB → 原 Gain → 原 GTM → 选定 base → 可选 A/C correction → 原 chroma → 原 gamma → 显示 sRGB。独立候选可同时观察增益后的 `gain` 和 GTM 后的 `base`；`gain` 的值可能大于 1。base 与 correction 的输出都是 chroma 之前的线性 RGB。

| 候选 / `algorithm` | 实际机制 | 主要参数与诊断图 |
| --- | --- | --- |
| 原基线 / `baseline` | 保留原始局部 TM 网络，使用原训练模式前向路径 | 原管线中间结果；可叠加 A/C |
| GTM / `gtm` | 直接使用原 GTM 的线性输出，不运行原 LTM | 可作为校正基底；无可训练局部特征 |
| A / `region_curves` | 正增量构造单调共享曲线库；非负 gain 经 `u=x/(1+x)` 进入曲线；低分辨率图像/语义特征预测混合权重；每个专家用同一标量曲线处理 RGB，再顺序累加融合 | `num_experts`、`curve_bins`；`curves`、`weights` |
| B / `spatial_grid` | 预测三维 bilateral grid；全分辨率 guidance 采用 gain/base RGB 的廉价 6→1 affine guide；三线性切片得到 gate、三个 tone-curve 参数和 local gain，混合 GTM 与局部重映射 | `grid_size`、`grid_depth`；`grid`、`guidance`、`coefficients` |
| C / `gain_residual` | 预测低分辨率、有界 log2 EV 残差，边缘感知上采样后用同一个标量乘 RGB；零残差初始化。独立 C 的 anchor 是 GTM，组合 C 的 anchor 是选定 base | `max_ev`、`filter_radius`、`filter_eps`；EV 与 gain 图 |
| D / `exposure_fusion` | 从同一张增益后图像合成固定 EV 曝光，使用 `x/(1+x)` 压缩；构建 RGB Laplacian 金字塔和权重 Gaussian 金字塔，逐层归一化融合并重建 | `exposures`、`pyramid_levels`；`exposures`、`weights`、各尺度权重 |
| E / `base_detail` | log 亮度 guided filter 分解 base/detail；低分辨率预测空间 base 对比度/平移及独立 detail gain，边缘感知上采样后重建亮度与 RGB | `filter_radius`、`filter_eps`、`max_ev`；`base`、`detail`、`base_toned`、`detail_gain` |

A 的 shoulder 坐标保留可用的 `gain>1` 高光差异，不声称恢复采集时已裁剪的信息。曲线库是训练后跨图像共享的参数化变体，逐图预测的是混合权重，不是完整曲线；专家没有预设“皮肤必须提亮”或“天空必须压暗”的身份。顺序累加避免显式堆叠 K 张完整 RGB/查表索引，但训练 autograd 仍保存各专家中间值，不能据此声明恒定训练内存。B 的 grid 存储参数 logits，切片后再约束；不在全分辨率运行 width 通道特征 CNN。C 不添加无约束 RGB 残差。D 确实执行多尺度融合，固有的 RGB/权重金字塔仍有内存和计算成本，关闭诊断图也不能消除；单图合成曝光没有增加传感器信息或独立噪声样本。E 的空间 base 控制可以区分相同亮度的不同语义区域，detail 控制独立生效；固定滤波分解不等于语义分解。`filter_radius` 以输入像素计，分辨率变化会改变覆盖范围。最终显示输出会裁剪，需结合线性阶段诊断理解。

独立候选仍用 `algorithm` 选择；组合使用 `base_algorithm` 与 `correction_algorithm`。此时 `algorithm` 应省略或保留默认 `baseline`，不能同时指定另一个非默认选择。base 支持表中全部路径，correction 仅支持 `none`、A、C。例如原 LTM+C、B+C、D+A、E+C 不要求把 A/C 当作全场景替代器。

校正总是以选定 base 的 **chroma 前线性输出**为 anchor，A/C 的预测和渲染都观察该 anchor。A 将 `candidate-anchor` 限幅到 `±correction_max_delta` 后按 gate 混合；C 在 `exp2` 前门控标量 EV，保持后处理前的 RGB 色度比例。`correction_strength` 范围 `[0,1]`；手动 `correction_mask` 是同尺寸 B1HW `[0,1]` ROI，属于用户控制。设为零时线性 anchor 精确保留。`correction_semantic_channel` 是可选语义 gate，要求 S2 和有效通道；语义缺失或零置信度时该校正为零，base 仍运行。普通 S2 条件与这个显式校正 gate 的作用不同。

五种算子均能用于全场景或人像。实际风格和区域行为由训练数据、语义开关、损失及控制参数决定；算法名称本身不保证人像肤色、逆光保护或全场景泛化，评估结论只能覆盖所用数据。

当前 Python 阶段是同步接口，不是实际 ISP/NPU 调度或 DirectLink SDK 集成。局部控制预测仍依赖 Gain/GTM 后的图像，校正预测还依赖已渲染的 anchor；不能把“TM 后降采样”换成“先降采样再 TM”并声称等价。ConditionEncoder 先分别缩小 gain/base/置信度门控语义，再拼接，在 `analysis_size` 上运行特征 CNN；语义可以具有独立的低分辨率。原 baseline 和 fixed-reference 原 LTM 参考计算保留未优化的全分辨率兼容路径，不能把它们当作已经全部低分辨率化的硬件实现。

## S0 / S1 / S2：语义的三个使用方式

| 模式 | `semantic_mode` | 渲染时使用语义图 | 训练中的语义监督 |
| --- | --- | --- | --- |
| S0 | `none` | 不使用；即使传入也忽略 | 无语义辅助头 |
| S1 | `train_only` | 不使用；评估/推理不加载分割器，原权重文件也可不在部署机 | 从 manifest 或冻结分割器获得辅助标签；仅训练时计算 TM 辅助 logits |
| S2 | `explicit` | 使用 manifest 软图或冻结用户分割器输出，经置信度门控 | 可选区域损失；没有 S1 辅助头 |

默认语义通道依次是 person、skin、sky；它们是 `[0,1]` 概率，可重叠，不是 RGB 图或互斥类别 ID。默认 `semantic_channels: 3`，修改数量时必须同步修改标注含义。

没有标注且没有 model 来源的样本返回零 `semantics`、零 `confidence` 和零 `semantic_valid`，不会当成所有类别均为负例。已标注背景可以是全零语义图，同时保持有效标记为 1。`semantic_valid` 表示人工标注监督有效性；`confidence` 表示运行时语义信任程度，含义不同。S1 人工标签按 validity 监督，模型伪标签在分子中按 confidence 加权，分母按有效标签量归一化，因此整批低置信度也会减弱辅助监督；缺失标签保持零辅助损失。S2 中零置信度屏蔽普通语义条件，图像特征驱动的 base 仍有效。S0/S1 渲染不随 mask 改变。没有可训练候选的 baseline/gtm 不能使用 S1，因为它们没有辅助头所需的特征。

## 加载用户预训练分割器

本库没有附带生产分割权重，也不会下载模型。`segmentation` 顶层配置支持 `none`、`torchscript`、`factory`。provider 固定为 eval/no-grad，参数不进入 TM optimizer，TM checkpoint 记录路径、来源、预处理与类映射，**不内嵌外部分割权重**。S0 从不调用 provider；S1 只在训练时调用，S2 可在训练/评估/推理调用。

来源 `source: manifest` 是默认值，不加载模型；`model` 使用模型输出；`prefer_manifest` 在人工标注有效位置逐通道覆盖模型伪标签，未标注处采用模型输出。模型输出先转为概率，再按有序 `class_groups` 映射成 person、skin、sky 等 TM 通道。组内概率求和并截断到 1；对 softmax 互斥类别是 union，对 sigmoid 独立类别则只是明确的 capped-sum 映射，不能当作校准的概率并集。

下面是用户 TorchScript 模型的独立 YAML（例如 `/path/to/segmenter.yaml`），可用于 `--semantic-config`。模型、通道索引、输出容器和预处理均需按自己的模型调整：

```yaml
segmentation:
  backend: torchscript
  source: model
  weights: /path/to/USER_SEGMENTER.ts
  input_encoding: srgb
  input_size: [256, 256] # [H,W]；适配器输入始终是库的线性 sRGB
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
  output_mode: softmax_logits
  output_key: out # 若模型直接返回 BCHW tensor，应删掉此字段
  class_groups: [[1], [2], [3]] # 仅示例：模型 1=person、2=skin、3=sky
```

`input_encoding: srgb` 会先从线性输入转换为显示 sRGB，再 resize 并逐通道 `(x-mean)/std`；模型若训练在线性域应改为 `linear_srgb`。输出可为 tensor、由 `output_key` 选择的 dict，或由 `output_index` 选择的 tuple/list；key/index 二选一。`output_mode` 可为 `softmax_logits`、`sigmoid_logits`、`probabilities`。显式启用模型来源必须声明 `class_groups`，组数等于 `semantic_channels`，索引为模型输出通道，不能直接填类别名称或假定 RGB。输出的尺寸、有限性与概率范围会校验；provider 返回模型输出分辨率的软图，集成适配器按需要对齐到输入用于损失/指标。

若模型提供可信 confidence，可用 `confidence_key` 或 `confidence_index` 选择 B1HW 或 TM 通道 BCHW 输出；否则默认 confidence 是模型原始通道最大概率。这只是启发式，并未校准：特别是独立 sigmoid 类别全部接近零时，即使背景判断确定，也会得到低置信度。

已有 Python 架构和 tensor state dict 时使用 factory。让本地可导入模块 `my_segmenter.py` 的 `build_model(num_classes=...)` 返回对应 `torch.nn.Module`，将上例 backend/weights 部分替换为：

```yaml
  backend: factory
  factory: my_segmenter:build_model
  factory_kwargs: {num_classes: 4}
  weights: /path/to/USER_SEGMENTER_STATE.pt
  state_dict_key: state_dict # bare tensor state dict 则删掉；只选一层 key
```

factory 显式导入用户本地模块构造架构，然后以 `weights_only=True` 严格加载 tensor state dict，不接受任意 pickle 完整模型对象。模块及权重须来自可信来源；架构、类别数和权重 keys 必须匹配。`configs/tm_library/gain_residual_torchscript.yaml` 是需要替换用户路径的示例，仓库没有该权重。

S2 部署时可以覆盖 checkpoint 中记录的模型配置/路径；train 同样接受这两个参数，但 resume 会检查语义配置身份，不能静默改变来源、预处理或映射：

```bash
python -m tm_library.infer --checkpoint /tmp/tm-run/best.pt \
  --manifest /tmp/tm-fixture/val.jsonl --output /tmp/tm-s2 \
  --semantic-config /path/to/segmenter.yaml \
  --semantic-checkpoint /path/to/USER_SEGMENTER.ts
```

`evaluate` 也支持同样的覆盖项。YAML 权重相对路径基于 YAML 所在目录，命令行权重路径基于当前工作目录。

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
| `scene` / `camera` / `scenario` | 场景/相机/测试情形分组报告；严格 scene 划分要求有意义的 scene ID |
| `burst_id` / `subject_id` | 可选连拍组/主体 ID；可开启对应跨划分重叠检查 |
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

`--semantics-dir`、`--confidence-dir` 和对应 `--semantics-layout` 可选。构建器严格匹配大小写一致的文件 stem，包括元数据与标注，不会独立排序后按位置配对；重复或不匹配 stem 会报错。它不会把 reference 目录自动变为 target。train/val 应按采集场景、连拍组或主体隔离，避免同一图像/同一场景变体泄漏；跨相机实验还应固定相机划分。不要将多次采集全部填成一个无意义 scene，或给同一场景的每张图另起 scene 来绕过检查。

## 配置与训练

`configs/tm_library/` 提供以下实验配置。核心 YAML 结构如下（YAML 中的相对路径基于该配置文件所在目录；命令行路径覆盖项基于当前工作目录）：

| 实验 | 配置 |
| --- | --- |
| 原基线与独立 A–E | `baseline.yaml`、五个算法名 `.yaml` |
| 组合 base+correction | `baseline_c.yaml`、`spatial_grid_c.yaml`、`exposure_fusion_a.yaml`、`base_detail_c.yaml` |
| C 的 S1 / S2 | `gain_residual_train_only.yaml`、`gain_residual_explicit.yaml` |
| 用户分割器占位示例 | `gain_residual_torchscript.yaml` |
| 固定原后处理的算子诊断 | 五个 `<algorithm>_diagnostic.yaml` |
| 单独改变区域损失的消融 | `gain_residual_region_loss_ablation.yaml` |

```yaml
model:
  architecture_version: 2
  base_algorithm: baseline
  correction_algorithm: gain_residual
  correction_strength: 1.0
  correction_max_delta: 0.25 # A 校正的最大加性差；C 使用 max_ev
  semantic_mode: none
  semantic_channels: 3
  width: 16
  analysis_size: 64
  max_ev: 1.0
  freeze_backbone: false
  post_mode: adaptive
  use_3d_lut: false
data:
  train_manifest: /tmp/tm-fixture/train.jsonl
  val_manifest: /tmp/tm-fixture/val.jsonl
  image_size: 64
  augment: true
  split_policy: scene
  group_checks: [burst_id, subject_id] # 可选；检查提供了 ID 的样本
training:
  protocol: joint
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

训练使用 AdamW、有限损失检查、梯度裁剪和 scheduler。RGB L1 比较对齐显示 sRGB，另有 log 亮度、梯度及可选区域平衡项；`semantic` 是 S1 的辅助 BCE 权重。语义 mask 可以参与监督损失而不进入 S0/S1 渲染器。`freeze_backbone: true` 冻结原网络，仅训练候选/辅助头；非 baseline base 中未使用的原 LTM 本来就冻结。建议先以小样本确认数据域和梯度，再开展真实数据训练。

`training.protocol: joint` 训练完整可训练管线；`operator_diagnostic` 必须同时设置 `freeze_backbone: true` 和 `post_mode: fixed_reference`。`adaptive` 中 chroma LUT/gamma 控制随候选像素变化，即使权重冻结也会变化。`fixed_reference` 每张图从同一冻结的原 LTM 参考预测并缓存 LUT/gamma，应用到候选，适合隔离局部算子影响；它不是全数据共用一组常数控制，也不把候选线性输出换回原基线。

S0/S1/S2 示例使用匹配的图像目标权重 `rgb: 1.0`、`log_luma: 0.1`、`gradient: 0.1`、`region: 0.0`；S1 增加辅助监督，而非偷偷改变图像区域目标。开启 `region` 是单独消融，应与匹配目标实验分别报告。默认 `data.split_policy: scene` 要求有意义且不重叠的 train/val scene，违反会报错；`id_only` 是显式兼容选项，仍检查图像 ID 和解析后的 input 文件路径重叠。可选 `group_checks` 检查提供的 `burst_id`/`subject_id` 重叠，应在记录真实 ID 后开启；`unknown` 占位也可能被视为重叠，不能替代正确组织真实采集划分。

`rgb`、`log_luma`、`gradient` 至少一个权重必须为正；不能只依赖区域或语义损失，否则未标注样本会失去图像监督，验证集可能出现无意义的零分。

训练保存最后一轮 `last.pt`、验证集最佳 `best.pt` 和 `history.json`（逐轮损失及划分检查结果）。最佳模型只由 val 决定，不应使用 test 选择 checkpoint。续训时 `epochs` 表示总训练轮数：

```bash
python -m tm_library.train --config configs/tm_library/gain_residual.yaml \
  --resume /tmp/tm-run/last.pt \
  --train-manifest /tmp/tm-fixture/train.jsonl \
  --val-manifest /tmp/tm-fixture/val.jsonl \
  --output-dir /tmp/tm-run --epochs 4
```

训练 checkpoint 包含 v2 模型配置、模型权重、分割适配配置和训练状态；续训恢复 optimizer、scheduler、RNG 及非递归的完整最佳快照。模型、数据划分、损失、训练及分割来源/预处理/类映射/权重路径身份会严格检查；总 epochs 和输出目录可按续训约定改变。当前要求 `num_workers: 0`，避免轮次边界恢复遗漏预取 worker 随机状态。CPU 相同输入与环境的精确恢复验收以 [v2 review](../docs/tm_library_v2_review.md) 的实际证据为准；CUDA 的 `grid_sample` backward 可能不确定，训练器采用 `warn_only=True`，不声明 CUDA 精确续训。保留 manifest、外部分割权重和实际运行参数以复现。不要加载来源不可信的 checkpoint。

## 原 checkpoint 与 Python API

原始 checkpoint 是原 `PhotofinishingModule` 的 bare state dict。通过 `load_upstream` 严格加载，不会静默跳过不匹配层；`use_3d_lut` 必须与 checkpoint 对应（文件名带 `-lut` 的模型使用 `true`）。候选仍需要训练自己的局部算子；原权重不代表候选已经训练好。

```python
import torch
from tm_library.config import ModelConfig
from tm_library.model import TMModel

torch.set_num_threads(1)
model = TMModel(ModelConfig(
    base_algorithm="baseline", correction_algorithm="gain_residual",
    semantic_mode="none", correction_strength=0.5,
))
model.load_upstream("photofinishing/models/photofinishing_s24-style-0.pth")
model.eval()
with torch.no_grad():
    image = torch.rand(1, 3, 64, 64)
    result = model(image)
print(result["output"].shape, result["maps"].keys())

torch.save(model.checkpoint(), "/tmp/tm-model.pt")
restored = TMModel.from_checkpoint("/tmp/tm-model.pt").eval()
```

`output` 是显示 sRGB，`linear` 是校正后、chroma 前的线性 RGB，`maps` 是诊断图；S1 仅训练时另有 `semantic_logits`。输入为 float BCHW RGB `[0,1]`。`.to(device)` 同时移动上游色彩矩阵。默认 `return_maps=True`；不需诊断时传 `False`，结果 `maps` 为空，避免保留诊断张量。没有 correction 时传入校正 ROI/strength 会报错。`model.checkpoint()` 是仅模型的 v2 保存入口；训练 CLI 另外保存训练/分割配置与状态，手动使用 provider 时应自行保留其配置。

分阶段调用对应同一个计算流程；以下例子接续上面的 `model`、`image`，显示原 LTM+C 的调用顺序：

```python
with torch.no_grad():
    context = model.prepare_global(image)
    base_controls = model.predict_controls(context, stage="base")
    base = model.render(context, base_controls, stage="base", return_maps=False)
    anchor = base["image"]
    correction_controls = model.predict_controls(
        context, stage="correction", anchor=anchor,
    )
    corrected = model.render(
        context, correction_controls, stage="correction", anchor=anchor,
        correction_strength=0.5, return_maps=False,
    )
    staged = model.finish(corrected["image"], corrected["controls"], return_maps=False)
    direct = model(image, correction_strength=0.5, return_maps=False)
    torch.testing.assert_close(staged["linear"], direct["linear"], rtol=0, atol=0)
    torch.testing.assert_close(staged["output"], direct["output"], rtol=0, atol=0)
```

`prepare_global` 返回含 `image`、`gain`、`base`、Gain/GTM 控制的 context；`predict_controls(context, semantics=None, confidence=None, stage='base', anchor=None)` 返回低分辨率算子 controls；`render(..., correction_mask=None, correction_strength=None, return_maps=True)` 返回 `image`、`features`、`maps`、`controls`。`finish(linear, controls=None, return_maps=True)` 执行 chroma/gamma；fixed-reference 模式必须传本帧 render 返回的 controls，其中包含缓存参考 context。语义在 base 与 correction 的 predict 中分别传入，correction 只能在 anchor 已存在后预测。无 correction 时直接 `finish(base['image'], base['controls'])`。若需要整套诊断图，`forward` 会合并 base/correction/post/global 图；单独 `finish` 仅返回后处理图。

每个 A–E 算子也有 `predict(gain, base, semantics=None, confidence=None)` 和 `render(gain, base, controls, return_maps=True)`，`forward` 调用两者。stage 是普通 Python 边界，没有异步执行器。未叠加 correction 的 `baseline` + S0 在原网络支持的尺寸上保持原训练模式前向一致；极小输入用 replicate padding 后裁回。原源码和 checkpoint 没有改写。

手动调用分割适配器时：

```python
from tm_library.semantics import SegmentationConfig, SemanticProvider

seg_config = SegmentationConfig.from_dict({
    "backend": "torchscript", "source": "model",
    "weights": "/path/to/USER_SEGMENTER.ts",
    "input_encoding": "srgb", "input_size": [256, 256],
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "output_mode": "softmax_logits", "class_groups": [[1], [2], [3]],
}) # 这里假设模型直接返回 tensor；dict 输出另设 output_key
provider = SemanticProvider(seg_config, semantic_channels=3, device="cpu")
s2_model = TMModel(ModelConfig(
    base_algorithm="spatial_grid", correction_algorithm="gain_residual",
    semantic_mode="explicit", correction_semantic_channel=0,
)).eval() # 本例未训练；不要将随机 TM 权重当作画质示例
with torch.no_grad():
    predicted = provider(image)
    result = s2_model(image, predicted["semantics"], predicted["confidence"], return_maps=False)
```

`apply_semantic_provider(batch, provider, semantic_mode, training)` 是训练/部署集成入口，返回副本并对齐模型软图以供损失/指标使用；S0 和 S1 评估/推理不调用 provider。此函数不需要 target。

## 评估、推理与证据边界

评估逐样本计算再汇总，并按 scene、camera、scenario 和有效语义区域报告。`--output` 可以是 JSON 文件，或目录（写入 `metrics.json`）。PSNR 使用单位范围 sRGB，MSE 下限 `1e-12`，完美匹配记为 120 dB；SSIM 使用 RGB Gaussian 11×11、sigma 1.5、replicate 边界与总体协方差。`luma_l1` 是 Rec.709 加权 sRGB code values 误差，不是物理线性亮度；`clipping_fraction` 是最终 `<=0` / `>=1` 通道比例，不能单独反推出裁剪前高光损失。参数量包含保存的原 LTM 等未执行参数，区别于仅运行算子的参数量。

v2 的 `delta_e76`、`chroma_l1` 使用 sRGB→D65 Lab；`hue_degrees` 是最短色相角误差，仅统计输出和目标都满足 `C*>1e-3` 的像素，并报告 `hue_valid_fraction`。区域报告补充 Lab L/chroma/hue 均值。`semantic_boundary_luma_gradient_l1` 是有效语义边界上的 **GT 参考亮度梯度误差**，不是通用 halo 检测器。区域使用 semantics×validity×confidence 加权；若区域只来自模型伪标签，报告也依赖其质量。

`linear_l1`、`linear_log_l1`、`linear_gradient_l1` 比较未裁剪的 chroma 前输出与解码后的最终 sRGB target；后者包含下游色彩/影调，因此它们是阶段诊断，不是有标定的局部算子 GT 误差。`linear_below_zero_fraction`/`linear_above_one_fraction` 记录该阶段范围。`image_metrics(..., preclip=...)` 可额外检查调用方提供的原始最终显示 sRGB 裁剪前值；当前 evaluate CLI 没有传入该值，不能将线性越界统计称作最终显示 preclip 测量。

不同数据域、分辨率、设备或计时范围不能直接横比。延迟以 batch=1 测量，排除首次 pipeline warm-up，前后同步设备；`tm_only` 复用已准备的语义，`segmentation_plus_tm` 包括分割预处理、推理、软图对齐及 TM。两者均排除解码、CPU→设备传输、指标和保存，不是完整 RAW ISP 延迟；S1 没有运行时 segmenter。报告包含输入高宽、`analysis_size`、设备和计时说明，没有 NPU/远程硬件实测声明。实现细节见 `metrics.py` / `evaluate.py`。

推理从 checkpoint 恢复配置，不需要 GT，默认输出 8 位 sRGB PNG 和 `metadata.json`；输入精度保留不代表 PNG 导出也是高精度。默认不请求诊断图；加入 `--save-maps` 保存 float NPZ 图，保留 batch 维，未做显示归一化。文件名使用清理后的 ID 加哈希，metadata 记录原 ID 映射。S2 可以使用可信同几何的 manifest mask 或上述用户模型适配器；S0/S1 不加载分割权重。没有附带生产模型或自动下载逻辑。

CPU 测试覆盖定义性机制、梯度、奇数/极小/黑白输入、数据精度与颜色、语义门控、组合与 staged 等价、原基线一致性、优化和 checkpoint/CLI 闭环。v2 的实际审查与运行证据见 [v2 review 记录](../docs/tm_library_v2_review.md)；[v1 review 记录](../docs/tm_library_review.md) 保留为历史结果，不能把其通过次数当作 v2 验收，也不能用合成验证替代真实相机 IQ 测试。

尚未证明的范围包括：真实数据上优于原 LTM、跨相机泛化、肤色偏好与主观评价、视频时序稳定、真实多曝光去鬼影、低光去噪收益、移动端功耗/内存/吞吐，以及硬件 ISP 的量化、颜色标定和在线集成。需要分别开展固定划分消融（baseline/A–E、S0/S1/S2）、真实高分辨率测量和部署验证。

## 公开范式来源

这些链接说明相关研究背景；实现中的具体参数化、训练目标和语义控制以本库代码为准，不把论文或上游结果转述为本库实验结果。

- [Modular Neural Image Signal Processing](https://arxiv.org/abs/2512.08564)：本仓库共享 Gain/GTM/chroma/gamma 管线与原局部基线。
- [Deep Bilateral Learning for Real-Time Image Enhancement / HDRNet](https://groups.csail.mit.edu/graphics/hdrnet/)：B 的低分辨率 bilateral grid 与 slicing 范式；本库切片的是五个局部 tone controls，并非原论文的 affine RGB 变换复现。
- [Exposure Fusion — Mertens、Kautz、Van Reeth](https://jankautz.com/publications/exposure_fusion.pdf)：D 的多分辨率融合范式；本库使用单图合成曝光和学习权重，区别于真实 bracketed captures 的融合。
- [Fast Bilateral Filtering for the Display of High-Dynamic-Range Images — Durand、Dorsey](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf)：E 的边缘感知 base/detail 思路；本库采用 log 亮度 guided filter 和分别学习的控制。

使用原仓库代码和权重时仍须遵守 [原许可证](../LICENSE.md)，并按原 README 引用上游论文。
