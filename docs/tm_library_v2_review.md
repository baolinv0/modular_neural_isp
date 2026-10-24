# TM 算法库 v2 实现与独立 review 记录

日期：2026-10-05。v2 基础版本：`866c8738b31de2272cd3836047475e44cc6c90da`，工作分支：`feat/tm-algorithm-library`。

本记录描述 v2 变更与验收范围。接口以 [v2 设计](superpowers/specs/2026-10-05-tm-library-v2-design.md)、[实现计划](superpowers/plans/2026-10-05-tm-library-v2.md) 和 [README](../tm_library/README.md) 为准。[v1 review](tm_library_review.md) 是历史交付记录，其测试计数与 reviewer 结论不自动适用于 v2。原 `photofinishing` 源码和 checkpoint 保留，库通过封装复用 Gain/GTM/chroma/gamma。

## v2 变更

| 范围 | 实现变化 | 必须核验的行为 |
| --- | --- | --- |
| 组合 | 支持 baseline/GTM/A–E base 与可选 A/C correction；anchor 为所选 base 的 chroma 前 RGB | 原 LTM+C、B+C、D+A、E+C 实际训练/评估/推理；零 strength/ROI 精确保留 anchor；语义 gate 缺失回退 |
| 分阶段接口 | `prepare_global`、`predict_controls`、`render`、`finish`；A–E 各自 predict/render | direct/staged 输出与梯度一致；校正预测发生在 anchor 生成后；`return_maps=False` 空图 |
| 小图特征 | ConditionEncoder 在分别降采样并门控后拼接条件输入 | CNN 只处理 analysis 尺寸；奇数/极小尺寸、单独低分辨率语义、dtype/device 和形状约束 |
| A | 非负 gain 用 `x/(1+x)` shoulder；共享单调曲线库，顺序累加专家 | 可用 `gain>1` 高光差异保留、曲线/权重约束、梯度；不声称恢复已裁剪信息或恒定训练内存 |
| B | 真正三维 bilateral grid 与五参数 slicing；廉价 RGB affine guide | guide/grid 梯度、深度敏感性；无全分辨率 width 特征 CNN/分配 |
| C | 标量有界 EV；独立模式 anchor=GTM，组合模式 anchor=所选 base；exp2 前门控 | 原 LTM+C 零初始化回退、色度比例保持、EV 限幅与训练更新 |
| D | 保留真实 Gaussian/Laplacian 单图合成曝光融合 | 重建与多尺度机制、无诊断时不保留诊断图；固有金字塔成本仍存在 |
| E | 空间低分辨率 base contrast/shift 与独立 detail gain | 同亮度不同语义区域响应、detail 不抵消、极端输入有限；滤波半径按输入像素 |
| 后处理实验 | adaptive 或 frozen original-LTM `fixed_reference` 控制 | fixed-reference LUT/gamma 对候选扰动保持一致；冻结权重本身不等于固定控制 |
| 外部分割 | 用户 TorchScript 或本地 factory+严格 tensor state dict；冻结 eval/no-grad | 实际保存再加载的本地模型 fixture；预处理/类映射/输出容器校验；参数不进入 optimizer |
| 语义来源 | manifest/model/prefer_manifest；S0/S1/S2 不同调用边界 | 无 manifest mask 时模型标签生效；有效人工标注覆盖；伪标签 confidence 加权；S1 删除分割文件后仍可推理 |
| 数据与协议 | scene 默认严格划分，可选主体/连拍检查；语义场景 fixture；joint/diagnostic 配置 | overlap 拒绝、有意义 scene、targetless inference；S0/S1/S2 匹配图像目标，region 单独消融 |
| 保存与续训 | `tm_library_v2`/architecture 2，严格拒绝 v1；完整训练及最佳快照 | 原 bare checkpoint 严格加载；CPU epoch 边界 model/optimizer/scheduler/RNG/best/config 身份一致 |
| 诊断 | Lab D65 ΔE76、chroma/有效 hue、区域颜色、GT 边界梯度与线性阶段范围/误差 | 色彩数学与有效性；标签质量边界；计时区分 TM-only 与 segmenter+TM |

S1 的辅助头只在训练中输出 logits；评估和推理不实例化外部分割器。S2 的分割配置/权重路径允许部署覆盖，训练恢复则严格检查配置身份。外部分割权重不内嵌 TM checkpoint，库不附带生产权重、不下载模型。默认最大类别概率 confidence 是启发式，尤其不能把独立类别全零当作校准的背景信任；组内 capped-sum 映射也不是通用概率并集。

## 独立审查

实现与审查分离：operator reviewer、数据/分割/训练 reviewer，以及 foundation/组合/staged reviewer。问题交回实现方修复后再复核；独立审查结论以实际返回的检查范围为限，不把某一 reviewer 的结论扩展为全分支所有文件都已审查。

| 检查 | 状态与证据 |
| --- | --- |
| 算法算子与内存形状 review | 无剩余阻断项；73 项算子测试通过，另独立验证五种尺寸的 staged 输出/输入及参数梯度精确一致、gain 0–65536 有限反传、先门控再降采样与 analysis 尺寸拼接。 |
| 分割加载、数据、所有训练分支 review | 无剩余阻断项；70 项数据/分割/指标测试通过，置信度修复后另 6 项回归通过；审查来源选择、冻结加载、全状态续训、分支闭环与匹配损失。 |
| Foundation、组合与 staged review | 无剩余阻断项；独立验证带语义/ROI 的 B+C 两种后处理模式输出及全部梯度精确一致；七种 base 在同原权重下参考 LUT/gamma 相同；baseline+A/C 极端颜色和 ±16 EV 有限反传。 |
| README 示例与实现一致性 | 集成负责人独立执行模型保存/恢复和 staged 示例，零容差一致；相对文件链接检查通过。原 photofinishing 源码与权重未改。 |

发现的伪标签置信度问题：原辅助 BCE 若以 confidence 加权后再用同一 confidence 总和归一化，会抵消绝对信任衰减。修复后分母按标签可用性计，信任 1/0.1/0.001 的相同样例损失约为 0.693/0.0693/0.000693，梯度绝对值为 0.5/0.05/0.0005；confidence 负责缩小辅助信号，validity 负责定义标签可用性。

## 执行证据

最终本地验收使用 Python 3.12、CPU，OMP/MKL 各 1 线程。两套环境并行执行，耗时不是模型性能基准。只记录已返回的本地结果，远程 CI 状态另见 PR。

| 环境/验收 | 命令或场景 | 结果 |
| --- | --- | --- |
| PyTorch 2.5.1+cpu 全套 | `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/tm_library -q` | **264 passed**，142.71 秒 |
| PyTorch 2.14.1+cpu 全套 | 同上 | **264 passed**，144.36 秒；65 条 TorchScript API 弃用提示 |
| 独立与组合 CLI | configs/tm_library 全部 19 个 YAML | 每项训练 1 轮、resume 至第 2 轮、评估、无 GT 推理；每项 2 张 train/2 张 val、32×32 semantic fixture，输出 PNG/NPZ/metadata 完整 |
| S0/S1/S2 + 匹配损失 | 有/无人工标签、模型来源、partial prefer_manifest | 实际优化与来源策略测试通过；S0/S1/S2 图像目标一致，region loss 独立配置 |
| 用户模型接口 fixture | 实际保存/重载 TorchScript 与 factory state dict | 加载、结构化输出、类映射、预处理、冻结、非法配置回归通过；不代表用户生产权重已测试 |
| S1 运行时独立 | 训练后删除分割权重 | 独立评估及移除 target 的推理通过，external_segmenter_active=false |
| S2 部署覆盖 | CLI 权重覆盖；API 配置/权重覆盖 | 19 配置闭环中 TorchScript 配置使用真实加载 fixture 且 manifest 无 mask；缺失权重报错及覆盖恢复测试通过 |
| CPU 精确续训 | S0/S1/S2 连续训练 vs epoch resume | 模型、optimizer、scheduler、RNG、history、语义配置及完整非递归 best 快照一致 |
| 定义性与边界检查 | 零 ROI/strength、原基线、A 高光、E 平坦语义区、极端输入 | 全套测试及独立 review 探针通过；baseline+C 零 EV 保留原 LTM |
| 数据划分与颜色 | scene/burst overlap、精度/encoding/对齐与缺失标签 | 对应回归通过；可选 subject/burst 检查应提供真实 ID，unknown 占位也会被视为重叠 |
| 格式检查 | `git diff --check` 与最终 staged diff check | 通过 |

19 个 CLI 配置包括 baseline、A–E、C 的 S1/S2、四种组合、A–E 五个 fixed-reference diagnostic、独立 region-loss 消融、TorchScript 接入示例。GTM 路径另外由参数化训练测试覆盖。CLI 合成 fixture 只覆盖前两个场景的闭环，六种场景及保护行为另由数据测试覆盖。

复验方式见 README。合成 `global` 和 `semantic` 任务都是软件 fixture；后者包含角色/照明控制和 no-person/night/already-good 保护场景，不能用其指标给 A–E 排真实 IQ 名次。训练的 best 只由 val 选择，不使用 test 选择 checkpoint。

## 证据边界

Python stage 接口没有实现 ISP/NPU 异步调度、SDK、DirectLink、量化或移动端部署。预测仍在 Gain/GTM 后进行，correction 依赖已渲染 anchor；原 baseline 与 fixed-reference 的原 LTM 参考保留全分辨率兼容路径。小图 CNN 和 B affine guide 降低特定分配，不消除所有全分辨率渲染成本；A 的 autograd 保留专家中间值，D 的多尺度金字塔有固有内存成本。

`linear_*` 指标比较 chroma 前原始输出和解码后的最终显示 target，它们不是局部阶段标定 GT。最终输出 clipping 比例不等于裁剪前高光损失；当前 evaluate CLI 不供应原始最终显示 `preclip`，不能声称已测其范围。GT 参考语义边界梯度误差不是通用 halo 检测器，模型标签区域指标依赖标签质量。没有视频序列时不声称时序稳定。

本次没有用户真实相机数据、生产分割 checkpoint、GPU 或移动 SoC。尚未证明真实 IQ 改善、跨相机泛化、肤色偏好/主观评价、低光去噪、真实多曝光去鬼影、功耗/内存/吞吐或硬件颜色标定。单图合成曝光不会创造采集信息，A shoulder 不恢复已裁剪信息。候选是公开范式启发的研究实现，不是任何厂商算法或专利的精确复刻。使用原代码/权重仍须遵守 [原许可证](../LICENSE.md) 并引用上游论文。
