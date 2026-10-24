> **单文件回传请改用 [REPORT_SINGLE_FILE_TEMPLATE.md](REPORT_SINGLE_FILE_TEMPLATE.md)。本文件保留为历史多文件归档模板，其中 CSV/图片/日志包要求不适用于复制粘贴回传。**

# TM 五方案实验回传报告

模板版本：1.0；日期：2026-10-24。适用于原始 LTM baseline 与 A–E 的配对训练结果，也允许记录组合方案。

> 这是记录模板，不是新一轮训练任务。先交现有结果；缺失项填 unknown / not_run 并写入 missing_items.csv。没有测量的数值留空，不填 0。不要根据测试集挑 checkpoint、排除坏图或改指标口径。

## 0. 工程人员先完成的最小交付

- [ ] 填本报告第 1–8 节；第 9 节可选。结论可以是“暂不能判断”。
- [ ] 原始 LTM baseline + A/B/C/D/E 均有 run_registry.csv 记录；未测 baseline 必须注明。
- [ ] 交原始训练日志、最终实际配置和评估原始 JSON，不只截图。
- [ ] 交完整评估集 dataset_manifest.csv、逐图 per_image_metrics.csv 和预测文件索引。
- [ ] 交同一组样本的输入、对齐 GT、六种输出和关键 ROI 裁剪；见第 6 节。
- [ ] 交 checkpoint 清单、SHA256 和获取位置；checkpoint 太大可以不打包，但原文件保留。
- [ ] 已有区域指标、耗时和诊断图则一起交；没有则注明，不阻塞首轮分析。
- [ ] 压缩包目录和表格能够通过 run_id / sample_id 对应。模板 CSV 只有表头，须填真实记录。

最低分析条件是“可比配置 + 逐图指标 + 同样本原图/GT/输出”。缺少其中一项，仍可提交，但部分问题无法回答。

## 1. 本轮到底比较什么

- 工程负责人 / 日期：
- 数据集及版本 / camera 来源：
- 实际代码仓库 / branch / commit / 未提交 diff 文件：
- 运行环境版本文件：
- 当前主要目标：匹配 paired GT / 跨相机迁移 / 人像质量 / 全场景影调 / 其他：
- 期望验证的问题（不提前写成已成立结论）：
  1. A–E 是否优于原始 LTM baseline？
  2. 优势/退化发生在哪些场景、区域？
  3. 差异来自算子、训练条件、后处理还是语义？
- 输入属于：融合后 HDR linear RGB / 归一化 linear sRGB / 已成片 sRGB / 其他：
- 测试是否独立于训练和 checkpoint 选择：
- 若没有独立 test，本报告只能称为 validation 比较：

## 2. 训练可比性与复现身份【必交】

实际数值填 run_registry.csv；完整最终配置和原始日志放 configs/、logs/。

| 核查项 | 各方案是否一致 | 实际差异及影响 |
| --- | --- | --- |
| train/val/test 划分、scene、同人物/连拍隔离 | 待填 | |
| 输入曝光尺度、归一化、输入编码、WB/CCM 前处理 | 待填 | |
| 公共 backbone 初始权重及 SHA256 | 待填 | |
| Gain/GTM/chroma/gamma 冻结与优化器参数组 | 待填 | |
| adaptive / fixed_reference 后处理模式 | 待填 | |
| 初始化策略；baseline 是原权重还是同协议重训 | 待填 | |
| image loss / region loss / 辅助语义 loss 权重 | 待填 | |
| optimizer、LR、scheduler、weight decay、AMP | 待填 | |
| batch size、梯度累积、有效 batch、总 optimizer steps | 待填 | |
| epoch、crop 尺寸、resize、数据增强 | 待填 | |
| 随机种子及实际重复次数 | 待填 | |
| best checkpoint 选择集合、指标、方向、epoch | 待填 | |
| resume / 中断 / OOM / 数据跳过 / NaN / early stop | 待填 | |
| 推理尺寸、tile/padding、输出精度及保存编码 | 待填 | |

注意：同 epoch 不一定同训练预算；同 seed 不保证所有模型经历相同随机序列。真实差异保留，后续按“可归因/有混杂”分组分析，不事后隐藏或修改。

### 语义使用情况

- 各 run 的 S0 / S1 / S2（按实际填写）：
- 语义来源：人工 mask / manifest 伪标签 / 冻结分割模型 / 无：
- 模型架构、权重 SHA256、预处理、类别映射和 confidence 定义：
- 分割模型是否 frozen + eval/no-grad，是否误入 optimizer：
- 训练和推理的语义来源是否相同，有无 GT 语义辅助推理：
- 区域 loss 是否只用于某些方案：
- 评估 ROI 来源、有效性、confidence 权重是否对所有方法完全相同：

只训练一种语义模式不能证明“语义有效”。独立的 S0/S1/S2 重训消融缺失时写 not_run；不要求此轮补齐。

## 3. 数据与评估协议【必交】

dataset_manifest.csv 至少完整列出本轮所有评估样本；scene_id 不得使用同一个 unknown 代替所有场景。训练/验证划分也交原 manifest，或匿名 sample_id + scene_id 清单，以便检查重叠；不要求打包训练图像。无法恢复清单时标记 unavailable。

- train / val / test 图像数、scene 数、camera 数：
- 与训练是否同相机；实际跨相机设置：
- 划分方式及 scene / burst / subject 交叉检查结果：
- paired GT 类型：同 RAW 人工渲染 / 相机 JPEG / 生成伪 GT / 其他：
- 源与 GT 是否逐像素对齐，如何检测及处理视差、运动、遮挡：
- 无效样本、无效像素、异常样本处理规则及数量：
- 输入是否保留 >1 的线性高光；输入高光是否已裁剪：
- 评估在 full resolution / resize / crop 上进行；具体规则：
- crop border / padding / tile / 评估 mask：
- 指标脚本版本、命令、完整参数、浮点还是导出 PNG 结果：
- 图像范围、sRGB 编码、RGB/BGR 顺序、clamp 阶段：
- 平均方式：先逐图算再算术平均 / 全局汇总 MSE 再转 PSNR / 其他：
- 已有测试结果是否曾参与调参、选模型或选最好 seed：

同场景未对齐 iPhone 参考不是像素 GT。若目标是生成伪 GT，指标只能直接反映“匹配该伪 GT”，不能独立证明真实恢复或 iPhone 风格迁移正确。相同场景多帧不是多个独立场景。

### 指标口径

1. 必交 PSNR、SSIM 逐图值及运行状态；不相同的实现/范围不得混在同一排序。
2. 原 v2 口径可沿用：PSNR 使用 [0,1] sRGB；SSIM 为 RGB Gaussian 11×11、sigma 1.5。若采用别的库或设置，填实际值，不必为符合模板重算已有报告。
3. 优先补交现有 luma_l1、Lab D65 delta_e76、chroma_l1、hue_degrees、hue_valid_fraction；没有则留空。delta_e76 不是 CIEDE2000；sRGB code-value luma 误差不是线性物理亮度误差。
4. LPIPS 等感知指标可选，记录实现/权重/输入范围；不单独作为“风格更美”的证据。
5. 最终输出通道落在 <=0 或 >=1 的比例，称 endpoint/clipping fraction，不等于传感器高光裁剪或 HDR 信息恢复。
6. chroma 前 linear_* 诊断不等于最终显示 preclip。只有真实导出的最终显示裁剪前值才填写 final_display_preclip 统计；不能用线性阶段越界率冒充。

## 4. 训练过程【必交原日志，可整理 CSV】

training_history.csv 一行对应一次原始记录；有 global_step 填 step，不强制把 step 日志重建成 epoch。保留 loss 各分量、LR、val 指标、非有限值和恢复事件。训练 loss 权重不同，不直接横比总 loss。

- 各方案 best / last checkpoint 是否一致；不是则两者路径分别记录：
- 验证频率 / 是否完整验证集：
- 是否收敛、震荡、过拟合、异常下降：
- 训练集与验证集差距（使用相同可比较指标）：
- 关键异常及其原日志位置：

## 5. 结果总表与逐图差异【必交】

下表由同一 protocol_id、同一评估集合、同尺寸和同指标实现下的数据汇总。失败、丢图数量必须列出。完整 paired 分析使用六模型共同成功的样本，并另外报告各模型失败/缺失率；不得默默删除坏图。

| run_id / 方法 | seed | split | scene / image 数 | PSNR | SSIM | ΔPSNR vs baseline | 成功/失败/缺失数 | val 选定 epoch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 原始 LTM baseline | | | | | | | | |
| A：region_curves | | | | | | | | |
| B：spatial_grid | | | | | | | | |
| C：gain_residual | | | | | | | | |
| D：exposure_fusion | | | | | | | | |
| E：base_detail | | | | | | | | |

per_image_metrics.csv 保留每张图的值；Δ 与胜率、最差退化、场景级 bootstrap 等可由分析端统一计算。单 seed 仍可交，但不能据图像级置信区间推断训练随机性的稳定性；若已有多 seed，所有 seed 都交，不只交最好的。

### 场景分层（允许多标签，禁止子组数量相加当总数）

| 子组 | scene/image 数 | baseline | A | B | C | D | E | 明显退化说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 单人 / 多人 | | | | | | | | |
| 逆光 / 明暗半脸 / 室内窗景 | | | | | | | | |
| 低光 / 夜感 / 剪影 | | | | | | | | |
| 无人像 / 天空 / 高光灯源 | | | | | | | | |
| 已经良好 / 低反差 | | | | | | | | |

若现有数据没有可靠标签，填 unknown，优先为可视化案例人工标注，不要求先标全数据集。肤色分层如有可信记录可加，不凭亮度自动推断族群或肤色类别。

### 区域指标（有可信 ROI 时交 region_metrics.csv）

至少区分 person/skin/sky/background 中实际存在的区域；可增加 face/highlight。不存在、mask 无效或面积太小时记录状态并留空指标，不填 0。

- 所有方法使用同一套同几何 ROI；不能让各方法输出自己定义有利 ROI。
- 优先 GT/人工核验 ROI；冻结分割器生成的 ROI 需明确是伪标签。
- 给出区域有效面积、加权面积、标签来源及加权方法；不要只交区域均值而没有 ROI。
- 区域 PSNR 需明确如何从区域加权 RGB MSE 转换；区域 SSIM 若非已有可靠实现则不必交，禁止把 ROI 外填零后的整图 SSIM 当区域 SSIM。
- 边界梯度误差不是通用 halo 检测，需结合原尺寸边界案例。

## 6. 可视化与失败案例【必交】

### 选择规则

先保存完整评估预测，工程端保留；首轮传回建议 20–30 个互异样本，数量不足则全交。包含：

- 跨场景固定随机种子抽样的普通样本（不按分数筛选）。
- 各方法相对 baseline 最差的 2–3 张（取并集，可重叠，不需凑满）。
- 各方法最好的 1–2 张（取并集）。
- 已知关键人像/无人像/夜景/已良好案例；若缺失写明。

这组选择用于定位问题，不代表总体胜率。case_index.csv 记录 selected_by、选择规则、sample_id 和全部文件路径；同一案例必须能横向看到六个方法，不能每个方法只展示各自最好图片。

### 每个案例交付

- 输入、对齐 GT、baseline、A/B/C/D/E 输出；同一尺寸、同一显示编码。
- 候选预测的 float32 [0,1] 显示域结果优先保存 .npy/.npz；已有 8-bit PNG 也可交，但标明量化和指标是在导出前还是导出后计算。
- 输入线性数据如 .npy/.npz/.tiff，必须写明键名、layout、范围、encoding；不要仅传“线性数据直接当 JPEG”的暗图。
- 1:1 原尺寸文件 + 人脸、主体/背景交界、高光和暗部纹理等关键 ROI；所有方法的 ROI 坐标完全相同。
- 拼图可交，但不得单独作为证据；禁止自动对每格独立归一化、自动曝光或自动对比拉伸。
- 若有 S2：保存原始软语义、confidence 和可视化；图例明确每一通道。
- 有图像/ROI缺失，写清原因并保留失败状态。

主观观察逐项写“改善/相同/退化/不确定”和可定位描述：人物亮度、面部塑形、肤色、背景高光、前背景关系、局部反差、噪声/halo/接缝。主观偏好不等同于 GT 误差；如已有多人盲评，附匿名化逐人原始评分、左右随机化和选图规则，未做则 not_run。

## 7. 算子诊断【有则交；关键失败样本优先】

使用 float NPZ 保留实际数值，显示热力图只用于辅助。文件名含 run_id、sample_id；记录维度/键名及全分辨率/小图位置。推荐同一组案例导出，不强制全测试集导出大张量。

| 方法 | 希望取得的原始图/控制 | 收到后主要检查 |
| --- | --- | --- |
| 共用 | gain/GTM/局部输出/最终输出；实际可导出者 | 问题发生阶段；后处理是否放大或掩盖差异 |
| A | curves、weights | 专家是否塌缩、权重是否常数、区域过渡；曲线是共享训练参数，不是逐图预测 |
| B | grid、guidance、coefficients | 网格/guide是否退化，五控制是否饱和，边界控制是否合理 |
| C | lowres_ev、ev、gain、correction_gate（若配置） | 是否恒零、频繁触顶、全局偏移，噪声区域是否过度提亮 |
| D | exposure_ev、weights、weights_scale_* | 是否总选同曝光，尺度间变化、边界与噪声；曝光候选固定不当 |
| E | base、detail、base_toned、base_contrast、base_shift、detail_gain | base/detail是否各起作用，细节是否放大噪声，滤波尺度是否不合适 |

如已有汇总，可给 min/P05/median/P95/max、边界饱和占比及 A/D 权重熵。阈值必须随报告注明；不将单个统计异常直接解释为“机制失败”。D 曝光/多尺度图可能很大，按案例选取即可。

## 8. 运行成本【已有则交，否则标未测】

- 同设备、batch=1、同输入 H×W、同 analysis_size、dtype、编译设置；数据加载/保存/指标计算排除情况写清。
- 区分 tm_only（整段 TMModel，不含分割器）与 segmentation_plus_tm（含分割预处理、推理、对齐+TM）。local_operator_only、完整 RAW ISP 计时如另测，另列 scope，不互相混用。
- 保存 runtime_samples.csv 原始每次耗时，报告 warmup/repeats、计时 API、设备同步方式；建议充分 warmup 后至少 100 次稳定测量，现有其他设置如实保留。
- 汇总 p50/p95；有峰值显存则写 allocated/reserved、重置方法、包含模型/输入/分割器的范围及单位。不能用 nvidia-smi 总占用直接当单算法增量。
- 参数量区分 checkpoint 总参数和实际执行路径参数；没有工具核验者留空。MAC/FLOPs 为可选，标明 profiler 与未支持算子。
- 训练耗时、训练峰值显存和推理成本分别记录。GPU结果不外推为 NPU/ISP 的延迟、功耗或并行收益。

## 9. 已做消融/额外评估【可选，不作为首轮交付门槛】

| 项目 | done/not_run | 证据位置 | 当前能回答的范围 |
| --- | --- | --- | --- |
| adaptive vs fixed_reference，同协议训练/评估 | | | |
| 同图像目标与匹配损失的 S0/S1/S2 重训比较 | | | |
| S2 同 checkpoint 的正常/零语义/扰动语义推理 | | | 只检验依赖/敏感性，不替代重训消融 |
| A/C 零 strength / ROI 的 anchor 回退 | | | |
| baseline+C、B+C、D+A、E+C 等组合 | | | |
| 不同 analysis/grid/pyramid/filter 配置 | | | |
| 跨相机 / 无人像 / 外部数据 | | | |
| 视频时序与 AE 变化 | | | 仅静态 paired 数据不支持视频稳定结论 |

已有 adaptive checkpoint 仅切换到 fixed_reference 推理属于后处理敏感性实验，不等价于固定后处理训练的公平算子比较。已有测试集已用于多次选优时，后续需另外保留独立确认集。

## 10. 工程观察与待分析问题

- 整体表现最好的候选（暂定及依据，不要求唯一赢家）：
- 最值得保留的组合：
- 最明显失败类型及 sample_id：
- 是否观察到同一种方法“人像提升但背景退化”：
- 数据/训练/评估中已知的混杂变量：
- 结果与预期不符的地方：
- 缺失项、后续可补事项与负责人（同步 missing_items.csv）：

分析端将返回：可比性判断、整体/场景/区域 paired 差异、最差退化、证据支持的结论、可能原因与最小补实验。未提供的数据不会被当作已验证。
