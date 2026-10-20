# S24 数据合成、AE–TM 训练与完整实验计划

2026-10-05。审查基线：`09aa3af`。这是下一轮实验设计和实现合同；本文中的 S24 生成、训练及新增接口**尚未执行或实现**。本轮实际验证见 [S24_PIPELINE_VALIDATION.md](S24_PIPELINE_VALIDATION.md)，工程任务见 [实施计划](../superpowers/plans/2026-10-05-s24-capture-tm.md)。沿用 [共享物理采集设计](DATA_SYNTHESIS_DESIGN.md)，不覆盖历史实验结论。

## 1. 可以开始什么，不能直接推断什么

建议首先完成 **S24 静态代理场景下的 Apple 单帧 AE–TM 与 Samsung 三帧 AE–TM**，各四组。S24 提供真实内容与目标外观；采集曝光、光子噪声、增益和饱和由声明过参数的模拟器产生。动态和实机验证分开扩展。

作者[数据说明](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md)明确：共 3,224 张；`raw_images` 是去马赛克、黑电平归一化后的三通道 PNG16；`denoised_raw_images` 是 Lightroom AI 降噪并经人工调整的伪参考；style-0 是专家渲染 JPG。本文未读到用户下载的数据文件，因此下载是否完整、图像大小、实际 JSON 单位和对齐误差仍待检查。

| 数据 | 本研究用途 | 边界 |
|---|---|---|
| `denoised_raw_images` | 静态 camera-linear 内容代理、共同参考 | 不等于噪声为零；不恢复源裁切或原生 Bayer |
| `raw_images` | 原始输入域核验、源裁切指示、残差诊断 | 不作为“干净 HDR 真值” |
| `srgb_images_style_0` | 静态主实验的固定期望外观 | 必须先核验几何；含 JPG 和专家处理误差 |
| `data/*.json` | `cam_illum`、CCM、来源曝光和场景分类 | ISO 不能唯一确定模拟/数字增益 |
| `blur_masks` | 排除隐私模糊参考区域 | 同时检查其边界；不是主体 mask |
| `masks` | 次光源子集分析 | 不是主体或 visibility mask |
| camera `srgb_images` | 独立原始 ISP 对照 | 作者指出与 RAW 有裁切位移，不能直接逐像素作主 GT |

最初可支持的结论是“在 S24 内容代理与指定传感器模型上，同一目标外观下的采集/渲染取舍”。原始被裁掉的高光、真实运动模糊、HDR 遮挡、手机连续 AE 与实机画质需要另外的证据。

## 2. 现有代码核验与优先缺口

| 功能 | 已有实现 | 下一步 |
|---|---|---|
| Bayer、时间积分、电子/ADC 链 | `acquisition.py` | 复用；补 S24 源资格验证 |
| Apple 12 动作、Samsung 24 计划 | `pipeline.py::build_acquisition_plans` | 实际参数写入 manifest；保持时序合法 |
| 归档生成与 RAW 重算 | `pipeline.py`、`pipeline_validation.py` | 扩展 per-scene 色彩和新参考合同 |
| 共享神经 AE、条件 TM | `learned_policy.py`、`learned_tone.py` | 复用网络；先训练小适配器 |
| 八组训练与离散期望 | `joint_experiment.py` | 加新目标/代价、lazy IO、设备和 batch |
| S24 导入 | 当前 float 导入器不接受 PNG16 | 新增显式 S24 导入器，禁止 PNG8 降位深 |
| S24 色彩 | 当前 manifest 只有全局 WB/CCM | 新增 scene 色彩配置，贯通参考、候选、推理；AE预览仍为前色彩camera RGB |
| style-0 GT | 当前 `fixed_target` 为解析 Reinhard/OETF | 新增明确的目标模式；`--weights` 不会改变 GT |
| AE 代价 | 旧 J 曾出现 Apple 过曝选择 | 加采集保真/新裁切项，先验收候选排序 |
| 大数据训练 | 整场景候选与准备缓存、CPU 单场景 | 候选分片、有限缓存、GPU/batch/resume |

本轮只写计划与验证记录，没有把上述缺口记为“已完成”。

## 3. 数据准入、划分和原 TM 域校验

### 3.1 下载数据审计

以 basename/stem 配对 RAW、denoised、JSON、style-0、mask，**不能用两个排序列表 zip 来假定对应关系**。检查实际 PNG 为 uint16/HWC/3 通道，解码后 RGB 顺序、有限值、维度、JSON 的正 `cam_illum` 与 3×3 有限 CCM。记录所有输入的 hash、缺项和拒绝原因。

保留官方 train/val/test。先检测跨 split 完全重复和疑似相同场景/session，再产生曝光、噪声、crop 和运动变体。若发现跨 split 源族重叠，发布冲突清单并隔离冲突样本；若必须改划分，另命名一个去重协议，不能冒称官方 benchmark。所有变体继承 `source_family_id`，相同源的不同风格也不能跨 split。

先从官方 train 取最多 64、val 16、test 16 个合格独立源做 pilot，按 scene/light/ISO 桶覆盖，不重新按 60/20/20 分这些图。具体可用数量由审计决定。调目标和损失只使用 train/val；pilot test 开启后不再用于修改协议，正式结论同时单独报告未参与 pilot 的测试源。

### 3.2 色彩与基线

本仓库 `utils/img_utils.py::imread/im2double` 的 PNG16 约定为 `RGB uint16 / 65535`，不再减黑、不逆 gamma。原 photofinishing 训练入口是

$$w_s=I_{s,G}/I_s,\quad H_s=CCM_s\,\operatorname{diag}(w_s)X_s,$$

其中 `I_s=cam_illum` 是 illuminant color，不是 WB gain；原入口对线性 sRGB 做 `[0,1]` clip。不能用 `gt_illum`、`pref_illum` 或单位矩阵悄悄替换它。

先做两个独立域核验：

1. **原算法复现 R0**：按原 `clip(raw_to_lsrgb(...))`、style-0 权重和原前向跑 noisy/denoised 输入，报告 PSNR/SSIM、亮度/色彩误差和对齐诊断。此项检查输入是否正确，不证明 AE 改善。
2. **模拟路径 R1**：无噪声、非饱和参考采集与共同固定前端对照，记录二次 CFA/去马赛克和新 wrapper 的误差底限。模拟主链保留 WB/CCM 后的 >1 值，仅在明确的 TM shoulder 处压缩；负色域值按固定规则裁零并报告比例。

`ConditionalToneMapper` 已有 shoulder 与局部曲线修改，所以 A/S00 是共同 wrapper 的冻结版本，不能写成“逐算子不变的原始 TM”。R0 与四组结果分开列。

## 4. S24 → 共享场景 → 物理采集

### 4.1 参考尺度的首版选择

首版采用 **相对代理锚点**：`X_s` 是 S24 去噪 camera RGB，在模拟参考快门 `t_ref=1/120 s`、单位模拟增益下定义为相对场景。令

$$L_{s,\delta}=2^\delta X_s.$$

这是一个显式构造，不声称恢复拍摄时的绝对辐射。原 exposure/ISO 保留用于来源和分层，不用 `ISO/100` 当模拟增益，不直接除 shutter 后冒称准确光子率。获得原生 RAW、有效增益/黑白电平与设备标定后，才另建 calibrated 路线。

每源固定3个episode：train含δ=0和由数据seed2026、source-family身份确定的两个 `Uniform[-2,+1] EV` 样本；pilot/val/test固定 `{-2,0,+1}`。这是一次确定的有限归档，不是每epoch在线改δ。一个episode的所有候选共用该尺度，禁止逐候选max/percentile归一化；变体不是独立统计样本。AE不接收潜变量δ。后续online增强需另设确定schedule和有界磁盘缓存。

### 4.2 成像阶段、源裁切与有效参考

使用 `post_optics`，默认不额外叠 PSF。首版把denoised图按长边256（pilot）/512（正式默认）面积缩放成**虚拟工作传感器**内容；短边按比例就近取偶数（恰在两偶数中点取较小值），H/W均至少12，记录实际轻微比例误差；极端宽高比不能满足时拒绝而非任意拉伸。target、mask和坐标使用同一已核验变换。这是改变工作采样网格，不是复现S24原生像元/满阱。

在源分辨率保留 `source_clip_indicator`：noisy或denoised任一通道≥0.995；这是保守的端点裁切指示，不等价于可验证的原生CFA饱和标签。另保存privacy mask、降噪残差/纹理改变诊断及源处理说明。主 `valid_reference_mask` 排除privacy、非有限、配准越界和source-clip指示区域；缩放时任一贡献源像素无效则输出无效。梯度仅用两端均有效的像素对，SSIM仅汇总整个窗口支持域有效的中心。AE图像不按此GT mask隐藏内容。低信号和降噪不确定性单独报告，不自动删掉所有暗部。

源裁切区域仍可显示和报告外观指标，但不能计入“未裁切高光恢复/干净辐射”主指标。所有组使用同一固定有效区域，同时报告覆盖率；候选不能通过改变 mask 获利。

### 4.3 独立目标的两种协议

| 协议 | 目标构造 | 用途 |
|---|---|---|
| **E-style 主实验** | 对齐后的 S24 style-0 JPG，一次解码、固定空间变换 | 检验专家外观下采集/渲染的配合 |
| **E-physics 控制** | 对未缩亮的 `H_s` 做固定解析 renderer；目标不随物理候选或 δ 漂移 | 检查光子、噪声、裁切与候选排序 |

对同一源，所有 δ、噪声和采集计划保持同一个期望外观；`camera_reference=2^δ X_s` 则随真实构造亮度改变，用于采集保真。目标保存 hash、几何和 renderer 身份，不从已裁切 EV0 候选生成，不使用训练中的 TM 动态重建 GT。

E-style 必须确认 RAW/denoised/style-0 的对应、裁切和方向。只允许从源对确定一次几何变换，不能对每个候选输出单独配准或调亮。无法可信对齐的样本被排除配对主实验并记原因；若合格量不足，先做 E-physics，不把解析 GT 叫 style-0。可后续增加冻结 style-0 teacher 协议，但它是 teacher 一致性而非专家 GT；teacher 必须可核验为 train-only，另记 checkpoint/预处理 hash。

### 4.4 预览和模拟 RAW

静态场景表示为 `L(u)=L`，不是伪造真实动态视频。历史预览在合法过去时间分别物理采集。先支持可变历史快门/增益：一半 episode 沿用当前固定预览协议，另一半从合法预览动作采样；每帧记录实际生效 `t/g_a/g_d`，预览与最终采集噪声分离。全部读出早于最终采集请求；AE 只能使用这些观测和可用设备状态。

AE缩略图保留现有 `bounded_demosaiced_sensor_linear_rgb`：完整已观测RAW的去马赛克camera RGB、WB/CCM之前、clamp[0,1]，再缩到长边128，不能把TM crop变成AE场景。候选、参考与最终finish才使用scene WB/CCM。来源JSON中以noisy–denoised对或未来拍摄计算的 `noise_stats/snr_stats` 不进入AE；需要噪声特征时从当前预览和固定模型估计。

共享链保持：曝光积分 → CFA → Poisson → 满阱 → 电子读噪 → analog gain → ADC 噪声/black/量化/裁切 →固定去马赛克 → scene WB/CCM。真实曝光补偿和 virtual gain 各在规定位置执行一次。

默认物理配置沿用 10,000e⁻、3e⁻ 读噪、0.5 DN ADC 噪声、12bit/black64、rolling0、readout1ms，状态 `assumed_engineering`。补充噪声/满阱/读出敏感性后仍不能将这些默认值称作 S24 校准参数。RAW–去噪残差只能支持经验一致性诊断，不能单凭其拟合出可靠电子标定。

## 5. Apple 与 Samsung 主链

**Apple**：因果预览 → 选择一个 `(t,g_a)` → 一帧 synthetic Bayer → 固定前端 → 根据真实曝光补偿一次 → Gain/GTM/LTM →固定目标。使用现有 12 个动作，保留等 EV 但不同快门/增益的组合。

**Samsung**：同类因果预览 → 选择三帧计划 → 按真实时序采集 →固定配准/融合 F0 →已归一化辐射 → virtual gain 参与 TM 条件及曲线 →固定目标。使用现有 24 计划，固定帧数、顺序和 100ms 外部窗口；readout 引起的实际快门缩放按 manifest 记录。

静态 Samsung 可以验证不同曝光的信息覆盖、噪声和 virtual gain；不能验证真实鬼影。比较 Apple/Samsung 时报告帧数、总开光、最大快门、延迟和能耗代理，不把三帧优势解释为公平的单帧算法优势。

## 6. 模型、目标与训练顺序

### 6.1 首版网络与固定组件

- AE：复用 `TemporalExposurePolicy`，三帧 CNN/差分/直方图＋生效曝光＋候选物理特征，输出合法计划分数。约 56k 参数；不读取 GT、δ、源降噪图或未来 visibility。
- TM：style-0 原系数预测网络冻结，复用 `ConditionalToneMapper` 的 gain/shoulder/GTM/LTM 小适配器，约 2.7k 训练参数。
- DN/去马赛克/色彩/F0：四组一致、固定。首版沿用双线性前端、当前 F0，不同时更换 DN 或学习融合。原 S24 denoiser 可另作固定后端敏感性，不混入主结果。
- 先保持 FP32；GPU 仅加速网络。Poisson/量化与离散动作无需可导，使用候选期望梯度。全 backbone 微调和学习 F0 放在主 factorial 完成后的独立扩展。

### 6.2 先修目标，再训练

旧 `J=MSE+.2亮度+.05梯度+.02missing` 作为历史诊断保留。新版本不可默改旧实验的 J。

建议 `ObjectiveSpec(version="s24-v3")`：

$$J_3=L_{display}+.2L_{luma}+.05L_{gradient}+\lambda_R L_{capture}+\lambda_C L_{newclip}.$$

`L_display`为有效参考区域的display MSE；luma采用同样有效区域，gradient仅用有效像素对。`L_capture`比较**前色彩的曝光归一化重建**与 `camera_reference`：先对评测用hatL裁到非负，再计算 `|log2((max(hatL,0)+.001)/(L+.001))|`、上限4EV、除以4，在有效像素的三个通道取均值。原有signed重建保留并另报signed bias，不为做log而改变RAW/前端。该项在候选采集后固定，与学习TM无关，可作训练/验证代价，不能传给AE输入。

`L_newclip`在固定、有效的有纹理高亮区计算静态主实验的channelwise新裁切比例。保护区由**未乘δ的线性sRGB参考H_s**亮度高于有效p90、有效相邻梯度高于正梯度p50确定；无正梯度则为空。保留模拟器 `full_well_mask | adc_mask` 的bitpacked RAW标签，按现有去马赛克支持域传播为RGB通道mask。Apple计相应通道裁切；Samsung在相同静态传感器坐标对三帧同通道mask取AND，三通道平均。不把低信号、F0配准拒绝或越界叫裁切，它们独立记录、重建错误进入capture项；动态协议需另定义参考坐标mask。

空保护区报告为null/count0、条件汇总只含有保护区的源；该episode训练时**省略newclip项、该项梯度为零**，使J3有限。没有任何有效reference像素的episode拒绝。此为有纹理高光代理，不是主体语义标签；小点光源另外报告，不一律重罚。

首选 `λ_R=λ_C=0.1`；只在 train/val 诊断上考察 `{0.05,0.1,0.2}` 的有限组合。选择能排除纹理平坦化/严重裁切候选，又保持合理暗噪–高光取舍的最小权重；找不到合格组合则停止扩大数据并修代价。目标、权重、mask、阈值在正式 test 前锁定，warm-start、训练、选模和报告使用同一代价。

物理项不会直接给 TM 梯度，这是设计意图；TM 从外观项学习，AE 从采集与成片共同反馈学习。降低 MSE、低输出方差或高策略熵各自都不能证明采集信息更好。

### 6.3 训练阶段

1. 完成 R0/R1、来源审计及生成器检查；确定源/目标可用覆盖。
2. 对 train/val 逐候选评分，检查最优、最差、等 EV 配对及 flat/clipped 反例；oracle 仅用于监督/诊断。
3. AE warm-start5 epochs，沿用当前 `warmstart_loss`：冻结初始TM代价的最小动作监督，精确并列最小值分配均匀质量，其余为零，不改为温度softmax。验证只选epoch，不优化梯度。首版AE lr `1e-3`，梯度裁剪5。
4. 四组从同一状态开始，学习 TM 的组同样为零初始化适配器，避免把单独 TM 预训练收益只给联合组。正式最多20 epochs，TM lr `3e-4`，AE lr `1e-3`；val 选最优，允许 epoch0。
5. AE/joint以 `sum_k p_k J_k` 训练；同一episode的两个训练噪声风险先平均，再按有效batch完成一次更新，四组相同；首版计算完整工作画布风险。动态重评候选质量基于当前TM，不能永久使用初始oracle代替联合反馈。
6. 选定 checkpoint 后，测试只用观测选择一次计划，重采/读取该计划，再渲染；不在 test 用最优候选代替 AE。独立测试 noise seeds 用 `100,101,102,103`，train 为 `0,1`、val 为 `10,11`；所有组使用同一对照噪声。

首次GPU profile建议batch2、candidate chunk2、workers2；按相同工作H/W做shape bucket，少量稀有形状可batch1并累积到相同有效batch。不要padding候选画布后改变全局统计；AE保持其既有内部缩放行为。按实测显存调整，保持有效batch与优化步数可比。当前环境为CPU，未取得训练GPU，因此不预估正式训练耗时。

## 7. 各四组、主比较与公平性

| 组 | Apple | Samsung |
|---|---|---|
| 00 | 同库规则 AE＋冻结 TM wrapper | 同库规则 bracket＋冻结 F0/TM wrapper |
| 10 | 学习 AE＋冻结 TM wrapper | 学习 bracket＋冻结 F0/TM wrapper |
| 01 | 同一个规则 AE＋学习 TM | 同一个规则 bracket＋冻结 F0、学习 virtual gain/TM |
| 11 | 学习 AE＋学习 TM | 学习 bracket＋冻结 F0、学习 virtual gain/TM |

冻结规则仍随场景变化。当前 Samsung 规则只搜索固定 ±2EV 子库；新主实验必须使用与学习策略相同的 24 计划。新增传统同库规则仅根据历史观测估计裁切、归一化噪声及运动/快门代价；不读取未来候选或 GT。旧规则可保留作额外参考。

共用源划分、工作分辨率、动作库、预览、噪声、目标、mask、初始化、数据顺序、更新和选模规则。记录 warm-start 成本、候选渲染次数、训练时间、显存及实际采集预算。更新次数相等不等于算力相等；另做一次匹配渲染次数的 01/11 敏感性对照，不能在主表中假装已匹配 FLOPs。

主比较为各方案的 **11−01**；外观主指标为固定有效区域的逐源PSNR差，并同时报display MSE差；J3只作共同优化/选模代价。其次11−10、10−00、01−00。沿用现有正向协同约定，`interaction_synergy=J10+J01−J00−J11`，正值支持此代价下超加性，joint-minus-*负值为更好。正式seeds0/1/2，共24运行条目；18个包含学习、6个冻结基线条目，不称作24个新训练模型。基线重复不得作为独立训练种子的额外证据。

## 8. 分阶段规模和验收

| 阶段 | 数据/训练 | 必须产物与退出条件 |
|---|---|---|
| G0 软件验证 | 常量/边缘/脉冲；解析12场景32²，两噪声 | 单位、积分、满阱/ADC、GT/因果性、RAW 重算通过；本轮已有记录 |
| G1 S24 资格 | 全量目录审计＋train/val 代表样本 | 完整配对/位深/颜色/方向、源族划分、参考覆盖和目标对齐清单；未运行 |
| G2 S24 小生成 | train8＋val4，长边256，δ三档、两噪声 | 同源两方案共同 GT、逐候选物理检查、不同快门/增益差异；未运行 |
| G3 静态 pilot | 最多64/16/16，长边256；seed0，warm1+train3 | 八组端到端、checkpoint恢复、合理 oracle、无裁切伪优解、时间/IO/显存profile；未运行 |
| G4 正式静态 | 合格官方 train/val/test，长边512；3seeds，warm5+train20 | 预先锁定协议、源族配对 CI、完整区域/动作/质量报告；未运行 |
| G5 动态/设备扩展 | 合成全局运动→真实线性HDR序列→标定手机RAW | 逐层支持运动/HDR/真机主张；不能用静态 G4 替代 |

G2/G3 失败先定位域、目标或代价，不增加 epochs 掩盖失败。pilot 通过后依据实际吞吐和磁盘占用决定一次生成全部源或分批；不先预存全分辨率所有候选/缓存。

### G0/G2 的六类检查

| 检查 | 具体预期 |
|---|---|
| 辐射一致性 | 无噪声非饱和常量，除真实 t/g 后差异仅量化/数值误差；不得候选独立归一化 |
| shutter/gain | 等 EV 对中长快门有更多光子，analog 不增加光子；Monte Carlo 均值/方差与声明模型匹配 |
| 运动积分 | 解析匀速边缘随快门增加拖影；静态 S24 本身不提供这个真值 |
| 饱和位置 | 闪亮点/脉冲先积分后满阱，与错误先裁切形成可观察差别 |
| virtual gain | 固定 RAW/F0 只改 v，改变目标 PDF/曲线/亮度，RAW hash 与采集统计不变；只施加一次 |
| GT/因果性 | 改动作/噪声时 target hash 不变；所有历史 readout早于最终请求；改变 TM crop 不改变 AE 输入 |

S24 噪声域诊断单列：按 ISO/曝光/亮度桶比较原 noisy−denoised 残差与模拟 residual 的偏差、方差–均值趋势、空间相关和色彩相关。这里只验经验接近性，不把 Lightroom 残差当独立高斯白噪声或电子标定。

## 9. 评测、统计与可以证伪的判断

主评测不做事后曝光配平。记录全图/固定有效区域 PSNR、SSIM、亮度 MAE、纹理/梯度误差、暗部 bias/随机方差、有效高光误差、源/新采集裁切、前色彩辐射误差；所有区域附像素数和覆盖率。没有区域时为 `null`，而不是零。LPIPS/色差可补充，不能代替采集保真。

动作诊断包括 shutter/gain/bracket 分布、相同源 δ 改变的响应、相同 EV 动作比较、同当前 TM 的 oracle regret、训练集选出的 best-fixed-action 对照。若所有动作本来就相近或同一动作确为最优，策略集中不自动是失败；若 oracle 明显随场景变化而策略固定，则报告适应性失效。

在每训练seed的 `source_family_id` 内平均噪声、δ、crop，再计算该源族配对差；然后跨三个seed平均同一源族的差，作源族配对bootstrap（2,000次、95%）。另给每seed独立结果，注明主CI条件于这些训练种子；不能把seed×源族行拼成独立样本。只有3种子不夸大训练方差精度。关键场景/光照子集与无效参考覆盖同时报告。

首版预声明“联合提高外观”的门槛：held-out `PSNR11−PSNR01` 均值≥0.2dB且95%源族CI下界>0；有效高光MSE差的CI上界≤1e-4、有效梯度MAE差≤.002、归一capture误差差≤.01。非劣界使用同样配对CI，附各指标有效源数；这是工程研究门槛，不声称是人类JND。允许在G3 train/val上修订并发布新协议，禁止基于test修改。J3降低本身不能替代外观门槛。若只优于00、与01相当，或门槛/覆盖不满足，只报告对应有限、中性或负结果。

## 10. 关键消融与扩展顺序

主八组锁定后依次做，不在首轮同时引入：

1. **外部曝光状态价值**：同网络/参数/预览/动作，把 t/g 状态置零对照；可变历史曝光使亮度–曝光歧义真实存在。当前静态相同曝光历史不足以证明时序或 metadata 收益。
2. **代价**：历史J vs J3；移除 capture/newclip 项，检查过曝、纹理和 oracle变化。
3. **目标**：E-physics vs E-style；各自独立表，不混平均。
4. **Samsung bracket**：两边同12个固定形状计划 vs 两边同24个全库；分离库扩展与学习价值。
5. **virtual gain**：固定 RAW/F0，冻结 v 或去掉其曲线条件，保持参数/训练预算可比；不能用改变曝光替代。
6. **传感器敏感性**：假设满阱、读噪、ADC和非零 readout 的独立配置，不用 test挑最好参数。默认首轮不作设备标定结论。
7. **运动**：从同一静态源做已知全局轨迹，声明 synthetic-motion；参考为固定t=0，静态外观做同一几何 warp。真正遮挡/非刚体/光变须另采 HDR动态源、visibility及参考，另设结果表。

## 11. 规模、存储与需补代码

当前候选归档约 `(19+9F)HW` bytes/候选/噪声；准备缓存为 `48HW+4632` bytes，加 missing 约`3HW`。在仅作示例的4000×3000图、两噪声、两方案下，每源合计约79GB。**这不是核验过的 S24 实际尺寸**，但证明全量复制不可取；candidate chunk只控制激活内存，不能解决整包读取。

首版新归档显式version3，支持per-scene颜色/target、独立preview尺寸及各split不同repeat列表；旧v2保留桥接。候选逐个分片，12bit RAW用uint16无损保存。F0的**观测估计**variance/reliability可从RAW及固定参数重算；latent expected_e产生的 `raw_noise_variance` 不能从RAW恢复，需源场景才能重算或作为可选诊断保存。模拟器full-well/ADC裁切标签以bitpacked保留供J3；不能从DN端点冒充精确full-well标签。

RAW本身也需预算：两方案每episode约 `168RHW` bytes，在示例512×384、R=2时约66MB；3,224源×3episode约639GB（未含4-repeat测试额外量和其它字段、未压缩）。优先实测小包后定磁盘配额；空间不足时使用固定scene/协议/seed的lazy-seeded再生成后端和有界磁盘LRU，验证与materialized RAW逐值一致，不改变episode或噪声。prepared只保留有限GPU/LRU条目，严禁重新concat全库。key包括源族/episode/计划/噪声及传感器/色彩/GT/F0/TM/代码身份。

训练先对完整工作画布计算原全局预测器、统计和LTM，避免把裁块当独立场景。首版不需要重写原全局模型。若再上原生大图/crop训练，另加完整图context、全局Bayer相位、rolling行坐标、halo、GT/mask变换与场景级风险聚合；不能直接在每个crop独立预测gain并当同一模型。

| 优先级 | 新/修改文件 | 交付 |
|---|---|---|
| P0 | 新 `s24_sources.py`、`s24_cli.py` | 数据审计、正确PNG16解码/配对、官方split/源族、S24导入 |
| P0 | 新 `scene_reference.py`；改 types/data/pipeline/capture_plan/joint_algorithm | scene WB/CCM、固定style-0/解析目标、mask、前色彩reference、推理色彩参数 |
| P0 | 新 `s24_diagnostics.py` | 六类生成器检查＋S24域/几何/残差/候选排序报告 |
| P0 | 改 `joint_objective.py`、`capture_plan.py`、`pipeline.py` | 版本化J3、同库观测规则、可变因果preview协议 |
| P1 | 新 `candidate_store.py`、`joint_dataset.py`；改 pipeline/validation | 候选lazy分片、有限缓存、原子完成/恢复、v2兼容 |
| P1 | 改 `joint_experiment.py`、`joint_cli.py` | FP32 GPU/batch、统一schedule、resume、渲染计数、选模协议 |
| P1 | 新 `s24_evaluation.py` | 区域count/null、源族CI、oracle/fixed-action、动作响应与图像面板 |
| P1 | 新 `configs/capture_tm/s24_*.json` | 资格/pilot/正式锁定配置、环境/数据/checkpoint manifest |
| P2 | 新 multiscale/synthetic-motion 或真实动态导入扩展 | 原生crop、合成轨迹、真实HDR时序、visibility；独立验收 |

精确接口和任务测试见实施计划。现有 `pipeline_cli demo/validate` 可运行；计划中的 S24 命令要等相应模块实现后才可运行，不能把新 CLI 示例当完成证据。

## 12. 实验产物与启动所需输入

产物分为源审计/拒绝表、冻结配置、候选归档索引、验证JSON、训练与选模日志、checkpoint及恢复推理、逐源逐噪声指标、CI/动作/质量面板、环境依赖与完整命令。研究数据量大，Git记录代码、配置、索引和授权可分享的诊断样例；大数据和权重使用可复核的归档/下载索引。

开始实际 S24 G1/G2 需要：可访问的数据根目录（含train/val/test）或至少每split几组完整 PNG16/JSON/style-0/mask 样例。正式 G4 还需确认 CUDA 设备和可用磁盘。当前环境找不到这份下载数据、PyTorch为CPU版；因此本轮完成的是方案与现有模拟管线的软件验证，没有声称生成 S24 数据或训练新 S24 模型。

## 来源

- [S24 作者数据说明](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md)：格式、字段、目标与来源边界。
- [Time-Aware AWB (ICCV 2025)](https://arxiv.org/abs/2504.05623)：S24来源；本文以已取得的作者README核验格式，不冒称复现其AWB实验。
- [Modular Neural ISP](https://arxiv.org/abs/2512.08564) 与本仓库 `datasets/README.md`、`photofinishing/dataset.py`、`utils/img_utils.py`：原模型及实际输入域。
- 本仓库 [JOINT_RESULTS_REVIEW.md](JOINT_RESULTS_REVIEW.md)、[DATA_SYNTHESIS_DESIGN.md](DATA_SYNTHESIS_DESIGN.md)：既有失败分析、物理合同和待补证据。
