**五种 Tone Mapping 候选方案：研究动机、实现机制、参考依据与实验判别**

版本：2026-10-17；用途：方案讨论、原型定义与实验筛选。本文不预选最终架构，也不声称已完成性能验证或专利新颖性判断。

**1．共同研究问题与项目定位**

核心问题是：在有限计算、带宽和调参成本下，如何让不同对象、不同受光区域获得合适的亮度与层次，同时维持整图光照关系、肤色、纹理和稳定性？

按照附件 Definition.txt，本项目从 iPhone 成对数据训练 Modular Neural ISP baseline，再应用到 S24；跨相机后出现亮度、色调和整体观感差异。同场景 iPhone 图像提供风格参考，但不保证像素对齐。因此，须区分三个问题：输入相机颜色/曝光校准是否正确、当前渲染算子是否能表达目标、预测器是否能选出正确参数。替换 TM 架构不会自动解决全部跨相机误差。

共同 motivation：

- 同一输入亮度可能属于人脸、天空或背景阴影，期望输出不同；单个全局映射可能难以同时满足。Google Live HDR+ 对局部曲线近似的解释提供了直接实例。[R2]
- 低分辨率分析负责决定参数，高分辨率显式算子负责执行，有机会降低大图神经处理的成本；HDRNet、Live HDR+ 和 LTMNet 均有相关机制，但三者结构并不相同。[R2、R5、R6]
- 语义可以为区域优先级提供条件，但“它是什么”不能单独决定“改多少”；还需要区域亮度、受光、噪声、置信度和目标风格。Apple 的公开材料支持语义参与局部渲染，但没有披露完整后端算子。[R1]
- 对现有 baseline，既需要能低风险修正局部偏差的模块，也需要检验主体 LTM 算子是否限制画质的对照路线。

五方案比较的是不同的表示方式和约束。A、C 在当前限制自由度的版本中更适合局部控制或修正；B、D、E 更适合作为主体 TM 候选。A、C 并非数学上不能独立完成 TM，只是不应在当前实验中被默认赋予任意表达能力。

| 方案 | 核心假设 | 网络预测内容 | 高分辨率执行 | 主要动机 |
|---|---|---|---|---|
| A 区域曲线混合 | 少量共享策略能覆盖主要区域差异 | K 条曲线、区域混合权重/修正 | 共享曲线求值后加权 | 保持解释性，以少量策略处理对象差异 |
| B 空间参数场 | 需要连续、随位置变化的局部映射 | 网格曲线或其他受限参数 | 邻近网格插值并执行 | 表达同一对象内部复杂照明 |
| C 基础映射＋局部增益残差 | baseline 大体正确，剩余偏差较小且平滑 | 低分辨率有界 log-gain | 基础结果乘局部增益 | 保留已有能力，控制修改幅度 |
| D 合成曝光多尺度融合 | 局部外观可从多个亮度候选中选择 | 候选曝光的融合权重 | 固定金字塔融合与重建 | 用多尺度选择保留局部层次 |
| E 基础层／细节层处理 | 动态范围与细节幅度应分别控制 | 基础层曲线、细节增益 | 分解、调整、重建 | 避免整体压缩连带损失质感 |

下面的公式是本项目原型定义，不是对厂商内部系统的复刻。

**2．A：区域曲线混合（Region-conditioned Curve Mixture）**

**Motivation。** 全图同一曲线难以兼顾逆光人脸与明亮背景；逐像素任意预测又容易增加自由度和调参困难。A 假设主要差异可由少量处理策略表达，区域信息只负责选择、混合和调节这些策略。

**拟实现方式。** 小图图像特征、区域亮度统计和可选语义条件预测 K 条单调亮度曲线；高分辨率执行：

\[
Y_A(p)=\sum_{k=1}^{K}w_k(p)T_k(Y(p)),\qquad w_k(p)\ge0,\quad\sum_kw_k(p)=1.
\]

共享指同一张图内共享，曲线可以逐图变化。曲线与语义类别不必一一对应：同为人脸，顺光与逆光可能使用不同混合；多人场景还需要实例统计。权重采用低分辨率软图与统一上采样，另设基础处理通路用于不可靠区域回退。

**参考链。** NamedCurves 从颜色分组产生曲线调整结果，再融合出空间编辑效果，为“少量曲线＋软分组融合”提供近似结构依据。[R3] 它使用颜色名称、RGB 曲线、前端图像网络及 attention；颜色分组不等于人脸/天空语义，整套方法也不等于上述亮度公式。OPPO CN115330633A 先融合类别映射参数生成分块曲线，再对邻近曲线输出做空间插值，更准确地位于 A/B 的交界。[R4] Apple 公开的人物、皮肤、天空 masks 支持选择性调整的产品动机，不证明其采用 A 的公式。[R1]

**本项目增加的设计。** 对象/实例条件、区域受光统计、单调亮度曲线、置信度回退和有限曲线预算；这些需要自己的实验支持。

**潜在优势。** 策略直观，容易检查“哪个区域用了什么曲线”；适合主体与背景关系、不同人物的亮度取舍。

**主要风险。** 少量策略可能无法描述同一对象内部的复杂光照；软权重边界错误会产生亮带。每条曲线单调不代表整个空间算子无 halo 或保持任意两像素的亮度排序。

**最小判别实验。** 固定曲线参数化、权重分辨率和预测器，比较不同 K；对比类别权重、加入区域统计、加入实例统计。观察增大 K 是否仍有稳定收益，以及误差是否集中在对象内部而非对象之间。

**3．B：空间参数场（Spatial Tone-Parameter Field）**

**Motivation。** 语义标签不能完全解释光照：同一人脸可以一侧受光、一侧背光，同一面墙可能同时接近窗户与暗角。B 允许处理参数随位置连续变化，避免少量区域策略成为表达瓶颈。

**拟实现方式。** 小图网络预测 G×G 网格，每格给出单调亮度曲线参数 \(\theta_{ij}\)。第一版采用明确的曲线输出插值：

\[
Y_B(p)=\sum_{(i,j)\in\mathcal N(p)}\alpha_{ij}(p)T(Y(p);\theta_{ij}).
\]

\(\mathcal N(p)\) 为邻近网格；\(\alpha\) 为归一化空间权重。不要在非线性参数化下默认“先插值参数再求值”与“先求值再插值”等价。语义可进入参数预测器，但不改变首轮固定执行器。

**参考链。** Live HDR+ 明确描述小图预测分块曲线、对大图像素应用邻近四块曲线并混合，用于实时近似成片 HDR+。[R2] HDRNet 原论文使用双边空间的局部仿射系数与 slicing，是更广义的参数场，不能等同于纯亮度曲线网格。[R5] LTMNet 直接学习 tone-curve grid，适合作为 B 的近邻基线，并提供官方代码。[R6]

**本项目增加的设计。** 在统一 tone-only 接口下，研究语义/区域统计能否提升网格预测，以及空间平滑和边缘条件如何影响泛化与伪影。并非提出“小图预测＋大图执行”这一已有范式。

**潜在优势。** 能描述对象内部照明变化；网格分辨率可调，便于探索表达能力与成本的关系。

**主要风险。** 低分辨率网格跨越边界时易泄漏，高分辨率网格增加参数量并可能过拟合或闪烁。若开放 RGB affine，颜色自由度也变化，不能与纯亮度版本混成同一实验。

**最小判别实验。** 固定每格曲线容量，扫网格密度；比较固定空间插值与边缘引导插值时，将后者单独标为升级版本。结合逐图参数拟合，区分网格容量不足与预测器失效。

**4．C：基础映射＋局部增益残差（Base Mapping with Bounded Local Gain）**

**Motivation。** 当前 baseline 往往已有可用的全局影调和细节，主要问题可能是人物局部偏暗或背景关系失衡。C 假设无需重学整套映射，只需用小幅、平滑的修正补齐剩余偏差，降低训练样本需求和改坏已有区域的风险；这些收益是待验证假设。

**拟实现方式。** 第一阶段冻结基础映射 \(T_0\)，网络预测低分辨率 log-gain，经固定上采样得到 \(\delta(p)\)：

\[
Y_C(p)=T_0(Y(p))2^{\delta(p)},\qquad |\delta(p)|\le d_{\max}.
\]

限制增益范围与空间带宽，并从 \(\delta=0\) 初始化；输出裁剪前保留浮点处理空间。若作用域是经非线性编码的图像，\(\delta\) 应称 log-gain，不能直接解释为真实拍摄 EV。即使作用于显示线性亮度，也只是数字亮度倍率，不能等同于改变传感器曝光。

**参考链。** Samsung US20260030731A1 以直方图匹配与正则化建立全局 LUT，再对低分辨率剩余局部差异构建增益图，上采样、去噪并应用到大图；这是与 C 很接近的结构依据。[R7] 原文不是我们拟用的语义神经残差预测器，也不等于已规定上述显式上下界。Apple ProRAW 同时提供局部明暗调整的 ProfileGainTableMap 和输出全局曲线 ProfileToneCurve，为两类控制共存提供接口证据，但不能由此反推 C 的乘法公式或内部执行顺序。[R8]

**潜在优势。** 容易与现有 baseline 对接，关闭残差可回退；改变幅度可控，适合检验“跨相机剩余误差是否主要是低频影调差异”。

**主要风险。** 平滑增益无法充分改变局部曲线形状；过强增益会放大噪声、造成色域溢出和边界光晕。已被裁剪掉的细节不能由残差恢复。若必须大量放宽增益范围与分辨率才有效，C 的小修正假设可能不成立。

**最小判别实验。** 固定 T0，比较全局调整、低频残差、有语义残差；扫增益上限与空间分辨率。记录修改区域外的退化、近黑噪声、裁剪前越界率，检验残差是否仍然小而平滑。

**5．D：合成曝光多尺度融合（Synthetic-Exposure Multi-scale Fusion）**

**Motivation。** 一张数字偏亮的候选适合呈现暗部，偏暗候选适合显示高光；不同空间尺度对“亮度关系”和“纹理层次”的需求也不同。D 将直接预测复杂映射改为在固定候选外观之间学习选择，并借助多尺度融合控制过渡。

**拟实现方式。** 从同一未提前裁剪的输入构造 J 张确定性合成曝光 \(E_j=S_j(Y)\)，固定曝光倍率、映射规则和金字塔。网络只预测权重：

\[
P_\ell^{out}=\sum_j\widetilde W_{j,\ell}\,\mathcal L_\ell(E_j),
\qquad
Y_D=\operatorname{Collapse}(\{P_\ell^{out}\}_\ell).
\]

\(\mathcal L\) 为 Laplacian 金字塔，权重可从预测权重的 Gaussian 金字塔获得，并逐尺度归一化。首轮不同时学习候选曝光、分解和重建，以便明确收益来源。

**参考链。** Hasinoff 等人的 HDR+ 论文从融合 HDR 构造合成曝光，并通过金字塔曝光融合完成局部影调处理，是 D 的经典依据。[R9] 其真实采集阶段的 burst 与后端合成曝光是不同步骤；我们的 D 不要求重新拍摄多曝光。Samsung US20240257324A1 从融合 HDR 生成多张 LDR，以语义及 ISO 等信息修正融合权重；具体示例使用 base/detail 两层融合，不应写成其直接采用 Laplacian 金字塔。[R10]

**本项目增加的设计。** 固定合成候选，采用语义/噪声条件的权重预测器；将 Google 的多尺度思想与另一来源的语义调权思想组合成可比较原型，而非把组合归给某篇文献。

**潜在优势。** 自然地表达“这里偏向亮候选、那里偏向暗候选”，可在不同尺度分配处理；是检验局部层次是否受纯曲线方案限制的重要对照。

**主要风险。** 多张候选及多尺度中间图增加读写与缓存需求；候选设计限制输出空间；亮候选可能放大噪声。频带包含正负值，重建后仍可能越界或产生光晕。合成曝光不会创造输入已经丢失的观测信息。

**最小判别实验。** 单尺度 vs 多尺度；固定规则权重 vs 学习权重；无语义 vs 有语义。若仅单尺度学习权重有效，不能把收益归因于多尺度。记录完整渲染耗时、峰值内存与大图读写，而不只报告小图预测网络 FLOPs。

**6．E：基础层／细节层分别处理（Base–Detail Decoupled Rendering）**

**Motivation。** 动态范围压缩与质感保持可能发生冲突：脸亮了却变平，背景压暗后纹理变弱，提高局部反差又放大噪声。E 显式分开缓慢变化的亮度结构与细节残差，使两者接受不同约束。

**拟实现方式。** 对统一亮度域进行固定边缘保持分解，以 log2 域为例：

\[
\ell=\log_2(Y+\epsilon),\quad b=\operatorname{EPF}(\ell),\quad d=\ell-b,
\]
\[
\ell_E'=f(b;\theta_b)+a(p)d,
\qquad Y_E=\max(2^{\ell_E'}-\epsilon,0).
\]

固定分解算子、参数及黑位处理。网络预测基础层映射与受限非负细节增益；语义、区域统计与噪声条件参与参数选择。将 a=1 作为重要对照，先验证是否真的需要改变细节。

**参考链。** Durand 与 Dorsey 在 2002 年提出双边分解、压缩 base 并保留 detail 的 HDR 显示方法。[R11] MediaTek US20250054119A1 进一步明确语义参与分解、区域统计、曲线压缩和细节调整；因此“语义＋base/detail”已有直接披露。[R12] 本项目先固定分解，仅学习控制参数，是为降低原型自由度而作的实现选择。

**潜在优势。** 能把亮度骨架与质感变化分开分析，适合验证“主要短板是否来自压缩与细节耦合”。

**主要风险。** detail 不等于纯净纹理，也含噪声与边界残差；简单放大会产生假质感和过冲。边缘保持不意味着没有 halo；分解尺度变化会改变 base/detail 的含义。大图滤波成本需要单独测量。

**最小判别实验。** 只处理 base、base＋统一 detail gain、base＋语义/噪声条件 detail gain；并固定或分别标明分解尺度。重点检查面部明暗塑形、肤质、平坦区噪声和高反差边缘。

**7．五条路线的关系：比较约束，不把名称当作互斥集合**

| 关系 | 成立条件或说明 | 实验含义 |
|---|---|---|
| B 可写成 A 的形式 | 将每格曲线作为一条 Tk、空间插值作为权重 | A 固定少量共享策略；B 固定空间网格，不能无限扩容后仍宣称本质不同 |
| 单尺度 D 是 A 的特例 | 候选为点对点曲线、权重非负且归一化 | D 的主要区分在多尺度融合与固定候选族 |
| 纯线性单尺度 D 也是增益模型 | 候选仅为未裁剪的线性倍率 | 不应把等价算子包装成独立证据 |
| E 可代数改写成增益 | log 域增益为 f(b)-b+(a-1)d | 其中可能有高频项，通常不满足 C 的低频有界约束 |
| A 可覆盖部分 C | 两条曲线包围基础映射的允许增益范围，权重足够自由 | 真正差异是增益/权重的带宽、范围和条件，而非加权或乘法的名字 |

这不削弱实验价值。不同参数化、约束、学习难度和硬件执行成本，即使部分函数可相互表示，也可能产生不同实际结果。

**8．统一实验接口与筛选路线**

**先冻结输入输出。** 对主体算子比较，统一同一阶段的 pre-TM 输入、曝光归一化、白平衡/颜色预处理、亮度定义、输出颜色恢复、色域处理及裁剪位置。C 内部可以含固定 T0；其他候选不能无记录地多获得一段 baseline 处理。如果只比较 baseline 后的插件，应另立 A/C 增量实验，不与从 pre-TM 重建输出的实验直接混算。

按亮度比回乘 RGB 只是在所选域中保持一定比例关系，不自动保证感知色相/饱和度、肤色或色域。首轮采用相同颜色接口以隔离 tone 差异；随后单独开放颜色分支。目标偏色若来自 AWB/CCM，应先定位，而不是要求五个 tone-only 候选代偿。

**将语义作为独立维度。**

| 条件 | 训练时 | 推理时 | 判别目的 |
|---|---|---|---|
| S0 | 常规监督 | 图像及统一元数据 | 隐式内容感知的基准 |
| S1 | 区域监督或语义蒸馏 | 不接显式 mask | 语义收益能否转移到轻量模型 |
| S2 | 使用语义条件训练 | mask、区域统计及置信度 | 显式语义的额外收益是否抵得上成本 |

先用相同、可靠的 mask 做小规模算子筛选，随后在入围者上做 S0/S1/S2；oracle mask 只用于诊断，不能计作真实部署效果。复用已有语义时报告增量成本，独立生成语义时报告完整成本。

**用三类证据确定下一步。**

1. 逐图参数拟合：在真实参数范围、量化、网格与平滑约束下测实用拟合能力。非凸优化未必找到数学上界，因此应称表达能力诊断或经验拟合上限，不能当严格最优。
2. 统一学习比较：共享可比的信息与训练预算，报告预测器容量及各自渲染成本；测试逆光人像、多人、侧光、暗光、混合光、天空/发丝边界。不能只看平均 PSNR。
3. 实际约束验证：软件浮点到硬件表示的误差、时序稳定性、端到端延迟、峰值内存、大图读写与功耗。ISP/NPU 重叠执行不能消除参数依赖，是否支持曲线网格、软权重或两层处理必须由真实接口确认。

| 方案 | 最关键的证伪问题 |
|---|---|
| A | 少量共享策略是否不够，误差是否主要发生在同一对象内部？ |
| B | 增加网格后仍拟合不好，还是网格能拟合而网络预测失败？ |
| C | 达到目标是否必须使用大幅、高频修正，从而违反小残差假设？ |
| D | 多尺度融合是否有独立收益，还是收益只来自新增候选或更大成本？ |
| E | 分开处理细节是否改善质感而不放大噪声，还是固定 detail 已足够？ |

**评价和数据边界。** 非对齐 iPhone/S24 参考用于匹配语义区域后的影调、颜色统计和盲评，不能直接计算逐像素 PSNR/SSIM。可靠对齐目标或经过验证的伪 GT 可用于像素监督，但仍需记录生成偏差。附件 CONTRACT.md 的单场景探针只验证候选目标可用性，不能据此宣称 TM 训练有效或跨场景泛化。不要因伪 GT 的确定性投影使用某一种曲线表示，就只在该投影目标上选择同类算子，否则会有表示偏置。

第一阶段应交付五个清晰受限的原型、统一结果表及失败归因；第二阶段选择两条有竞争力的路线进行语义、颜色与部署验证。最终可选单一方案或有证据支持的组合，目前无需预先确定。

**9．参考文献与证据定位**

以下分别标注论文、官方工程介绍与专利公开文本。专利证明相关机制曾公开，不证明对应产品已采用，也不构成自由实施或法律状态意见。

| 编号 | 文献信息 | 类型与关键定位 |
|---|---|---|
| R1 | Apple. *On-device Panoptic Segmentation for Camera Using Transformers*. 2021. [官方全文](https://machinelearning.apple.com/research/panoptic-segmentation) | 官方技术介绍；人物/皮肤/天空语义与选择性渲染。没有公开完整后端 TM |
| R2 | Jiawen Chen, Sam Hasinoff. *Live HDR+ and Dual Exposure Controls on Pixel 4 and 4a*. Google Research, 2020-08-03. [官方全文](https://research.google/blog/live-hdr-and-dual-exposure-controls-on-pixel-4-and-4a/) | 官方工程介绍；Local Curve Approximation 一节；小图曲线预测、邻近四块输出混合 |
| R3 | David Serrano-Lozano, Luis Herranz, Michael S. Brown, Javier Vazquez-Corral. *NamedCurves: Learned Image Enhancement via Color Naming*. ECCV 2024; arXiv:2407.09892. [全文](https://arxiv.org/html/2407.09892v1) | 论文；§3.2–3.4，颜色分组、曲线和融合。颜色类别不等于对象语义 |
| R4 | OPPO 广东移动通信有限公司. 《图像色调映射方法及装置、电子设备、存储介质》. **CN115330633A**，2022-11-11 公开. [全文](https://patents.google.com/patent/CN115330633A/en) | 专利；类别参数→分块曲线→相邻输出插值；跨 A/B |
| R5 | Michaël Gharbi, Jiawen Chen, Jonathan T. Barron, Samuel W. Hasinoff, Frédo Durand. *Deep Bilateral Learning for Real-Time Image Enhancement*. ACM TOG 36(4), 118, SIGGRAPH 2017. [项目与论文](https://groups.csail.mit.edu/graphics/hdrnet/) | 论文；双边空间局部仿射系数、低分辨率预测与 slicing。非纯亮度曲线 |
| R6 | Luxi Zhao, Abdelrahman Abdelhamed, Michael S. Brown. *Learning Tone Curves for Local Image Enhancement*. IEEE Access, 2022. [官方论文页](https://research.samsung.com/research-papers/Learning-Tone-Curves-for-Local-Image-Enhancement)；[官方代码](https://github.com/SamsungLabs/ltmnet) | 论文；LTMNet 局部曲线网格。公开研究不等于特定 Galaxy 量产管线 |
| R7 | Samsung Electronics Co., Ltd. *Mapping a low resolution, noisy tone mapping operation onto high resolution images*. **US20260030731A1**，2026-01-29 公开. [全文](https://patents.google.com/patent/US20260030731A1/en) | 专利；全局 LUT＋局部残余增益；非所拟语义残差网络 |
| R8 | Apple. *Capture and process ProRAW images*. WWDC21, session 10160, 2021. [视频与逐字稿](https://developer.apple.com/videos/play/wwdc2021/10160/) | 官方接口资料；ProfileGainTableMap、ProfileToneCurve、语义 masks；不反推内部顺序 |
| R9 | Samuel W. Hasinoff, Dillon Sharlet, Ryan Geiss, Andrew Adams, Jonathan T. Barron, Florian Kainz, Jiawen Chen, Marc Levoy. *Burst photography for high dynamic range and low-light imaging on mobile cameras*. ACM TOG / SIGGRAPH Asia 2016. [官方论文页](https://research.google/pubs/burst-photography-for-high-dynamic-range-and-low-light-imaging-on-mobile-cameras/)；[作者版 PDF](https://people.csail.mit.edu/hasinoff/pubs/HasinoffEtAl16-hdrplus-preprint.pdf) | 论文；Tone mapping 段的合成曝光与曝光融合。不能把后处理候选与采集帧混淆 |
| R10 | Samsung Electronics Co., Ltd. *Machine learning segmentation-based tone mapping in high noise and high dynamic range environments or other environments*. **US20240257324A1**，2024-08-01 公开. [全文](https://patents.google.com/patent/US20240257324A1/en) | 专利；合成 LDR、语义 delta-weight 与 base/detail 融合；不是直接的金字塔证据 |
| R11 | Frédo Durand, Julie Dorsey. *Fast Bilateral Filtering for the Display of High-Dynamic-Range Images*. ACM TOG / SIGGRAPH 2002. [作者项目页](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/)；[论文](https://people.csail.mit.edu/fredo/PUBLI/Siggraph2002/DurandBilateral.pdf) | 论文；双边 base/detail 分离，压缩 base、保留 detail |
| R12 | MediaTek Inc. *HDR Tone Mapping System and Method with Semantic Segmentation*. **US20250054119A1**，2025-02-13 公开. [全文](https://patents.google.com/patent/US20250054119A1/en) | 专利；语义分解、对象统计、曲线压缩、纹理/噪声相关细节调节 |

核验说明：本次读取了附件项目定义和实验范围，并用一手网页、论文或专利文本核对机制与元数据。R9 的整份 PDF 读取受到文件体积/重定向限制，相关细节通过作者托管 PDF 的可检索原文片段及 Google 官方记录交叉核对；未声称完整复现原算法。本报告为定向来源核验，不是覆盖全部相关工作的新颖性检索。
