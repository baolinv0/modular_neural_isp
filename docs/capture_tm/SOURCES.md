# C 实现来源映射

核对日期：2026-10-05。下面区分既有机制和本实现的工程选择；保留原仓库作者、模型与 LICENSE。

新版单帧/HDR 联合算法见 [JOINT_ALGORITHMS.md](JOINT_ALGORITHMS.md)。相较下表记录的旧 C1，新增三帧时序 CNN、完整 HDR 曝光计划、固定观测驱动融合，以及原 TM 上的条件 adapter 与离散期望联合训练。Samsung 的虚拟增益由原 GainNet、渲染意图及学习残差共同给出，并真实参与目标 PDF/曲线构造；不再把用户的曝光意图单独当成全部虚拟增益。这个修正依据专利中“由图像统计计算虚拟增益、再调整至目标亮度”的描述，仍属于神经网络改写而非传统公式逐项复现。

| 来源 | 固定版本/原始资料 | 本实现使用的部分 |
|---|---|---|
| Modular Neural ISP | [baolinv0 fork](https://github.com/baolinv0/modular_neural_isp/tree/e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3)；[Afifi et al. paper](https://arxiv.org/abs/2512.08564) | 原始 PhotofinishingModule 和原 style-0 权重，新增 frozen adapter；未复制别的分支未核验实现 |
| Apple | [US9432647B2](https://patents.google.com/patent/US9432647B2) | 采集 AE 偏置与 DRC 补偿的协作、保留渲染意图；固定 shoulder 是本实现的 HDR 扩展，非全权利要求复现 |
| Samsung | [US12243201B2](https://patents.google.com/patent/US12243201B2/en)，[公开申请 US20230114798A1](https://patents.google.com/patent/US20230114798A1/en) | virtual gain 改变 histogram statistics，经 LPF/CDF 映射构造 tone curve；推前分布 Z=clip(GY) 消除原印刷式 cutoff 歧义，anchor/退化处理为工程选择 |
| AdaptiveAE | [论文 v1](https://arxiv.org/html/2508.13503v1)，[作者公开仓库](https://github.com/OpenImagingLab/AdaptiveAE) | 中间图像、三图 histogram、阶段观测与 blur/noise 合成思路。原作 A3C、HDR fusion/多阶段；本实现小 CNN 与监督候选 scorer，未使用不存在的公开训练权重 |
| RL-3A | [baolinv0/RL-3A commit](https://github.com/baolinv0/RL-3A/commit/7b7e031969522dd7e96836bb2507c5c87a4e0b5a) | 光子/ISO 分离、relative proxy、固定 ISP、source group 变体；私有仓库通过已连接 GitHub 读取，未随本 PR 搬运它的数据或代码 |
| RawGen | [Kim et al., 2604.00093](https://arxiv.org/abs/2604.00093)，[作者项目](https://dy112.github.io/rawgen-page/)，[SamsungLabs commit](https://github.com/SamsungLabs/RawGen/tree/6ca22da610891f9c16d0327d50edeac32f38c988) | 默认 XYZ 输出的显式 inverse OETF 与 camera mapping；本实现是输出导入器，不是训练/运行其生成网络 |
| HDR/noise capture | [Hasinoff et al., Noise-optimal capture for high dynamic range photography, CVPR 2010](https://people.csail.mit.edu/hasinoff/pubs/hasinoff-hdrnoise-2010.pdf) | 保持 exposure/光子与 electronic gain 的作用分离；噪声模型常数是工程示例，不冒充论文/手机标定数值 |

RL-3A SHA 经 `branches/main` 确认为 commit SHA（tree SHA 为 `bd40024b1e18ee553b7c57b688043c8ddcd605be`）。相关读取契约：`rl3a/physics/exposure.py`、`physics/noise.py`、`radiance/exposure_inversion.py`、`motion/temporal_integration.py`、`data/joint_reset_dataset.py`、`data/s24_dataset.py`、`isp/fixed_isp.py` 与 `artifacts/noise/iso_shot_read_v1.json`。

RL-3A natural noise fit 基于自然图像对 denoised reference 的残差，既不是 electron/full-well 标定，也不确定 analog/digital 分解。RawGen 默认 XYZ PNG 是 OETF 编码 XYZ，9cam 输出是成片，单帧无真实快门轨迹；二者均不能替代真实线性 HDR motion/RAW 采集标定。

不把上述现有机制、公开网络观测或参考物理模型包装成论文创新。若将 C 发展为研究 topic，需证明“给定同样成片 intent 和合法采集预算，特定 renderer 下的采集策略”相对同容量 image-only、物理优化与其它 TM-aware AE 的真实优势，且区分单帧/多帧、RAW/代理域、跨相机泛化和评估器耦合。
