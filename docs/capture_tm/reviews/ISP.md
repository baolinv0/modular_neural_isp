# ISP / 数据独立只读审查

审查人：独立 ISP/data reviewer；核查时间：2026-10-05T06:18:00.551882+00:00。基线 `e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3`，审查对象为尚未提交的 `capture_tm/` 新实现。没有修改实现或测试，没有派生代理。只写本审查记录；测试输出与临时数据均在临时目录。

## 最终结论

本审查范围内，先前复现的时间/数据身份/标签来源缺陷已修复，独立复核通过。当前实现可以支持**相对线性 RGB sensor proxy 的 capture-domain 软件模拟与离线监督实验**；不能据此报告真机 AE 改善、完整手机 ISP 复现、绝对 radiance / calibrated sensor，或真实 source 高光已恢复。

`types / simulation / data / experiment` 定向测试：**55 passed in 4.31s**。另独立运行 shipped style-0 的 Apple 与 Samsung 两条 oracle 管线，每条 3 个 16×16 synthetic scenes、2 个真实快门候选、1 个 noise repeat，labels 全有限，两条均使用实际 checkpoint `5137b1a9da936814544a0259add95530e124d954fac8e08ece61333be630c09f`。Apple 输出范围 `[0.0131692, 0.9925811]`，Samsung `[0.0109204, 0.9939839]`。这是 CPU software smoke，非画质评测。

## 已复现并最终复核的修复

| 问题 | 初始复现 | 最终独立复核 |
|---|---|---|
| 实际 source 跨 split | 同字节 `a.npy` 复制为 `b.npy`，改 scene/source ID 且无 `input_sha256`，train/test 曾被接受 | 实际 decoded float32 shape/bytes SHA 由系统重算；同源改为 float64 `.npz` 重新打包也报跨 split，调用方 hash 不可替代它 |
| cache dataset 身份忽略实际数组 | `.npy` 原地由 0.1 改 0.3，manifest SHA 相同而 target 变更 | manifest SHA 仍相同，但新 `dataset_sha256` 不同；已纳入 cache/checkpoint compatibility；mask 同样有实际 payload hash |
| 观测窗口时间因果性 | 只比较 centers 时，自定义长观测曝光可跨入 future capture；历史 shutters 也可重叠 | 历史每次 shutter+readout 必须在下一次 shutter 之前结束；`observation_action=.04`, centers `[-.04,-.02,0]` 明确报 overlap；未来候选 `[1/120,1/20]` 的 legal mask 为 `[True,False]` |
| unsigned / bool timestamps 绕过严格递增 | `torch.uint8([255,0])` 的 diff 回绕；`torch.bool([True,False])` 的 diff 为 XOR，均曾被 Scene 接受 | bool/complex 显式拒绝；其他数值先转换 float64 再严格递增；上述两例均 ValueError |
| sensor provenance 不是对象 | `measured_sensor` 配字符串或 list 也曾被接受 | 两例均明确 `sensor provenance must be a dictionary` |
| labels 不能充分追溯 teacher | cache 曾仅存路径和指标，未保存真实 weights identity、source provenance、noise seeds | renderer checkpoint SHA、renderer identity、implementation SHA、source provenance、observation actions、preview/candidate seeds 已保存；request 也携 sensor/renderer 与 policy checkpoint SHA |
| budget 的语义易误读 | max_total=.02 仍可先拍 3 次 preview 加 final capture，总计 .03717s | protocol 明确 `remaining budget for one final capture; prior previews excluded`；request 记录 caller 必须扣除已拍/待拍 costs；这是 post-observation 预算，不是 entire-episode 预算 |
| 真实 source family 分组缺入口 | 不同 raw/denoised/crop 文件不能仅靠 bytes hash 识别原始 scene | CLI 暴露 `--source-id`、explicit layout/timestamps/sensor-profile；仍需 caller 使用 original sample/episode identity 分组 |

## 数值与域审查

- 物理顺序是相对线性 radiance 的一次快门积分 → electron Poisson → full-well clip → 一次 read noise → analog gain → ADC noise / black offset / round / ADC clip → floating digital gain。Analog 与 digital gain 不改变 photon expectation；digital >1 不被 RAW 上限隐藏；read-noise 产生的负 black-subtracted 值保存在 `CaptureResult.rgb`。
- 独立 Monte Carlo：constant signal .1，full-well=1000e、read=3e、ADC=1.2DN、16bit、black=8000DN、512×512 RGB。Gain=1 的实际 variance / analytic proxy = **0.999354**；gain=4 为 **0.999347**。符合 unclipped variance 的工程模型。模型没有 CFA、dark current、rolling shutter、PRNU、row noise、dual-conversion gain 或光谱标定，不能称 native Bayer/hardware RAW。
- 零信号独立复现 black-subtracted capture mean 为 `-2.70e-6`，fixed frontend rectification 后 mean 为 `.0011933`。后者是渲染域非负输入处理，不是校准后的 denoising；不要把它解释为低照度真实噪声已解决。
- RawGen import 明确要求 OETF-encoded XYZ，逆 sRGB OETF 一次后用 caller matrix 映射，保留上端 HDR float，不将九相机 viewing preview 当 AE ground truth。源本身被 bounded generation/storage 限制，matrix 变换产生 >1 不证明恢复了真实高光。
- RL-3A 的 RGB16 PNG 使用 pypng 保留真实 16-bit code 与 RGB channel；provided t / analog / digital inversion 是相对 proxy。其自然图像 residual noise JSON 被明确标为 normalized demosaiced RGB fit，未映射/冒充 calibrated electron profile。
- DNG 使用 optional rawpy、linear gamma、unity WB、camera color space 与明确 black/white normalization，无 embedded preview fallback。用 invalid DNG 独立验证得到：`DNG cannot be decoded into linear camera RGB; compressed ProRAW may require another decoder, no preview fallback`。这只验证 unsupported failure，未验证真实 S20/S25U/ProRAW 解码与 radiometry。
- `_frontend` 只有 fixed WB/CCM+nonnegative render preparation。`PhotofinishingModule` 的训练输入是 linear sRGB；实际 camera RGB 必须提供对应 camera-to-linear-sRGB WB/CCM。默认 identity 是工程 proxy 假设；CLI 现在可传 sensor profile。冻结 style-0 确实使用现成权重，但仍包含其 learned photofinishing 决策，不等于 full phone ISP 或 denoiser/detail enhancer 全链条。
- Teacher 所有候选共用同一 future midpoint 的 sharp latent target 与同一 render intent，cost 包含 display reconstruction、subject luma、模拟 RAW saturation、rendered-noise MSE 和 noiseless-capture bias。此处 noise MSE 是以 noiseless candidate 为中心的 surrogate，并非 sensor variance estimator；sharp latent/proxy 与 clean candidate 只进入 offline teacher，policy 不接收它们。

## 剩余边界与非阻断建议

1. `raw_saturation` 只度量**本次模拟新发生**的 full-well/ADC 饱和。独立复现：RL-3A source 所有像素已在上端点，换 1/480s capture 的 simulated saturation 为 0，但源缺失高光没有恢复。原 source endpoint fractions、clean-reference 验证状态与 `hdr_recovery_claim=False` 已在 provenance；报告应保留/解释这些区别。
2. Content hash 不能发现同源但不同裁剪、不同 noisy/denoised 文件、重叠 HDR sequence。真实数据必须在原始 sample/episode 分组后再划分 train/val/test；现有合成 smoke 不验证任何 RL-3A 596/107 或 RawGen 真数据 split。
3. 测试只用 synthetic/source-proxy latent；无真实 sensor noise 标定、radiance 标定、native RAW/motion ground truth、真实设备 budget/delay/actual-exposure 闭环；CUDA 不可用。rawpy 已安装，但未有真实 DNG fixture，特别不能宣称 ProRAW JPEG-XL 已支持。
4. Oracle 的 argmin 与报告复用有限 noisy repeats，存在 Monte Carlo 选择乐观偏差。正式研究需独立 evaluation seeds、更大 noise repeats、多个 source groups、置信区间与真实多 sensor/profile 泛化；当前固定 cache regret 是软件实验指标。
5. `implementation_sha256` 使已配对 cache/checkpoint 可审计，但目前主要比较保存的双方 identity。部署 render driver 仍须核对 request 中绑定的 renderer/sensor，而不能仅凭相同文件名认为当前运行环境等价。
6. 文档事实待最终核对：spec 首段曾写 `the paper's PPO`，AdaptiveAE 原文公开训练算法是 A3C；应替换为 `not an A3C reproduction`。这是文字错误，不影响物理实现。

## 可复验命令与本次文件指纹

```bash
cd <repository-root>
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider tests/test_capture_tm_simulation.py tests/test_capture_tm_data.py tests/test_capture_tm_experiment.py
```

审查结束时的关键文件 SHA256（之后文档编辑不影响此代码审查结论；后续代码变化须依变更范围复核）：

- `capture_tm/types.py`: `52462e91ac9e779a8437a60f437c177e704f316a56cc9a79b90afe4e51513aa0`
- `capture_tm/simulation.py`: `5a74c79009ec6f4e2683ea58804e2e939041a73b050ed3bc80e971e291dcb69c`
- `capture_tm/data.py`: `6ebe411225d14eb82869f39f057062a5408822bec8b17b2313b526de84e6db7d`
- `capture_tm/sources.py`: `8ee04f4093efb1d62ac82dcc7d00e0cb47ec8b7d6d33a254631d56171aeaa099`
- `capture_tm/experiment.py`: `072b169da6c5f5a720a7105ca4689e9882f2cb1b8c6129dd7a126090c1a9399d`
- `capture_tm/objective.py`: `5316c5ba00a4a6011ddbc409ba3986e105fb10c05fb3a3fa31ec62995375cc6d`
- `capture_tm/cli.py`: `07beb847c497b8840b68a29afd9848bac57b9c222872243b66fc27bca8e0c349`
- `capture_tm/modular.py`: `099d2b413a738b2eab25a0e670870857c9b1e26b2f547e45fab72f30a3487019`
