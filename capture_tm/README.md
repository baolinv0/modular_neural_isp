# 采集曝光与成片 TM：单帧与 HDR 联合算法

## 数据设计与复现入口

- [S24 数据合成与完整实验计划](../docs/capture_tm/S24_EXPERIMENT_PLAN.md)：S24 输入域、目标与源裁切、两方案八组、训练/评测和待补代码。
- [S24 工程实施计划](../docs/superpowers/plans/2026-10-05-s24-capture-tm.md)：拟新增接口、任务依赖和验收测试，尚未实现。
- [S24 计划前的管线核验](../docs/capture_tm/S24_PIPELINE_VALIDATION.md)：本轮 111 项专项和新 12 场景 RAW 重算；不是 S24 数据训练结果。
- [完整数据合成方案](../docs/capture_tm/DATA_SYNTHESIS_DESIGN.md)：共享物理模拟、两条任务链、素材分级、GT/时序/增益合同和待补研究验证。
- [逐步复现手册](../docs/capture_tm/REPRODUCE_DATA.md)：恢复旧 40 场景/24 模型、从零生成 Bayer 数据、训练推理、导入真实线性 HDR 与故障排查。
- [本次复现核验记录](../docs/capture_tm/REPRODUCE_DATA_VALIDATION.md)：实际执行范围与结果；与历史 314 项全回归记录分开。

已上传的 [artifacts](../artifacts/README.md) 是旧 RGB 模拟实验。新的 Bayer 采集包需按手册生成；不能把旧模型或旧指标换名为 Bayer 结果。

## Bayer 物理数据 pipeline

新增显式线性 HDR 素材导入、原生 Bayer DN 生成、rolling shutter/readout 调度、可迁移数据包、RAW 重算验证及八组训练直接接入。完整命令、GT位置、数据字段与设备标定边界见 [DATA_PIPELINE.md](../docs/capture_tm/DATA_PIPELINE.md)。

```bash
python -m capture_tm.pipeline_cli demo --output runs/capture_tm/raw-demo \
  --scenes 12 --size 16 --scheme both --noise-seeds 0 --seed 2026 --threads 1
python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/raw-demo/acquisition/manifest.json
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/raw-demo/acquisition/manifest.json \
  --output runs/capture_tm/raw-study --scheme both --epochs 1 --warmup 1 \
  --seeds 0 --noise-seeds 0 --threads 1 --prepare-threads 1
```

## 新版：两条完整算法与八组实验

新增 Apple 启发的单帧 AE＋TM、Samsung 启发的三帧 HDR AE＋TM。完整算法说明见 [JOINT_ALGORITHMS.md](../docs/capture_tm/JOINT_ALGORITHMS.md)。两个方案分别运行场景自适应规则/学习 AE × 冻结/学习 TM 的四组实验。

**首轮八组实验已完成（每组三个训练种子）**：[结果与失败分析](../docs/capture_tm/JOINT_RESULTS.md)。当前未证明联合优于只训练 TM；Apple 出现严重采集饱和，Samsung 联合组也有策略集中。完整回归 195 项通过，软件完成不代表画质目标完成。

| 模块 | 新实现 |
|---|---|
| 因果时序 AE | `learned_policy.py`：三帧 CNN、运动差分、直方图、实际曝光元数据与完整曝光计划评分 |
| 物理采集与固定 HDR 融合 | `capture_plan.py`：单帧动作或三帧包围曝光、观测噪声估计、平移对齐与运动拒绝 |
| 可训练 TM | `learned_tone.py`：真实原权重＋2,663 参数的统一 gain/GTM/LTM 条件适配器 |
| 固定目标与质量代价 | `joint_objective.py`：独立参考外观、亮度/细节/信息可用性代价 |
| 联合训练与八组实验 | `joint_experiment.py`, `joint_cli.py`：离散期望、匹配更新预算、验证选模与按场景配对统计 |
| 观测与实际采集推理 | `joint_algorithm.py`：`select()` 请求曝光；`finish()` 按实际生效参数融合及渲染；支持 checkpoint 加载 |

在仓库根目录运行，使用下方安装说明提供的 Python 环境：

```bash
python -m capture_tm.joint_data --output runs/capture_tm/joint-data \
  --scenes 40 --size 32 --seed 2026
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/joint-data/manifest.json \
  --output runs/capture_tm/joint-study --scheme both \
  --epochs 10 --warmup 10 --seeds 0,1,2 --noise-seeds 0,1 \
  --threads 1 --prepare-threads 8
```

每个方案输出四组的训练记录、`selected.pt`/`last.pt`、固定目标与输出对照图、逐场景/逐噪声指标、分类汇总与交互项置信区间。`--prepare-threads` 只影响冻结系数缓存阶段；CPU 核数不足时可降低。

```python
from capture_tm.joint_algorithm import JointCaptureAlgorithm

algorithm = JointCaptureAlgorithm.from_checkpoint(
    'runs/capture_tm/joint-study/apple/seed_0/A11/selected.pt')
request = algorithm.select(previews, effective_capture_state)
# 相机执行 request['plan']，返回含实际 t/g 和 center_s 的 CaptureResult 列表。
image = algorithm.finish(actual_captures)['output']
# 也可在加载的线性 Scene 上运行同一条单计划采集路径：
result = algorithm.run_simulated(scene, seed=7)
```

当前为真实原 TM 上的参数高效训练及分析型 RGB 传感器实验；原系数预测网络冻结。合成实验和机制测试不代表真实手机画质结论，固定 F0 也不等于生产级 HDR 配准/去鬼影。旧数据导入器仍可使用，但 RL-3A/RawGen 代理不能自动升级为干净 HDR 真值。

## 旧 C1 原型（保留）

在现有 `PhotofinishingModule` 和仓库自带的 S24 style-0 权重上新增完整、可运行的研究流程：构造相对线性场景 → 物理采集候选 → 两种专利启发的 TM → 依据最终成片质量生成监督 → 训练 AE 候选评分网络 → 独立测试 → 输出可执行曝光请求。原模型及权重保持原样；这里新增的是 `capture_tm`。

以下旧版接口对应 **C1：固定渲染器下的单次未来采集选择**。独立控制器支持延迟请求、真实生效动作和累计预算账本；离线标签用共同未来中点比较候选，尚未将这个账本连成多帧闭环训练/评测。它借鉴 AdaptiveAE 的观测设计，采用监督候选排序；不是其 A3C 或多帧 HDR 融合的复现。阶段输入仍固定，不能据此声称阶段分支的收益。

## 已实现的模块

| 模块 | 实现 | 输入/输出与约束 |
|---|---|---|
| 采集模拟 | `simulation.py` | 在线性域按曝光窗积分；Poisson 光子 → full-well → read noise → analog gain → ADC noise/black/量化/饱和 → 浮点 digital gain |
| 数据与来源 | `data.py`, `sources.py` | HDR/运动 demo、明确线性域导入、RawGen XYZ、RL-3A RGB16、可选 DNG；按原 source、实际内容和输入 SHA 防跨 split 泄漏 |
| Apple 分支 | `tone.py` | 消除实际采集偏置，独立保留用户影调意图；固定锚点与 HDR shoulder |
| Samsung 分支 | `tone.py` | virtual gain 改变目标统计，经 LPF/CDF 匹配构造曲线；像素不再额外乘一次该 gain |
| 原 TM 接入 | `modular.py` | 两分支共享同一真实、冻结、eval 状态的 style-0 backend；严格载入与 SHA256 身份，无随机权重 fallback |
| AE 网络 | `policy.py` | 最新图像小 CNN + 三帧软直方图 + 实际 t/ga/gd + 阶段 + 共享候选 scorer；合法动作 mask |
| 监督/评测 | `objective.py`, `experiment.py` | 成片误差、主体亮度、模拟 RAW 新饱和、同候选噪声偏差、noiseless capture bias；train/val/test 隔离 |
| 控制/部署入口 | `control.py`, `cli.py` | 请求帧与生效帧分开，剩余预算/防频闪/失败事务检查；仅观测即可推理 |

AE 默认 width=24、32 histogram bins、3 history frames，共 **53,881 参数**。`--image-only` 使用同一网络、参数和候选动作，仅将外部 capture state/stage 置零；保留全部历史图像。候选动作的物理特征仍在两边共享。是否加入外部条件有价值，应由该公平消融的真实测试决定。

## 安装与完整运行

以下命令在仓库根目录执行。本轮实际验证环境为 Python 3.12.14、PyTorch 2.5.1 CPU、numpy 1.26.4。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r capture_tm/requirements.txt pytest

python -m pytest -q
python -m capture_tm smoke --tm-mode apple --output runs/capture_tm/apple
python -m capture_tm smoke --tm-mode samsung --output runs/capture_tm/samsung
python -m capture_tm run --config configs/capture_tm/demo.yaml --output runs/capture_tm/demo
```

输出包括 `data/manifest.json`、`oracle/oracle.pt`、`train/policy.pt`、训练日志、`eval/evaluation.json`、对照图片和 `request.json`。新建输出目录；demo 不会覆盖已有 manifest。默认使用真实 modular 权重；`--backend analytic` 是显式的软件测试选项，只执行 sRGB OETF。

同标签、同随机种子、同预算的图像消融：

```bash
python -m capture_tm train --cache runs/capture_tm/demo/oracle/oracle.pt \
  --output runs/capture_tm/demo-image-only/train --epochs 20 --width 24 --seed 0 --image-only
python -m capture_tm evaluate --cache runs/capture_tm/demo/oracle/oracle.pt \
  --checkpoint runs/capture_tm/demo-image-only/train/policy.pt \
  --output runs/capture_tm/demo-image-only/eval
```

标签生成冻结 TM；AE 训练仅更新 policy。loss 用合法候选的 teacher cost，checkpoint 按 val regret 选择；test 不参与优化或选 epoch。baseline 是最接近参考曝光的合法动作，physics 是观测历史曝光归一化后的辐射/运动启发式，oracle 是当前采样监督的最小 cost，不代表真实最优保证。配置可设置 grid、噪声重复数、参考 intent、WB/CCM 和观测动作菜单；提高 repeats 可减少标签 Monte Carlo 误差。

## 将两种 TM 用于实际线性采集图

保存 **实际采集后的** sensor-linear RGB 为 float32 NPY `[3,H,W]`。允许 HDR >1 和 black subtraction 后的负噪声；固定 frontend 先做 WB/CCM，再显式 clamp 负值。不要把已除曝光的 latent radiance 当成这条命令的输入。真实 backend 的 H/W 均须 ≥12。

```bash
python -m capture_tm render --input captured_linear_rgb.npy --tm-mode apple \
  --capture-bias-ev 1 --render-intent-ev 0 --sensor-profile sensor.json \
  --output runs/capture_tm/render-apple
python -m capture_tm render --input captured_linear_rgb.npy --tm-mode samsung \
  --capture-bias-ev 1 --render-intent-ev 0 --sensor-profile sensor.json \
  --output runs/capture_tm/render-samsung
```

该例的 `capture_bias_ev=1` 表示相对参考采集亮度高一档。一般为 `log2((t/t_ref)*(ga/ga_ref)*(gd/gd_ref))`；这三个数必须来自实际生效采集。`render-intent-ev` 是用户期望；`virtual-gain` 是**尚未应用的额外渲染请求**。输入 tensor API 不能检测外部已经乘过的 render gain，调用方必须遵守这个合同。

总 requested EV 为 `-capture_bias_ev + render_intent_ev + log2(extra_virtual_gain)`；先将采集补偿限幅，再将补偿与 style 之和限幅，默认 ±4 EV。以输出账本的 applied EV / virtual gain 为准。输出 `rendered.npy` 是 float display RGB，PNG 为8bit预览，`curve.npz` 为曲线，`render.json` 保存请求/实际 gain 和 renderer/sensor 身份。

Apple 的 HDR shoulder 是工程扩展，并未逐字实现专利的有限白点 D(1)=1。Samsung 使用显式 `Z=clip(G*Y,0,1)` 构造目标统计，20% 固定锚点正则，属于近似分布匹配；亮度范围小于首个 log bin 时转为解析固定锚点，避免暗平坦场的原子/量化偏亮，Python API 返回退化 mask。两个分支的保黑、曲线单调保证仅针对 `pre_backend`，不能推广到随后学习模型的最终像素；极端浮点数下 shoulder 会趋近白点。

## 数据构造与真实来源

场景域固定为 `sensor_linear_relative_radiance`：参考快门、unit gain 下，未裁剪的期望信号等于场景数组。此定义是相对采集坐标，不是绝对辐射标定。动态场景的 float32 数组为 `[T,3,H,W]`，timestamps 使用 float64 且严格递增；静态单帧可在任意时刻采集。动态数据必须覆盖完整曝光窗，不会复制边界帧补未来。

推荐先按 original scene/episode 分 train/val/test，再在各 split 内产生光照、快门、gain、噪声、运动变体。`source_id` 应对同场景 crop、reset、noisy/denoised 文件一致；内容 SHA 只能发现重复数据，不能替代真实 source 分组。相同 source、原输入 SHA、canonical path 或实际 decoded frames 内容跨 split 均拒绝。frames/mask hash、timestamp、来源、随机种子、每次采集 metadata 与 renderer/协议指纹一起存入标签和 checkpoint。

| 来源 | 导入处理 | 适用边界 |
|---|---|---|
| AdaptiveAE 的 HDR 视频思路 | 明确线性化的视频、时间戳、camera transform → 线性曝光积分，再独立采样传感器噪声 | 可提供真实运动；本仓库不捏造 HDRV/DeepHDRVideo 下载文件或标定参数 |
| RawGen 默认 XYZ 输出 | inverse sRGB OETF **一次** → 用户提供 XYZ-to-sensor 矩阵 → 相对场景 | 不是其 `9cam` display preview；单帧没有真实时间轨迹，也不能还原生成源的已裁高光 |
| RL-3A linear RGB16 | 精确保留16bit PNG码值，除用户提供的 t/ga/gd 比例，记录源饱和/残余噪声 | demosaiced RGB proxy，非 Bayer；ISO 单独不足以确定 analog/digital 分解 |
| DNG | optional rawpy：linear gamma、unity WB、禁 auto bright、raw camera RGB | 未在真实 Apple ProRAW fixture 上验证；不能解码时拒绝，不用8bit预览顶替 |

默认 `SensorProfile` 是 `assumed_engineering`，unit reference white≈full-well；`measured_sensor` 要求来源记录。三通道独立 RGB 噪声不包含 CFA、光谱、row noise、dark current、rolling shutter 或多帧融合。WB/CCM 必须把实际 camera RGB 映射到本 backend 所需线性颜色空间；默认 identity 只适合 proxy/软件实验。RL-3A 的 natural-denoising residual noise fit 以独立桥接接口保留，**未悄悄当作 electron sensor calibration 使用**。

CLI 导入示例；多场景可追加到同一 output 的 manifest，但 sensor profile 必须一致：

```bash
python -m capture_tm import-rawgen --input rawgen_xyz16.png --output runs/capture_tm/real-data \
  --scene-id gen001 --source-id gen001 --split train --xyz-to-sensor xyz_to_sensor.json --sensor-profile sensor.json
python -m capture_tm import-rl3a --input denoised_raw16.png --output runs/capture_tm/real-data \
  --scene-id scene002_clean --source-id original002 --split val \
  --exposure-s 0.0083333333 --analog-gain 2 --digital-gain 1 --sensor-profile sensor.json
python -m capture_tm import-linear --input linear_hdr_video.npy --input-layout TCHW \
  --frame-times times.json --input-domain sensor_linear_relative_radiance \
  --scene-id video003 --source-id video003 --split test --sensor-profile sensor.json \
  --output runs/capture_tm/real-data
```

可用 `sources.import_linear_scene` / `import_rawgen_xyz` / `import_rl3a_proxy` Python API 提供相同字段。将 config 的 `data.manifest` 指向导入的 manifest（相对 config 所在目录解析），再执行 `run`。未经验证的 clean proxy 的 final-reference 指标只能解释为 source-relative surrogate；原 source 已裁剪的信息没有恢复。评测 `raw_saturation` 只表示本次模拟新增的 full-well/ADC 饱和。

## 实时请求与延迟账本

`request_from_observations` 只接收三帧分析 RGB `[3,3,H,W]`、实际 capture state `[3,3]` 与 checkpoint；分析 thumbnail 显式限制到 `[0,1]`，用于 renderer 的采集图保留 HDR。state 为 `log2(t/t_ref), log2(ga), log2(gd)`，应使用 `action_features` 生成。它不接收 clean scene、candidate output、teacher cost 或未来帧。

```bash
python -m capture_tm infer --observations observations.npz \
  --checkpoint runs/capture_tm/demo/train/policy.pt --observation-frame-id 7 \
  --remaining-capture-s 0.04 --future-center-offset-s 0.025 \
  --output runs/capture_tm/live-request.json
```

NPZ 包含 `previews` 和 `capture_state`。观察帧 k 完成后，提交 tick 为 k+1，生效 tick 为 k+1+delay；结果包含这三种帧号、绝对 t/ga/gd、采集 bias、请求补偿以及绑定 renderer/sensor 身份。驱动必须核实实际生效帧和实际参数再渲染。`remaining_capture_s` 要由调用方扣除已经发生及 pending/等待帧的费用；省略时假定 checkpoint 的完整剩余预算。预算含每次 final capture 的 shutter+readout，离线 oracle 的三帧 prior observations 是既有观测，不计入该剩余预算。

未来曝光起点必须晚于最后观测 shutter end+readout，历史观测曝光窗也不能重叠；live mask 同样按最新实际快门和未来 midpoint offset 检查。无合法动作会拒绝。真实驱动的额外秒级控制延迟须纳入 scheduled offset；当前离线 common midpoint 没有声称验证硬件 cadence。防频闪仅实现理想二倍 mains modulation 的快门整周期规则，不能保证所有 LED/rolling shutter。

`DelayedExposureController.capture_frame` 以实际 effective action 重采集；成功才提交 cursor/预算/ledger，失败不消耗状态。当前 API 发出 `request_only`，不会虚构已拍到的图像。cache replay 与 evaluate 都核对 dataset、sensor、动作、renderer 和完整 teacher protocol 身份；既有 cache 是快照，当前运行代码若改变，应重建标签并重训，而不能据旧快照宣称重新验证新 renderer。

## 验证与研究边界

测试覆盖光子/gain 分离、噪声统计、曝光运动积分、饱和来源、HDR/负读噪声、时间平移、分组隔离、精确 RGB16、curve/EV/意图、真实冻结权重、非法动作、容量匹配、延迟账本、checkpoint 身份和数据→训练→推理。另由独立 AE/TM/ISP reviewer 审核，发现与修复见 [验证记录](../docs/capture_tm/VALIDATION.md)，相关原始依据见 [来源映射](../docs/capture_tm/SOURCES.md)。

CPU synthetic/source-proxy 运行证明软件可运行。真实 iPhone/Samsung 画质收益、曝光 metadata 的增益、跨 camera 泛化、native Bayer ISP、学到的 denoising、完整多帧闭环与延迟鲁棒性仍需真实数据和标定实验；当前没有这些结果，也不把专利启发机制或参考论文架构作为新颖性结论。
