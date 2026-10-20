# C / AE 独立只读评审

评审时间：2026-10-05。仓库：`<repository-root>`；基线 `e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3`。评审对象是本次新增的 `capture_tm`；重点为 `policy.py`、`control.py`、`experiment.py`、`objective.py`、`cli.py`。未改动实现或测试。以下行号对应本次读取的未提交版本，后续修复可能移动行号。

结论：监督候选评分原型的主路径可运行，模型输入没有直接读取 latent/teacher/candidate output。现有 AE/控制/实验测试 49/49 通过，但发现 5 个需要处理的边界问题；其中 shutter 因果边界和有效帧号错误可直接破坏因果时序。当前证据不能证明真实手机画质收益，也不是 AdaptiveAE/PPO 的完整复现。

## 重要问题

### AE-1 / P1：中心时刻顺序不足以保证候选捕获发生在观测完成之后

位置：`capture_tm/experiment.py:117-120, 141-143, 155-159`。

`build_oracle` 只要求 `capture_center_s > preview_centers_s[-1]`。观察和候选都是有限曝光积分，实际观测只有在其 shutter end 后才可使用；候选必须在这个时间（加所需 readout/control latency）后才可能开始。当前接受 `capture_center_s=.001`、最后 preview center `0`、observation exposure `1/120`、candidate exposure `1/30`。最后观测到 `.0041666667 s` 才完成，而候选从 `-.0156666667 s` 已经开始。即便 policy 没有显式读 teacher，训练样本也把候选曝光已经发生之后的信息当作决策前输入。

实际复现输出：

```text
SHUTTER_CAUSALITY accepted=True
last_preview_end=0.004166666666666667
earliest_candidate_start=-0.015666666666666666
```

最小修法：加载 `observation_action` 后验证全部时间有限，且对每个可行候选要求 `capture_center_s - exposure_s/2 >= last_preview_center + observation_exposure_s/2 + declared_readout_latency`。如果 frame delay 不映射到真实秒，明确把 oracle 声明为单次候选比较，并把 delay 排除在该时间协议之外；不要宣称它验证了闭环延迟。加入一个以 moving scene 运行、中心顺序合法但 shutter 窗重叠必须拒绝的测试。

### AE-2 / P1：live 请求的有效帧号与控制器少差一帧

位置：`capture_tm/experiment.py:338-357`，对照 `capture_tm/control.py:156-168`。

控制器的 cursor 语义是观察 frame `k` 后，下次 request 的 `requested_frame=k+1`，其 `effective_frame=k+1+control_delay_frames`。live API 却返回 `k+max(1,delay)`。默认 delay=1 时 API 报 frame 1，但控制器真正的 due frame 是 2；所有 delay>=1 都少一帧。现有测试只检查 `effective_frame > observation_frame`，因此遗漏该问题。

实际复现输出：

```text
OFF_BY_ONE api_effective_frame=1 controller_effective_frame=2
```

最小修法：统一 helper 或直接用 `observation_frame_id + 1 + delay`，同时返回 `requested_frame_id=observation_frame_id+1`。校验 observation frame 为非 bool 的非负整数。以 delay=0/1/2 的 observation -> request -> controller due frame 一致性测试替换仅“大于”断言。

### AE-3 / P2：live 推理不接收剩余预算，可发出当下不合法的候选

位置：`capture_tm/experiment.py:338-346`，`capture_tm/cli.py:66-76, 114-123`。

API 始终用 checkpoint 的全额 `max_total_capture_s` 创建 mask，没有累计已用时长、remaining budget、pending action 或 stage 参数。某序列已经消耗 `.0723333333 s`、只剩 `.0076666667 s` 时，API 仍返回 `t=.0083333333 s`（另有 `.002 s` readout）；同一约束下 `controller.request` 立即报 `requested action must be feasible within the remaining budget`。因此 policy 内部的非法 action masking 不能证明 live 请求满足序列预算。请求标为 `request_only` 且控制器会拒绝，因此这不是实际超预算执行；但部署入口的 legality 是不完整的。

实际复现输出：

```text
EXHAUSTED_BUDGET remaining=0.007666666666666669
api_requested exposure_s=0.008333333333333333
controller_rejected=requested action must be feasible within the remaining budget
```

最小修法：API/CLI 增加 `remaining_capture_s` 或 `total_capture_s`，调用相同的 `limits.feasible_mask(..., remaining_capture_s=...)`；无可行候选时显式结束/拒绝。在非零 delay 情况，可由控制器提供扣除已知 pending/intervening captures 后的可用预算。控制器当前最终 commit 前再次校验、且失败 transactional 的行为是正确的；不必为此改成悄悄截断曝光。

### AE-4 / P2：cache replay 推理省略了 checkpoint/cache 兼容性校验

位置：`capture_tm/experiment.py:360-369`，对照 `285-290`。

`infer_request` 直接取 cache 的 previews 和 capture_state，却用 checkpoint sensor/reference exposure 来解释候选；它不检查两边 sensor/actions/renderer/constraints。一份合法但 `reference_exposure_s` 加倍的独立 cache 可直接用于旧 checkpoint 并成功返回请求，而 `evaluate_policy` 对同一组合明确拒绝 `Checkpoint/cache manifest_sha256 mismatch`。cache 的 capture_state 第一列按新 reference 定义，checkpoint 的 candidate feature 第一列按旧 reference 定义，含义不一致且无提示。

实际复现输出：

```text
REPLAY_MISMATCH infer_accepted=True
evaluate_rejected=Checkpoint/cache manifest_sha256 mismatch; rebuild labels or retrain
```

最小修法：抽取兼容性 helper，在 evaluate 和 cache replay 上共用。至少验证 sensor、actions、renderer、constraints、data/protocol identity。独立 live observations 入口需要明确沿用 checkpoint 的 reference/domain convention；不要把 cache replay 的缺失检查误称为支持跨 sensor 的推理。

### AE-5 / P2：checkpoint 未锁定完整 oracle 协议，evaluation 可静默混用不同时间设置

位置：`capture_tm/experiment.py:182-186, 261-267, 288-290`。

cache 保存了 `capture_center_s`、`preview_centers_s`、noise seed/repeats，checkpoint 没有保存这些字段。以同一 manifest、actions、sensor、constraints、renderer，另建 `capture_center_s=.08`、`preview_centers_s=(-.02,0.,.04)` 的 cache，旧 checkpoint 的 evaluate 成功，虽然训练与报告的观测/预测时间协议已变化。用于有意的时间泛化评估可以成立，但当前报告没有声明分布改变，也不输出训练与测试的两份时间协议，兼容性报错文案却暗示协议已经一致。

最小修法：保存并检查完整 `oracle_protocol`（观测 action、capture/preview time、repeats、noise seed、target/objective/render version）；如需要独立 noise 或时间泛化，允许显式参数并在报告同时列出 train/eval 协议。严格复现还应给 data 文件和 renderer checkpoint 内容做 digest，manifest 文件内容 hash 与 checkpoint 路径名本身无法锁定被外部覆盖的 `.npy`/weights 内容。

## 已核查正确的边界

- `_inputs` 只给 policy previews、历史 capture_state、候选 physical features、legal mask；target、costs、clean candidate/output 均不作为模型输入。训练 loss 使用 teacher 是合理的离线监督。
- 历史 features 与候选 features 共用 `action_features`：`log2(t/reference_t), log2(ga), log2(gd)`；候选 features 描述动作，未编码其实现后的质量。对原型而言这一区分正确。
- `train_policy` 优化只访问 train；checkpoint epoch 只由 val regret 选择；`evaluate_policy` 只聚合 test。源身份跨 split 检查存在。本次 data 作者正在增强来源识别，未把其修复中的状态算作已验证完成。
- full 与 `use_auxiliary=False` 实例参数数相同，独立复现 width=4 时均为 5621；后者保留所有历史 image histogram 和 latest CNN，只把 external capture_state/stage 置零。作为容量匹配的历史图像 ablation，代码结构公平。候选 action 特征仍保留，符合两边相同动作族要求。
- legal-only ranking loss 在计算前移除无效 teacher/scores，单候选及极端尺度测试通过；非法候选不能赢。
- 控制器按实际 effective action 计费，requested/effective provenance 分离；失败 simulation 不 commit 状态或 ledger。当前独立 controller 测试通过。
- objective 使用一个固定 intent/reference appearance；raw saturation 来自 capture mask；没有靠调暗 target 掩盖噪声。clean_output 是 offline teacher 的同一候选 noiseless render。
- checkpoint 使用 `weights_only=True`、严格 state_dict load、eval mode；未发现随机权重 fallback 或 test data 直接参与优化。

## 证据限制

- 工作流每条记录的 observation action 固定，三帧 state 相同；本次 fixture 的全体 state 只有 `[0,0,0]` 一种值。train/evaluate/infer 都不显式传 stage，使用固定 history 3 得到 `[2,3]`。所以本次结果不能支持“读取实际曝光 metadata 或阶段信息优于 image-only”的结论，哪怕参数容量匹配。
- oracle/evaluation 是固定时间的单步候选评分，未把 `DelayedExposureController` 串入训练或评测，也未计入闭环等待帧的画质/总时延。控制器测试证明软件 ledger 的时序，不能转化成闭环 AE 性能证明。
- demo/smoke 的 test split 只有一个 scene；结果适合验证程序能跑与 teacher regret 非负，不足以支持收益、统计泛化或真实手机质量主张。
- 低重复次数 oracle 有 Monte Carlo 误差；oracle top1/regret 是当前采样 surrogate 的比较，不是真实最优动作保证。
- 没有 full AdaptiveAE AlexNet/PPO/reward/trajectory 复现，本实现也明确如此声明。物理模型与 RawGen/RL-3A proxy 的校准限制需在最终报告保留。

## 复现与验证

命令：

```bash
python -m pytest -q tests/test_capture_tm_policy.py tests/test_capture_tm_control.py tests/test_capture_tm_experiment.py
```

结果：`49 passed in 4.25s`。

本次具体 oracle/checkpoint/cache fixtures 保存在 `<local-validation-workspace>/ae-review-ekvkwzy7`，无需改动仓库即可重跑最小检查：

```python
import torch
from pathlib import Path
from capture_tm.experiment import request_from_observations, evaluate_policy, infer_request
from capture_tm.control import DelayedExposureController, ExposureConstraints
from capture_tm.types import CaptureAction

root = Path('<local-validation-workspace>/ae-review-ekvkwzy7')
cache, checkpoint = root/'oracle/oracle.pt', root/'train/policy.pt'
state = torch.load(checkpoint, weights_only=True)
row = torch.load(cache, weights_only=True)['records'][-1]
request = request_from_observations(row['previews'], row['capture_state'], checkpoint)
ref = CaptureAction(state['sensor']['reference_exposure_s'])
controller = DelayedExposureController(ref, ref, ExposureConstraints(**state['constraints']))
controller.advance()  # observation frame 0 has completed
due = controller.request(CaptureAction(**request['requested_action']), source_observation_frame=0).effective_frame
print(request['effective_frame_id'], due)  # old implementation: 1, 2

evaluate_policy(root/'other_times/oracle.pt', checkpoint, root/'replay_other_eval')  # accepted
print(infer_request(root/'different_sensor/oracle.pt', checkpoint))  # accepted
```

重新生成 shutter 边界问题的最小步骤：

```python
from capture_tm.data import write_demo_dataset
from capture_tm.experiment import build_oracle

manifest = write_demo_dataset('/tmp/ae-review-new/data', size=16, scenes=3)
build_oracle(manifest, '/tmp/ae-review-new/oracle', backend='analytic',
             exposures_s=[1/120, 1/30], analog_gains=[1.], repeats=1,
             capture_center_s=.001)  # old implementation accepts overlapping shutter windows
```

AE-1 至 AE-5 应修复并添加针对性的边界验证后，再把该原型作为因果部署入口交付。无需以更多 synthetic training epochs 代替上述逻辑修复。

## 修复复核 / 2026-10-05

实现作者修复后进行了第二次独立只读检查，重新生成 fixtures 并验证实际 API 与 controller；不是仅根据作者反馈标记通过。新 fixtures 为 `<local-validation-workspace>/ae-rereview-z8s5txid`。

| 原问题 | 复核结果 | 实際证据 |
|---|---|---|
| AE-1 shutter 因果边界 | 已解决 | `.001` future center 被拒绝；重叠 preview shutter/readout 被拒绝；重复和 NaN preview times 被拒绝；逐 record candidate mask 包含最后观察 shutter end 和 readout |
| AE-2 有效帧号 | 已解决 | delay 0/1/2 下，观察 frame 0 -> requested frame 1 -> effective 1/2/3，均与 controller 实际 due frame 一致；负数、bool、浮点 frame ID 被拒绝 |
| AE-3 live 剩余预算 | 已解决（以调用方提供真实状态为前提） | `remaining_capture_s=.0076666667` 显式拒绝；`.001` future center offset 显式拒绝；CLI observations 分支传递这两项；返回值标明默认预算与 caller 扣除 prior/pending 成本的假设 |
| AE-4 replay 兼容性 | 已解决 | 不同 sensor/reference 的 cache replay 被拒绝；infer/evaluate 共用 `_check_pair` |
| AE-5 oracle 协议身份 | 已解决 | 不同 capture/preview times 的 cache 在 infer/evaluate 都因 protocol mismatch 被拒绝；只修改 `.npy`、保持 manifest bytes 不变，dataset digest 改变且旧 checkpoint replay 被拒绝；renderer 内容与实现 digest 已记录并参加 cache/checkpoint 对照 |

第二次测试结果：`53 passed in 2.75s`，命令仍为上文的 policy/control/experiment targeted pytest。独立脚本另覆盖三种 delay、历史 shutter、重复/NaN times、无 budget、live future timing、frame ID 类型、真实独立 sensor cache、独立时间协议 cache，以及 manifest 未变但 frame payload 变化。

`observation_actions` 菜单也经独立检查：每条 record 的三帧 `capture_state` 与对应实际 observation action 一致，gain=1/2/4 分别得到 `[0,0,0]`、`[0,1,0]`、`[0,2,0]`。因此第一版报告中“全体 metadata 恒定”的限制不再适用于新的 demo config；stage 仍固定，且仍未证明 metadata 的收益。

### 新发现 AE-6 / P2：gain 菜单变化会被物理基线误当成运动

位置：`capture_tm/experiment.py` 的 `_physics_estimate`：`motion=(preview[-1]-preview[0]).square().mean()`，调用方只传 `observed[-1]`。

加入 observation gain 菜单后，静态场景同一 radiance 在 ga=1/2/4 时 observation 亮度不同。物理基线直接拿 raw preview 差作为 motion，没有用各历史帧的实际 t/ga/gd 消除亮度比例。于是 gain-only history 会使长曝光被额外惩罚，削弱该 baseline。当前无任何收益主张，因此不能反推 full policy 的收益；但这是应修的比较语义问题。

无 clipping、无噪声、同一静态 `.1` radiance 的最小数值复现：

```python
from capture_tm.experiment import _physics_estimate
from capture_tm.types import CaptureAction, SensorProfile
import torch

actions = [CaptureAction(1/120), CaptureAction(1/30)]
gain_history = torch.stack([torch.full((3,4,4), x) for x in (.1,.2,.4)])
constant_gain = torch.full_like(gain_history, .4)
latest_action = CaptureAction(1/120, 4.)
x = _physics_estimate(gain_history, latest_action, actions, SensorProfile())
y = _physics_estimate(constant_gain, latest_action, actions, SensorProfile())
print((x-y).tolist())
# [0.09000000357627869, 1.440000057220459]
```

最小修法：传完整 observation action 历史，将每帧 preview 除其 `t/reference_t * ga * gd` 后再计算 motion proxy。clipping、噪声及亮度变化仍使其只是 heuristic，需要保留限制。增加静态、无 clipping 的 gain-only history 不产生额外 motion penalty 的测试。该问题已实时通知作者，尚未在本次复核中验证修复。

### 仍需保留的证据边界

stage 固定；offline common future midpoint 不映射 controller 的 frame cadence；label budget 现在明确只约束最终一次 capture、prior previews 排除。这些边界已在 protocol 明确，不能把独立 ledger 测试称为完整闭环 AE/delay 性能验证。demo 只有一个 test scene、少量重复、假设传感器物理模型和 source proxy 的校准限制仍成立；软件通过与 oracle regret 不构成真实手机质量收益证据。

## AE-6 最终关闭 / 2026-10-05

作者已修复 `_physics_estimate`：接收完整 observed action 历史，按每帧实际 `t/reference_t * ga * gd` 归一化 preview，再计算历史 motion。调用方也已传 `observed` 全历史；代码保留了 observation noise/censoring 会偏置 motion proxy 的说明。

独立重跑原来的 gain-only 复现，长短曝光额外 penalty 现在均为 `[0.0, 0.0]`。另外独立构造同时改变 t、analog gain、digital gain 的静态历史，额外 penalty 仍为 `[0.0, 0.0]`；再让最早帧 radiance 真正减少一半，得到正的 motion penalty `[0.002500000176951289, 0.04000000283122063]`，说明修复没有把运动项整体关闭。

`python -m pytest -q tests/test_capture_tm_experiment.py`：`9 passed in 3.91s`，包含新 gain-history normalization 回归测试。此前 policy/control/experiment 的 53 项全通过结果仍记于上节；本次唯一增量为 AE-6 及其测试。

最终状态：AE-1 至 AE-6 的已报告实现问题全部解决；本次 AE 评审没有未关闭的实现 blocker。stage 固定、单次离线 common midpoint 及校准/泛化/真实手机证据限制仍需保留，也不能据软件通过宣称收益。
