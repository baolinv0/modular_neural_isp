# 独立 TM 代码审查

审查日期：2026-10-05。仓库：`<repository-root>`；base：`e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3`。审查为只读，未修改实现。研究合同依据：`c_implementation_research/patents.md`。没有将工程近似解释为完整专利复现或法律覆盖判断。

## 结论

Apple 正常域的采集偏置补偿与 render intent 分开；Samsung 的额外 virtual gain 改变目标统计、曲线与输出，源统计没有预乘该 gain，最终像素执行没有再次乘入同一个 virtual gain。真实 shipped style-0 后端正确 strict-load 并冻结，两个 branch 可以共享同一个后端。

本次发现两个应修的实验接口身份问题，已及时通知 root：teacher/cache 的 renderer 信息没有使用已经计算的 checkpoint digest，以及 `infer_request` replay 不校验 cache/policy 的身份匹配。root 已确认将修。另建议将 JSON 外实际数据资产与 oracle protocol 的身份纳入缓存协议；此项与 AE reviewer 的实验协议审查重叠。

除这些身份问题及 root 已知的尺寸 guard 外，未发现正常域导致内部重复 gain、HDR 提前截断、错误权重加载或最终评分偷换目标的实现错误。Samsung 默认离散化在极暗原子输入下仍有明显跨曝光误差，应作为近似方法的限制公开；不能泛化 Apple 的低中调参考保持结论到 Samsung。

## 需要修复的接口问题

1. **teacher/cache 缺少完整 renderer 身份。** 初读 `capture_tm/experiment.py:178–186` 仅保存 `backend/tm_mode/render_intent_ev/checkpoint` 字符串。`modular.py` 已计算同一字节快照的 `checkpoint_sha256/renderer_identity`，但没有写入 cache。相同路径换权重后，cache/policy 的 `renderer` 对比不能识别此变化。最小建议：保存 digest、TM/frontend/objective 配置及方法版本/指纹，并在 checkpoint 和评估中复用同一身份。

2. **replay inference 可静默混用不同 sensor/cache。** 初读 `experiment.py:360–369` 的 `infer_request` 载入 cache 后直接调用 checkpoint policy，没有 `evaluate_policy` 的 `actions/renderer/manifest/sensor/constraints` 比较。最小复现使用 existing `c_validation/ae-review-ekvkwzy7/different_sensor/oracle.pt` 与同目录 `train/policy.pt`：cache reference exposure 为 `1/60`，policy 为 `1/120`；`sensor_identity_equal=False`，接口仍返回 `request_only` 且 `capture_bias_ev=0`。最小建议：统一 compatibility helper，评估与 replay 同用。直接 causal API 没有 cache 身份，需明确 capture-state 的单位/参考曝光由调用者满足。

3. **JSON hash 不覆盖文件内容（协议建议）。** `manifest_sha256` 仅 hash manifest JSON。替换同路径 `.npy/.npz` 或 mask 后，重建 cache 的 JSON hash 仍可相同；同 renderer 的 policy/cache 对比可能通过。建议包含实际资产内容或载入 tensor 的 digest，以及 repeats/seed/时间点/observation action 等 teacher protocol。该问题交给 root 与 AE review 合并处理。

## 已验证的作用域与行为

- Python：`python`。执行 `pytest tests/test_capture_tm_tone.py tests/test_capture_tm_modular.py tests/test_capture_tm_experiment.py -q`，初读版本 **34 passed**。
- 将 backend 的 **251/251 个 state entries** 与 shipped checkpoint 逐项 `torch.equal` 核对，全部相同；所有子模块 eval，所有参数 `requires_grad=False`。
- 对两个 mode、16 张随机 RGB `[0,4]`、capture bias `[-3,3]` 和 intent `0.75`，pre_backend/输出均有限且 bounded，返回曲线的最小相邻差为正。
- 灰度连续 ramp `[0,4]`，virtual gain `0.25/0.5/1/2/4` 的 pre_backend 均值严格递增。Apple 约为 `0.483/0.697/0.830/0.907/0.949`，Samsung 约为 `0.497/0.739/0.865/0.930/0.963`。
- Samsung 连续 ramp `[0.02,0.25]`，gain 2/1 的输出均值比约 **2.0113**；source hist 完全相同、target hist 不同。这支持一次执行与目标统计的 gain 关系，不支持精确 histogram/mean equality。
- 对 256 个逐 bin 原子核对 `_mass_preserving_lpf`，非负且最大质量误差 **0**。滤波保持质量不等于保持均值。
- `quality_cost` 对所有候选使用同一个 latent/reference appearance；raw saturation 来自 capture mask；clean_output 是同一候选的 noiseless capture，故 `rendered_noise_mse` 不通过换暗 target 隐藏噪声，`noiseless_capture_bias_mse` 还会计入 blur、clip、量化和 TM 近似造成的偏差。该项名称不应解释为仅测 EV 偏置。

## 需保留的限制

- **保证只到 pre_backend。** max-RGB 共同缩放保障正常域 channel ratio 和 bounded RGB；black/monotone 曲线性质不应延伸到 trained local/chroma photofinishing 最终输出。root 已知该作用域。
- **HDR shoulder 为工程扩展。** 当前固定 `0.7` 锚点及有界肩部不满足 Apple 示例的 finite white endpoint `D(1)=1`，此区别已在研究笔记明确，不应重新宣称原文逐式复现。
- **Samsung 暗原子近似误差。** 默认 bins 256、Identity backend、32×32 全灰 `0.001`：neutral mean `0.0025073313`，输入乘 2 且 bias EV=1 的 compensated mean `0.0048120865`，绝对差 `0.0023047554`。`0.01/0.1/0.5` 的同类差约为 `0.000569/0.000225/0.002900`。这是离散原子、LPF 和黑端点约束的组合误差；不是 gain 被重复执行。若接受该 approximate branch，报告该限制与实际 output mean/误差；若要求 dark atom 精确参考恢复，需要另行定义 atom fallback/分辨率或校准。
- **未实现外部重复 gain 检测。** 当前 tensor API 把 virtual_gain 定义为额外 request；没有 gain IDs 或外部已应用 rendering gain ledger。内部路径正确防重复乘法，但不能宣称能检测调用者已先把同域 rendering gain 乘入 `x`。应明确输入必须满足未应用该同域 virtual rendering request 的合同。
- **冻结权重不等于冻结逐帧参数。** 原模型 learned Gain/GTM/LTM 等仍根据候选 pre_backend 预测各自参数；这些是独立训练 photofinishing stage，不接收上述 virtual gain。结论应写“共享冻结权重与同一算法/意图”，不能写“所有候选逐帧系数相同”。

## 修复复核状态

本报告记录 root 修复前的独立观察与最小复现。需在 root 身份校验变更后重新运行对应 regression 并将已解决状态附在这里；不应将已确认待补项误报为最终遗留 bug。

### 修复后独立复核（2026-10-05）

**最终状态：本次识别的 renderer/cache/replay 身份问题和暗平坦分布问题已解决；未发现新的正常域阻塞问题。** 本节取代前文“root 将修/待复核”状态，前文保留为可追溯的修复前记录。

最新执行 `pytest tests/test_capture_tm_tone.py tests/test_capture_tm_modular.py tests/test_capture_tm_experiment.py -q`：**41 passed in 10.61s**。独立 probe 目录为 `c_validation/tm-final-tb3ax981`，用 12×12、3 个 demo 场景实际构建 real Modular Samsung cache，另建 analytic cache 并训练 1 epoch 验证 replay。

- real cache 的 `checkpoint_sha256` 与加载后 backend 的实际 digest 相等，`renderer_identity` 相等；重新逐项核对 shipped 权重 **251/251 相等**，所有 module eval、所有 parameter frozen。尺寸 guard 明确拒绝 H/W<12，最低合法 12×12 的 real oracle 构建通过。
- `renderer.implementation_sha256` 现包括 tone/modular/objective/frontend/experiment/simulation/types/policy/control/data/sources 以及上游 `photofinishing_model.py`、`utils/constants.py`。常量中的增益/伽马上下限、转换矩阵和 EPS 不再遗漏在本地方法身份之外。
- infer replay 与 evaluate 共用 `_check_pair`。独立构造 renderer、sensor、dataset_sha256、protocol 四种不匹配，两入口的 **8/8 个调用均在出结果前明确拒绝**。正常 inference 的 renderer、sensor 身份与 cache 相同，携带 64 位 policy checkpoint digest。
- dataset identity 中的 frames/mask hashes 由加载的 float32 内容系统重算，不能由传入 provenance hash 冒充。独立保持 manifest JSON 完全不变，仅改 mask `.npy` 一个值后重建 cache：`manifest_sha256` 相同，`dataset_sha256` 改变。protocol 包含 seed、repeats、时间点与 observation action menu；actions/constraints/sensor 另在 pair 中绑定。
- Samsung 对 `maxRGB` 分布范围不大于第一 log-input bin 宽度的图像明确标记 `degenerate_distribution_mask`，执行固定 analytic HDR anchor；统计仍作为 diagnostics。32×32 全灰 `0.001` 与输入×2、bias EV=1 均精确恢复到 float32 的 `0.0010000000475`；再加 intent EV=1 为 `0.0020000000950`。全灰 `0.01/0.1/0.5/1/4` 的中性与跨采集尺度补偿也相同，肩部仍按固定 HDR 公式执行。
- 非退化 ramp `[0.02,4]` 不启用 fallback；gain 1/2 的 source histogram 相同、target histogram 改变，曲线单调且 2 与 4 的 HDR 区别保留。原 gain 正负号、意图分离、一次执行及共享真实 renderer regression 均通过。

修复后的限制仍必须保留：**非退化 Samsung 是近似分布匹配，不能保证每个低中调/跨曝光值精确一致；输入若已被调用者乘入同域 rendering gain，接口不会自动识别；黑端点与单调保证只适用于 curve/pre_backend，不能延伸到最终 trained backend。** 退化分支是显式工程 fallback，不是新增的专利完整复现声明。fallback 改变了平坦场景的 Samsung 外观，应使用修复后方法重新生成相应 teacher/cache/结果；新的 method fingerprint 会识别新旧 cache 的方法差异。
