# C 实现验证记录

2026-10-05。CPU 软件与模拟验证；当前结论限于 C1 单步采集选择原型。

## 运行环境与测试

- Python 3.11，PyTorch 2.5.1+cpu，numpy 1.26.4，rawpy 0.27.1，PyPNG 0.20220715.0。
- 仓库根目录执行 `python -m pytest -q --tb=short`：**136 passed**，最终运行13.18秒。
- `git diff --cached --check` 无错误。
- 真实 style-0 `render` CLI 对 Apple/Samsung 均成功：64×64，float32输出有限且在[0,1]，capture bias=+1时 applied render EV=-1。
- 原始 `PhotofinishingModule`、上游依赖文件及模型权重未修改；新测试是本次新增套件，上游没有额外可用的 pytest 单元套件。

## 独立评审及关闭项

| 角色 | 发现与处理 | 最终复核 |
|---|---|---|
| [AE](reviews/AE.md) | 曝光窗因果、effective frame少1、剩余budget、infer pairing、time/noise protocol身份、gain误当motion | AE-1至AE-6全部关闭；阶段固定与单步评测限制保留 |
| [TM](reviews/TM.md) | 同路径换权重未区分、infer错sensor、暗平坦CDF原子误差 | digest/统一pair检查/解析退化fallback完成；non-degenerate仍近似，外部已应用gain不能自动检测 |
| [ISP](reviews/ISP.md) | 绝对时间精度、重复source绕过、实际数组/mask身份、time dtype与sensor provenance、来源与预算描述 | 55定向测试及真实style-0双branch oracle复核；此scope无阻断问题 |

所附 reviewer 报告保留初审缺陷与后续关闭记录；旧行号/旧复现输出用于解释修复，不代表最终代码仍有这些缺陷。`<local-validation-workspace>` 为当次执行 fixture 路径，不作为可下载数据集。可重新运行仓库测试与README命令复现最终行为。

Samsung `.001` 平坦输入（identity backend）与 `x*2, capture_bias_ev=1` 现在都输出`.001`；intent=1输出`.002`。真实style-0的251项state entry与原checkpoint一致，冻结且eval；最终黑端点/像素单调仍不作承诺。AE full/image-history 默认均53,881参数。gain-only历史变化经曝光归一化后不增加physics motion penalty；真实场景变化仍有正penalty。

## 完整训练记录

两条分支的24-scene 64px CPU复验已完成：14 train / 4 val / 6 test，12候选，3噪声重复，20 epochs，seed=0，真实同一style-0。full/image-only均为53,881参数；除use_auxiliary外，model kwargs、state shape、动作、数据、renderer、teacher协议、随机种子与训练预算相同。四次训练均按val选择epoch1，未按test结果重选。

| TM | 方法 | Composite cost | Regret | Display MSE | 新模拟RAW饱和 | 曝光+读出 ms |
|---|---|---:|---:|---:|---:|---:|
| Apple | 固定曝光 | 0.00220747 | 0.00046055 | 0.00013055 | 0.0100143 | 10.333 |
| Apple | full / image-only | 0.00175673 | 0.00000981 | 0.00018416 | 0.0018989 | 6.167 |
| Apple | 当前采样oracle | 0.00174692 | 0 | 0.00015911 | 0.0046161 | 7.556 |
| Samsung | 固定曝光 | 0.00157288 | 0.00001167 | 0.00008897 | 0.0100143 | 10.333 |
| Samsung | full / image-only | 0.00171569 | 0.00015449 | 0.00017865 | 0.0018989 | 6.167 |
| Samsung | 当前采样oracle | 0.00156121 | 0 | 0.00010181 | 0.0086760 | 9.639 |

**这轮没有外部metadata收益证据。** 两条分支的full与image-only均在全部6个test场景选择action0（1/240 s、ga=gd=1），逐场景动作和全部汇总指标完全相同。Apple的复合成本低于固定曝光，但display MSE更高；Samsung的学习策略在复合成本及MSE上均未超过固定曝光。因此不能将这些数字概括为整体画质提升，更不能宣称真机改进。

两模式observations-only CLI infer均通过：NPZ仅含previews `[3,3,64,64]` 和实际state `[3,3]`，无teacher/candidate输入；observation7→request8→effective9，delay1。曝光+readout为0.00616667 s，在0.04 s剩余预算内，并满足0.025 s midpoint offset的时序检查。Root再次严格载入四个checkpoint、统计实际parameters并核对test指标、协议与budget结果。

完整指标、容量与逐场景检查见 [validation_results.json](validation_results.json)。这些结果只验证合成场景的软件流程与评估行为，不可推广为真实手机画质收益。当前teacher确实对两种TM给出不同最优采集分布，但当前策略尚未稳定学出这种差异；真实数据、较丰富采集状态/光照/运动覆盖及预先确定的多seed实验是后续验证条件。

## 必须保留的证据边界

默认sensor工程假设、RGB而非native Bayer、identity颜色变换只适用proxy；实际camera WB/CCM和噪声参数须标定。DNG path已实现且不fallback预览，但无真实Apple ProRAW fixture。RawGen/RL-3A源的原始饱和/残余噪声只在provenance记录，不能被降低模拟曝光恢复；raw_saturation指标只计新模拟饱和。

teacher成本、噪声repeats、共同future midpoint和固定renderer限定了当前oracle；仍缺跨camera真实holdout、完整多帧/延迟闭环和GPU训练实验。现有非对齐PGT参考只能作appearance线索，不作pixel-aligned RAW/AE真值。stage分支尚未在多阶段任务训练，容量公平消融不等于已经证明外部metadata有收益。
