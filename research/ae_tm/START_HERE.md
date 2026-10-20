# AE–TM 科研工作流：唯一入口

**更新日期：2026-10-07。阶段：R1 GPU软件smoke回传通过；R2/真实S24因数据接入阻塞。实例已关机。**

## 目标

一年内形成本人主导代表作；当前只做一轮可信的AE–TM研究闭环，不建设通用自动科研平台。
联网ChatGPT负责问题、实验设计、初版代码与证据审核；GPU侧公司模型/Qwen在coding-agent宿主中执行；用户保留数据许可、资源与最终研究主张的决定权。

## 先读哪四份内容

1. [AGENTS.md](AGENTS.md)：必须遵守的研究和执行边界。
2. [RESEARCH.md](RESEARCH.md)：问题、竞争假设、基线和可能的否定结论。
3. [tasks/R0_environment_and_data.md](tasks/R0_environment_and_data.md)：当前任务，先不要越级跑R3。
4. [RUNBOOK.md](RUNBOOK.md)：一次GitHub往返和命令。

本轮最新事实见 [AutoDL运行报告](reports/R1/autodl-smoke-20261007/reply.md)。报告区分已回传的测试结果、仍在实例数据盘的原始输出及真实S24阻塞。

## 本轮源代码

仓库：`baolinv0/modular_neural_isp`。
原分支：`feature/capture-tm-c-20261005`。
已读取的提交：`49e158ac663498a622107dd2f306f38d8f1a47a9`。
长期集成分支：`feature/capture-tm-c-20261005`。研究计划、代码、任务与小型证据统一在该分支版本管理；需要并行实现时才开临时分支/worktree。GPU端按RUNBOOK拉取本分支，不再手工解压ZIP或应用旧patch。

## 当前事实

- 代码已有两套AE–TM实验和共享物理采集入口，历史结果不构成新的S24证据。
- 当前joint_cli已支持`--device`与`--evaluation-split val`；模拟器/准备cache保持CPU，按scene搬运到训练设备。默认CPU/test入口保留作历史兼容；开发运行显式用val，不读取正式test像素。
- 当前learned TM为原模型冻结系数网络上的小适配器，不等于充分训练的强TM基线。
- 已读的S24计划仍明确标注导入、逐场景WB/CCM、style-0 GT、GPU/batch等待完成。
- 本包新增工具的本地测试不等于原仓库全量测试通过，也不是GPU或成像质量验证。
- 本轮本地完整pytest为331通过/9跳过；远端执行器回传335通过/5跳过、8组合成software smoke退出0。它们不构成真实S24训练或画质证据。
- GPU侧复用了用户既有实例，最终回报已关机；数据盘实际50G，官方train链接是约287.8GB的分卷包。没有完成S24下载、PNG16配对、颜色/目标/mask验收。

## 本轮决策 D000

本轮在用户明确授权的6小时小样本范围内完成了环境检查与R1软件smoke。下一步先解决独立S24 train/val样本获取并完成R2验收，不改变研究问题和正式test。先不新建TM架构、不增加语义或多帧模块、不打开全量搜索。

## 下一次交给ChatGPT的内容

分支链接 + 代码提交 + `reports/R0/attempt-001/reply.md`。ChatGPT需主动读取对应任务、summary、代码diff与必要证据，不能只根据本文件的摘要作结论。

## 待更新

GPU原始逐组/parity输出的公开回传（若尚未复制）、S24独立样本和R2验收。私人路径、密钥与公司材料不写入公开仓库。

## 其他入口

- [完整文件导航](README.md)
- [年度目标与12–20周研究安排](ROADMAP.md)
- [首轮实施计划](IMPLEMENTATION_PLAN.md)
- [可执行命令与回传流程](RUNBOOK.md)
- [GPU执行者提示词](prompts/EXECUTOR.md)
- [验证范围](VERIFICATION.md)
