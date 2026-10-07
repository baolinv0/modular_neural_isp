# AE–TM 科研工作流：唯一入口

**更新日期：2026-10-07。阶段：R0，尚未取得本轮GPU/真实S24实验回传。**

## 目标

一年内形成本人主导代表作；当前只做一轮可信的AE–TM研究闭环，不建设通用自动科研平台。
联网ChatGPT负责问题、实验设计、初版代码与证据审核；GPU侧公司模型/Qwen在coding-agent宿主中执行；用户保留数据许可、资源与最终研究主张的决定权。

## 先读哪四份内容

1. [AGENTS.md](AGENTS.md)：必须遵守的研究和执行边界。
2. [RESEARCH.md](RESEARCH.md)：问题、竞争假设、基线和可能的否定结论。
3. [tasks/R0_environment_and_data.md](tasks/R0_environment_and_data.md)：当前任务，先不要越级跑R3。
4. [RUNBOOK.md](RUNBOOK.md)：一次GitHub往返和命令。

## 本轮源代码

仓库：`baolinv0/modular_neural_isp`。
原分支：`feature/capture-tm-c-20261005`。
已读取的提交：`49e158ac663498a622107dd2f306f38d8f1a47a9`。
协作分支：`research/ae-tm-loop-20261007`。工作流随分支版本管理；GPU端按RUNBOOK拉取分支，不再手工解压ZIP或应用旧patch。

## 当前事实

- 代码已有两套AE–TM实验和共享物理采集入口，历史结果不构成新的S24证据。
- 已读的joint_cli无`--device`；joint_experiment主要为CPU路径，不能靠设置CUDA_VISIBLE_DEVICES获得CUDA训练。
- 当前learned TM为原模型冻结系数网络上的小适配器，不等于充分训练的强TM基线。
- 已读的S24计划仍明确标注导入、逐场景WB/CCM、style-0 GT、GPU/batch等待完成。
- 本包新增工具的本地测试不等于原仓库全量测试通过，也不是GPU或成像质量验证。

## 本轮决策 D000

优先完成R0预检并回传。R1/CUDA与R2/S24在R0结果基础上细化，不改变研究问题和正式test。先不新建TM架构、不增加语义或多帧模块、不打开全量搜索。

## 下一次交给ChatGPT的内容

分支链接 + 代码提交 + `reports/R0/attempt-001/reply.md`。ChatGPT需主动读取对应任务、summary、代码diff与必要证据，不能只根据本文件的摘要作结论。

## 待更新

本地实际GPU型号/可用显存、模型服务方式、S24根目录检查、数据与代码的可公开范围。私人路径、密钥与公司材料不写入公开仓库。

## 其他入口

- [完整文件导航](README.md)
- [年度目标与12–20周研究安排](ROADMAP.md)
- [首轮实施计划](IMPLEMENTATION_PLAN.md)
- [可执行命令与回传流程](RUNBOOK.md)
- [GPU执行者提示词](prompts/EXECUTOR.md)
- [验证范围](VERIFICATION.md)
