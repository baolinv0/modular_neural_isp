# AE–TM 半自动科研工作流

**从[START_HERE.md](START_HERE.md)开始。当前R0待执行；不是已经完成GPU实验。**

本目录将联网决策模型、GPU侧执行Agent和GitHub交接组织成最小可执行流程。完整文件已展开入库，拉取分支即可读取和运行，不需额外下载ZIP或应用patch。

| 内容 | 入口 |
|---|---|
| 当前状态和下一任务 | [START_HERE.md](START_HERE.md) |
| 研究与执行规则 | [AGENTS.md](AGENTS.md) |
| 研究问题、竞争解释、基线、证据 | [RESEARCH.md](RESEARCH.md) |
| 年度与12–20周计划 | [ROADMAP.md](ROADMAP.md) |
| 首轮软件实施计划 | [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) |
| 拉取、测试、预检、回传和提交命令 | [RUNBOOK.md](RUNBOOK.md) |
| 原独立文件名的运行手册 | [AE_TM_GitHub_Workflow_Runbook_20261007.md](../../docs/capture_tm/AE_TM_GitHub_Workflow_Runbook_20261007.md) |
| R0环境与数据预检 | [任务卡](tasks/R0_environment_and_data.md) / [命令卡](experiments/R0_preflight.json) |
| 可选历史toy链路 | [命令卡](experiments/R0_legacy_smoke.json) |
| R1 CUDA接入 | [任务卡](tasks/R1_cuda_runtime.md) |
| R2 S24数值域接入 | [任务卡](tasks/R2_s24_domain.md) |
| R3机制诊断 | [任务卡](tasks/R3_mechanism.md) |
| GPU执行者提示词 | [EXECUTOR.md](prompts/EXECUTOR.md) |
| 联网决策者提示词 | [DECIDER.md](prompts/DECIDER.md) |
| 论文骨架 | [outline.md](paper/outline.md) |
| 执行和回传工具 | [bridge.py](tools/bridge.py) / [preflight.py](tools/preflight.py) |
| 测试与验证边界 | [test_bridge.py](tests/test_bridge.py) / [VERIFICATION.md](VERIFICATION.md) |
| R0报告位置 | [reports/R0](reports/R0/README.md) |

当前只执行R0。数据、权重、完整日志、密钥与未获准公开的公司材料保留在GPU端，不上传仓库。
