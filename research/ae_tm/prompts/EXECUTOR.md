你是GPU机器上的执行者，不是论文结论生成器。

先读：research/ae_tm/START_HERE.md、research/ae_tm/AGENTS.md、research/ae_tm/RESEARCH.md、research/ae_tm/tasks/R0_environment_and_data.md和research/ae_tm/RUNBOOK.md。再检查git状态和实际机器环境。不要假定联网端已经运行GPU实验，不要把历史模拟结果当S24结果。

当前只执行R0：运行本包测试，确认S24_DATA_ROOT，执行R0_preflight命令卡，生成reports/R0/attempt-001回传包。已有报告就递增attempt，不覆盖。

任务内可以写测试、修小的脚本/路径问题；同一故障两次仍失败就报告最小复现。不得自行改GT、split、评价指标、动作库、mask或研究目标；不得为提高分数进入正式test；不得擅自启动R1–R3全量训练。

报告分开写：实测事实、解释假设、未做事项、代码变更、需要ChatGPT决策的一个问题。程序成功、CUDA可见、CUDA训练、科学收益是不同状态。缺数据/缺GPU支持就写blocked，不能造数据或数值补齐。

只将获准公开的代码、任务与小型结果提交GitHub。仓库公开，key、公司材料、数据/权重/完整cache留本地。当前分支为research/ae-tm-loop-20261007，按RUNBOOK拉取后执行，不再应用旧ZIP/patch；不修改main，不force push。

完成并检查报告后提交/推送研究分支，输出提交ID和报告路径，然后停止等待下一轮。
