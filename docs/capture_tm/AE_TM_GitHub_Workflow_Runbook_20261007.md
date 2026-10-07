# 一次可执行的GitHub往返

## 1. 实际架构

用户：决定研究价值、资源许可与最终结论。
ChatGPT：读取GitHub证据→研究判断→给下一任务和初版代码。
GPU coding-agent：读取任务→编辑/测试/运行→报告/提交。
Python程序：实际训练、确定性评价和运行记录。
GitHub：代码、小型证据、当前状态，不承载原始数据/权重。

本对话没有GPU访问，GitHub提交也不会自动唤醒ChatGPT。每轮由用户贴出分支/提交/报告入口后触发下一次读取。当前以研究分支中的文件为准；ZIP/patch是此前的离线交付，拉取本分支后不再重复应用。

## 2. GPU端执行器

先用公司的现成coding-agent，不因本文重建ARIS/Naive。只有模型聊天API还不够，需要宿主提供文件读写、shell、进程/日志与git能力。

若没有合适宿主，Qwen Code官方文档支持自托管/公司兼容OpenAI接口。可在GPU机器上已有的Qwen Code环境使用：

```bash
export OPENAI_BASE_URL="http://你的内部服务地址:端口/v1"
export OPENAI_MODEL="你的实际模型ID"
# 通过本机环境/密钥管理设置OPENAI_API_KEY，不能提交到仓库。
qwen
```

这里的协议名不表示请求必须发送到OpenAI。上面地址与模型ID必须由本机实际服务提供；本包没有部署或验证该服务。不要求GPU访问ChatGPT/Claude。

官方依据：
https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/
https://qwenlm.github.io/qwen-code-docs/en/users/configuration/model-providers/

先验证小任务：读取一段文件→编辑测试文件→运行命令→读取退出码→汇报失败。只会生成代码但不能真实调用工具的模型，不能独立承担执行角色。不要为了挑选最强模型延误首轮。

## 3. 拉取研究分支（无需再应用补丁）

仓库：`baolinv0/modular_neural_isp`。研究分支：`research/ae-tm-loop-20261007`。
源代码基线仍为`49e158ac663498a622107dd2f306f38d8f1a47a9`；工作流以研究分支当前提交为准。

已有本地仓库时，在仓库根目录执行。若`git status --short`非空，先停下保留并处理自己的修改；不要reset、强制切分支或自动stash。

```bash
git status --short
git fetch origin research/ae-tm-loop-20261007
# 已有同名本地分支时只切换；没有时创建跟踪分支。
if git show-ref --verify --quiet refs/heads/research/ae-tm-loop-20261007; then
  git switch research/ae-tm-loop-20261007
else
  git switch --track -c research/ae-tm-loop-20261007 origin/research/ae-tm-loop-20261007
fi
git pull --ff-only origin research/ae-tm-loop-20261007
python -m unittest discover -s research/ae_tm/tests -v
```

尚未克隆时，可在没有同名目录的位置执行：

```bash
git clone --branch research/ae-tm-loop-20261007 --single-branch   https://github.com/baolinv0/modular_neural_isp.git
cd modular_neural_isp
python -m unittest discover -s research/ae_tm/tests -v
```

测试工具只依赖Python标准库；原训练环境另行检查，不要在R0照抄历史CPU版PyTorch安装命令覆盖现有CUDA环境。

先读取以下完整路径，避免误读仓库根目录的同名文件：

```text
research/ae_tm/START_HERE.md
research/ae_tm/AGENTS.md
research/ae_tm/RESEARCH.md
research/ae_tm/tasks/R0_environment_and_data.md
research/ae_tm/RUNBOOK.md
```

既有AGENTS.md/CLAUDE.md/QWEN.md不覆盖。根级AGENTS只提供本研究的入口；首次启动可将`research/ae_tm/prompts/EXECUTOR.md`交给模型。发生快进失败或分支冲突时报告并保留工作，不force push。

## 4. 运行R0：预检，不训练

S24_DATA_ROOT必须指向包含train、val、test目录的数据根，不能把数据复制进GitHub。

```bash
export S24_DATA_ROOT="/你的S24数据根目录"
python research/ae_tm/tools/bridge.py run research/ae_tm/experiments/R0_preflight.json   --output runs/ae_tm_bridge/R0-attempt-001
python research/ae_tm/tools/bridge.py packet runs/ae_tm_bridge/R0-attempt-001   --output research/ae_tm/reports/R0/attempt-001
```

相同输出目录再次运行会拒绝，以免覆写失败证据。再次尝试使用attempt-002。run记录保留在GPU本地；packet只拷贝小型摘要，不拷贝环境变量、完整日志或命令行参数。提交前仍须检查摘要是否包含内部信息。

预检程序退出0只代表“检查完成”。summary中的CUDA训练/S24训练就绪仍为false，需要执行R1/R2后才可判定。未找到数据时会显式blocked，不会自动造数据继续。

## 5. 可选：复现历史toy链路

只有现有依赖与style-0权重可用，并且已经准备好原代码CPU环境时，才单独执行：

```bash
python research/ae_tm/tools/bridge.py run research/ae_tm/experiments/R0_legacy_smoke.json   --output runs/ae_tm_bridge/R0-legacy-attempt-001
```

这张命令卡调用当前README确实存在的demo、validate、joint_cli入口。12场景16像素、1个seed、1epoch仅为软件smoke；可能用到历史生成协议自带test分区，不是新研究正式test，不得据此调新研究或声称画质收益。依赖/模型不满足就回传失败，不使用随机权重冒充原模型。

## 6. 回传GitHub

执行者填写reply.md中的事实/解释/阻碍，必要时增加最小错误复现或授权的图表。确认公开仓库可以接收这些文件。

```bash
git status --short
git add research/ae_tm/reports/R0/attempt-001 research/ae_tm/START_HERE.md
git commit -m "research: report R0 environment and S24 readiness"
git push -u origin research/ae-tm-loop-20261007
```

涉及代码修改时显式添加实际修改的文件，不用`git add .`。完整run/cache/data/weights不入Git。本包没有自动commit/push脚本，也不保存凭据。

随后给ChatGPT：仓库分支链接、提交ID、`reports/R0/attempt-001/reply.md`路径，使用DECIDER提示词。ChatGPT读取证据后给下一张任务；无需复制整个日志或聊天历史。

## 7. 四卡使用建议（先核对实际型号与服务位置）

按4×96GB假设：若公司模型由独立服务提供，四卡可用于实验；若LLM本地占卡，先预留GPU0给能在单卡装下的执行模型，GPU1做集成验证，GPU2/3跑独立对照。较大LLM需多卡时必须重算剩余资源，不承诺单卡能装任意模型。

R1完成后，每个进程先只用一个可用GPU，例如`CUDA_VISIBLE_DEVICES=1`；但可见卡设置不负责把CPU模型搬到CUDA。等CLI真实支持device并经过测试再传对应参数。初期不做DDP，避免四张卡只加快错误实验。

同一时间只让一个集成agent写研究分支。并行实验读同一代码快照、写不同输出目录。需要并行实现时用独立branch/worktree，按PR合并；不要一开始就开六个相互争改文件的agent。

## 8. 首轮成功的定义

一次闭环成功 = 用户知道什么可运行/什么阻塞；线上决策与线下动作一致；所有结果可追溯；下一任务由实证决定。

不是必须涨分，也不是一定开始全量GPU训练。后续科研成功还需要强基线、真实证据、可证伪的新贡献与外部review。

## 9. 文档维护

本文件是运行命令的维护入口。[研究目录运行手册](../../research/ae_tm/RUNBOOK.md)保存本次上传的同步副本；后续修改运行命令时同步两者，防止入口分叉。原始离线patch无需再提交或执行。年度研究目标见[ROADMAP.md](../../research/ae_tm/ROADMAP.md)，R1–R3任务中待实现的接口不能当作现有命令执行。
