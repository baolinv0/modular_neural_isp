# AutoDL 环境与 CUDA 软件链验证

日期：2026-10-07；本轮代码提交：`56b719fe448f78762c548849db6b03bbfb51a54a`。

状态：GPU 软件链验证回传通过，实例已实际关机并二次只读回读确认。真实 S24 数据接入阻塞，尚未运行 S24 训练。关键实测值见 [evidence.json](evidence.json)。逐组完整 JSON 仍在实例数据盘，未完整复制回公开报告。

## 已核验事实

- 通过 Browser Use 的网页终端进入用户已有实例，没有使用 SSH、收集凭据、创建实例或扩容。
- `nvidia-smi`：RTX 4080 SUPER，32760 MiB，驱动 580.105.08。显示的 CUDA 13.0 是驱动能力，不是训练环境的 Torch runtime。
- 数据盘实测 50G，首次检查近乎空盘，与用户给出的 120G 不同。
- 原环境 Python 3.8.20、Torch 2.4.1+cu121、CUDA 可用；不满足项目当前 Python 语法。原环境保留，新建 Python 3.11 环境。
- 官方 Torch CUDA 12.4 源下载过慢，安装尝试被定向终止，退出 143；随后以清华 PyPI 镜像重试固定 Torch 2.5.1。新环境最终状态见 GPU 回传。
- 首次定时关机输入/回读为 2026-10-08 02:30；终端时区为 UTC+8。最终控制台定时栏却显示 2026-10-08 04:30、该弹窗当前服务器时间为 2026-10-07 10:24，时区未能独立核验。两次回读不一致，不能声称定时保护已可靠对应用户时区与预算。实际实例已关机，二次只读确认时没有开机、重启或修改定时安排。后续运行前须重新核对控制台时区和定时值，不能沿用本轮定时值。

## 数据阻塞

[作者训练分享](https://ln5.sync.com/4.0/dl/336801ec0) 包含 `train.z01`–`train.z14`，每卷显示 20GB，以及 7.8GB 的 `train.zip`，约 287.8GB。这是同一压缩包的分卷，不能把末卷当独立数据包。该浏览器页面没有单图、metadata、普通 href 或复制服务器下载链接入口。没有下载全部分卷，也没有完成任何 PNG/JSON 解码或 S24 配对审计。

已核对[作者数据页](https://sites.google.com/view/mafifi/data)及[官方数据说明](https://github.com/SamsungLabs/time-aware-awb/blob/main/s24-raw-srgb-dataset/README.md)，未找到可直接取得本轮所需独立样本的小包。此结论仅描述本轮检索，不断言其他镜像不存在。

继续 R2 所需输入：可独立下载的官方 train 8 源、val 4 源，保留来源 ID 和目录；每源包含 `raw_images`、`denoised_raw_images`、`data/*.json`、`srgb_images_style_0` 及对应 `blur_masks`（按官方实际存在情况保留）。不要包含正式 test，不用排序 zip 配对，不重新归一化，不以单位 CCM 或解析 GT 补缺。

## 代码与本地验证

- 新增 `--device`，默认 CPU；模拟器及冻结系数准备仍在 CPU，磁盘 cache 按 scene 递归迁移；模型、features、mask、indices、teacher costs、checkpoint reload 与 NumPy 出口贯通设备。
- 新增 `--evaluation-split val`；默认 test 入口保留以兼容历史实验。开发模式不解码正式 test，仍验证其可见 schema 与声明来源身份。不能在未读取 test 像素时声称跨全量 decoded hash 审计完成。
- 开发过滤保留原 manifest 场景序号，预览/采集 seed 与原协议相同。test-first 排序回归要求 cache tensor 精确相等。
- 本地 Python 3.12、Torch 2.7.1+cpu；最终完整 `python -m pytest -q --disable-warnings`：**331 passed, 9 skipped**，241.05s，退出 0。跳过项包含无 GPU 的 CUDA 验证和可选 OpenEXR 检查。
- 曾出现一项旧错误提示兼容失败，已恢复提示；独立审查提出的元数据验证遗漏与种子改变已先复现失败再修复，复审允许部署为 GPU smoke 待验版本。

## GPU 验证范围

授权下执行固定代码的完整测试，然后单独保存 `synthetic_software_smoke_NOT_S24`：16×16 合成数据，train 1 源、val 1 源、正式 test 0，Apple/Samsung 各四组，seed 0、noise seed 0、warmup 1、epochs 1、threads 1、candidate chunk 4、CUDA。显式关闭 matmul/cudnn TF32，保留 FP32。每条测试/训练命令限时 1800s。

数值对照覆盖两方案 frozen-coefficient render、policy scores、期望代价及 AE/TM 梯度；固定 `atol=1e-4, rtol=1e-3`，记录最大绝对误差和以 1e-8 为分母下限的相对误差，不静默放宽容差。设备、有限非零梯度、参数更新、checkpoint reload、PNG、耗时与峰值显存须实际回传后才算通过。

## GPU 实际回传

- 新环境 Python 3.11.17，Torch 2.5.1+cu124，CUDA runtime 12.4，RTX 4080 SUPER；matmul/cudnn TF32 均为 false。
- `gpu-verify.exit` 原文为 `0`。完整 pytest 原文：`335 passed, 5 skipped in 349.37s (0:05:49)`。CUDA 用例已执行，5 个可选 OpenEXR 用例跳过。
- 稳定合成 smoke 共 8 组，train 1、val 1、test 0，完整 cache + warmup + 四组训练/重载/评价共 58.636620968580246s。选模继续允许 epoch0，不能把程序成功解释为质量改善。
- GPU 开发回归对两方案各四组断言实际参数/输入/输出/重载设备，学习模块有限非零梯度、参数更新、optimizer step、PNG 和正峰值显存；具体逐组梯度范数和峰值字节在原始 `software-smoke/summary.json` 中，公开报告尚未复制这些数值，不补写。

| 对照 | Apple 最大绝对误差 | Samsung 最大绝对误差 |
|---|---:|---:|
| render | 2.384185791015625e-7 | 3.5762786865234375e-7 |
| policy scores | 0 | 0 |
| expected cost | 0 | 0 |
| AE gradients | 2.5647750589996576e-10 | 4.220055416226387e-10 |
| TM gradients | 7.450580596923828e-9 | 7.450580596923828e-9 |

完整最大相对误差已按原 `CUDA_PARITY` 两行抄录至 evidence.json，两个方案全部断言通过，未放宽容差。这是 16×16 指定输入与冻结系数的数值验证，不是全分辨率 S24 的误差保证。

执行器在首次完结回传时已关机，未按协调者“先回传完整证据再关机”的顺序返回。二次仅从保留终端 canvas 取得上面的退出码、pytest 原行、两行 parity 和 summary 顶层字段。完整 summary 为 8929 字节，关机后无法再次从服务器读取为可靠完整原文；未为复制日志重新启动计费实例。原始 `software-smoke/summary.json`、`software-smoke/run/results.json`、checkpoints、PNG 和 `gpu-verify.log` 保存在实例数据盘。

[控制台](https://www.autodl.com/console/instance/list)最终显示“已关机”，实例列表仅一条。公开报告省略私有实例、主机和访问凭据。

这只验证软件/GPU 链，不能用于 S24 质量、真实采集收益或论文结论。R2 的 PNG16/颜色/CCM/style-0 GT/mask 接入尚未实现或验收。
