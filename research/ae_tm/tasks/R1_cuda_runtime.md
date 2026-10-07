# R1：让现有训练真实使用GPU

状态：PLANNED；R0回传后细化，暂不自动执行。

## 只解决这个问题

原四组实验能否在CUDA前向、反向、保存/加载、评价中使用一致的设备与相同数值语义？不改变损失、模型结构、数据与目标。

## 预期修改范围

capture_tm/joint_cli.py、joint_experiment.py，以及实际发现的必要设备辅助函数/测试。避免大范围重构。API建议新增device='cpu'，保留默认向后兼容；CLI --device是待实现参数，不是现有可用参数。

## 必须覆盖的细节

policy和TM转移；缓存加载的nested tensor；features/legal masks/indices；teacher costs；CPU模拟器和GPU渲染的边界；loss/backward；checkpoint map_location；评价转numpy前detach().cpu()。不要仅新增一个未用到的CLI flag。

第一版缓存仍落CPU磁盘、按需搬运，设备移动不得保存全部场景在显存。AMP、DDP、batch结构优化和新调度器都不属于本任务。

## 测试与证据

先写CPU向后兼容/嵌套tensor搬运测试，再做CUDA小样本：前向、非零有限梯度、optimizer step、checkpoint reload、评价输出。记录实际parameters/input/output device，峰值显存与耗时，不能只给nvidia-smi截图。

同一输入CPU/FP32 CUDA比较中间张量和输出，先报告最大绝对/相对误差；初始参考容差atol=1e-4、rtol=1e-3，不得为通过而静默放宽；容差需结合算子和具体差异讨论。

正式研究尚未打开test。原程序会自动评测test：历史toy smoke可用于回归，但R3前须有明确development评价入口，测试该入口不加载正式test数据。不能将正式test改名val来规避规则。

## 预算

单个训练GPU，第一次只做少量scene和1个epoch的smoke，运行命令单次最多30分钟。上报实际耗时后再批准更大任务，不预先承诺完整四组能在30分钟内跑完。
