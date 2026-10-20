# R2：S24数值域、目标与最小接入

状态：PLANNED；先读取R0数据报告；可与R1的独立代码部分并行，集成顺序由一名写入者负责。

## 依据与范围

沿用源提交docs/capture_tm/S24_EXPERIMENT_PLAN.md，但本轮缩小为：读取、配对、颜色前端、原模型复现与静态代理。不一次实现整个大规模HDR管线。

## 任务拆分

R2a：manifest/读取。按ID配对；uint16三通道正确解码；明确RGB/BGR；cam_illum/ccm逐场景处理；隐私mask；源裁切指示；保存官方split/source id。默认只用train/val。需报告完整解码失败和抽样范围，不把R0文件头检查叫完整检查。

R2b：原模型域核验。按原photofinishing入口处理，不能把新wrapper结果叫原网络逐算子复现；对noisy/denoised分别运行，检查输出范围、颜色、几何、style-0可比性。权重和固定目标分别记录；--weights不等于切换GT。

R2c：小型静态代理。先用8–16个train源和4–8个val源做smoke，固定geometry、WB/CCM、目标和mask。source裁切不是可恢复高光真值；全部变体保留源身份。扩展到64/16只在前两项通过后。

## 验收

单元测试覆盖：错配、重复ID、PNG8拒绝、颜色顺序、逐场景CCM、固定目标不会随候选改变、invalid mask稳定、正式test不加载。可视化同场景输入/参考/输出（获准公开时），记录误差与排除原因。

## 不允许

用单位CCM、重新归一化、解析GT、随机权重或生成图片悄悄填空；使用ISO/100代替标定模拟增益；把denoised数据当无噪原生HDR；用GT相关noise_stats/snr_stats进入部署AE输入。

## 资源

数据审计CPU优先；数值smoke使用一个可用GPU。任何全量数据生成先用pilot估算磁盘/时间，给下一轮决定。
