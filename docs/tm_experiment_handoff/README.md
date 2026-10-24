# 工程回传入口

先读 REPORT_TEMPLATE.md，并填写后重命名 REPORT.md。当前包仅是模板，未包含实验结果。

## 第一轮直接交什么

1. REPORT.md + run_registry.csv + dataset_manifest.csv + per_image_metrics.csv。
2. configs/：最终生效配置、命令、环境版本、dirty diff。
3. logs/：原始训练日志和评估 JSON；training_history.csv 有条件则整理。
4. cases/：建议20–30张互异样本的输入/对齐GT/baseline/A–E原尺寸输出、关键ROI，填 case_index.csv。
5. missing_items.csv：缺 baseline、单种子、无区域mask、无成本测量都如实写明；不要求先重训。

表格已经提供空表头，字段说明在 DATA_DICTIONARY.md。protocol_template.json 按真实情况填写后放 protocols/<protocol_id>.json。

## 已有则一并交

region_metrics.csv、runtime_samples.csv、diagnostic NPZ、全部评估预测、已有消融/多seed/盲评记录。

## 目录建议

- REPORT.md：实验主报告。
- *.csv：结构化结果。
- protocols/：实际指标/ROI/计时协议。
- configs/：每个run的最终配置、训练/评估命令、环境、diff。
- logs/：训练原始日志与评估原始JSON。
- cases/<sample_id>/：同样本输入、GT和六种输出；每种文件带eval_id。
- diagnostics/<eval_id>/：关键案例控制图，保留浮点数组。
- checkpoints/：可不直接打包，但登记可获取位置和SHA256，工程端保留原权重。

## 完成前核查

- eval_id、run_id、sample_id 对得上；相同ID不会因为文件名清理被混淆。
- 每张评估图每种模型有成功或失败记录；不静默删图，指标缺失不填0。
- 所有比较标明val/test、输出尺寸和协议；真实不同的实验不强行包装成公平消融。
- 列出的相对路径确实有文件；外部大文件位置说明清楚。
- 图片不逐格自动曝光或归一化；所有方法ROI用相同坐标。
- 发回报告和CSV即可先分析；大图包可分批发，仍保留同一份索引。

无需此轮重训、多种子复跑、端侧部署或额外大规模消融。若关键记录无法恢复，在报告中留明限制。
