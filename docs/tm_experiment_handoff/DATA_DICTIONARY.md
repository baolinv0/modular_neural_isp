# CSV 数据字典

CSV 用 UTF-8（含 BOM）便于 Windows/Excel 打开；小数点用句点，比例用 0–1，不用百分数文本。
原始 run_id/sample_id 一律当字符串，不自动转数字或日期。数值缺失留空，原因写 missing_items.csv；不得把缺失当 0。
run_registry 每行是一种确定的 checkpoint+评估协议，eval_id 唯一；同一 run_id 可以有多种 eval_id。
dataset_manifest 的主键是 dataset_id+sample_id，其余指标通过 eval_id 找到数据集。每个 eval_id+sample_id 必须唯一。
region_metrics 的键再加 region+roi_definition_id；case_index 每个案例每个 eval_id 一行。
JSON 字段填合法 JSON，逗号/引号由 CSV writer 正确转义。路径相对回传根目录；未打包的大文件写明确可获取位置。
表头不能擅自删除/改名；可以追加列，并在报告说明。表格无默认数值，全部由实际结果填写。

## run_registry.csv

| 字段 | 说明 |
| --- | --- |
| `eval_id` | 主键；一次确定 checkpoint+split+protocol 的评估身份 |
| `run_id` | 训练身份；同一训练的多次评估复用 |
| `method` | baseline/A/B/C/D/E 或明确组合名 |
| `base_algorithm` | 实际 base_algorithm / algorithm |
| `correction_algorithm` | none/region_curves/gain_residual |
| `semantic_mode` | S0/S1/S2；与实际配置对应 |
| `seed` | 实际训练 seed；未知留空并说明 |
| `dataset_id` | 与数据清单对应的版本 ID |
| `split` | train/val/test；不能将 val 改名 test |
| `protocol_id` | 与 protocols 中完整协议 JSON 对应 |
| `git_commit` | 实际代码 commit |
| `dirty_diff_path` | 未提交改动 patch；干净树可留空 |
| `config_path` | 最终有效配置；含覆盖参数和各 loss |
| `train_log_path` | 原始日志位置 |
| `command_log_path` | 实际训练与评估命令位置 |
| `checkpoint_path` | best/last 等实际评估权重位置 |
| `checkpoint_sha256` | 评估权重完整 SHA256 |
| `checkpoint_epoch` | 被评估权重实际 epoch |
| `checkpoint_global_step` | 被评估权重 optimizer step |
| `selection_split` | 选择该 checkpoint 使用的数据集合 |
| `selection_metric` | 选择指标及 max/min 方向 |
| `initial_backbone_sha256` | 初始公共 backbone 权重身份 |
| `freeze_backbone` | true/false；详细模块冻结清单见 config |
| `post_mode` | adaptive/fixed_reference/other |
| `epochs_completed` | 实际完成 epoch |
| `optimizer_steps_completed` | 实际 optimizer 更新次数 |
| `effective_batch_size` | 含累积的有效 batch；DDP 说明见配置 |
| `train_crop_hw` | 实际 crop 如 512x512 |
| `analysis_size` | 实际条件分析分辨率 |
| `evaluation_hw` | 原尺寸/固定HxW/多尺寸 |
| `segmentation_config_path` | S2 或 S1 训练分割配置；无则留空 |
| `segmentation_weights_sha256` | 外部分割器权重身份 |
| `execution_status` | completed/partial/failed/not_run |
| `notes` | 预算、baseline 预训练/重训、resume、异常等 |

## dataset_manifest.csv

| 字段 | 说明 |
| --- | --- |
| `dataset_id` | 数据集和版本 |
| `sample_id` | 稳定 ID，跨方法不变 |
| `scene_id` | 同场景分组，不能统一填 unknown |
| `subject_id` | 可用时填匿名主体 ID；未知留空 |
| `burst_id` | 可用时填连拍分组；未知留空 |
| `split` | train/val/test |
| `camera` | 真实拍摄设备/相机 |
| `scenario_tags` | 多标签以分号分隔；未知可填 unknown |
| `input_path` | 输入文件位置 |
| `input_sha256` | 输入文件 hash，至少评估集 |
| `target_path` | 对齐 GT 位置 |
| `target_sha256` | GT 文件 hash |
| `target_type` | paired_render/camera_jpeg/pseudo_gt/other |
| `height` | 评估图像高 |
| `width` | 评估图像宽 |
| `valid_mask_path` | 忽略无效/错位像素用的 mask；无则留空 |
| `roi_path` | 共用的可信区域 mask；可选 |
| `roi_source` | 人工/冻结模型/其他，区分 GT 与伪标签 |
| `alignment_status` | verified/partial/unaligned/unknown |
| `notes` | 对齐/曝光等异常，不事后静默删除 |

## training_history.csv

| 字段 | 说明 |
| --- | --- |
| `run_id` | 训练身份 |
| `record_index` | 该日志记录顺序号 |
| `epoch` | 可为原始小数 epoch |
| `global_step` | 原始 optimizer step |
| `phase` | train/val |
| `loss_total` | 当时定义的总 loss，不能跨不同权重直接比 |
| `loss_components_json` | 所有 loss 分量的 JSON 对象，CSV 正确转义 |
| `lr` | 主参数组 LR；多组见 extra_json |
| `psnr` | 该次实际评估 PSNR，没测留空 |
| `ssim` | 该次实际评估 SSIM，没测留空 |
| `elapsed_seconds` | 自本次训练开始/恢复计时的秒数，口径写明 |
| `event` | resume/oom/nan/best_saved/正常可留空 |
| `extra_json` | 其他原始记录和多组 LR 的 JSON 对象 |

## per_image_metrics.csv

| 字段 | 说明 |
| --- | --- |
| `eval_id` | 对应 run_registry 中评估身份 |
| `sample_id` | 对应 dataset_manifest 中 sample_id |
| `status` | ok/failed/missing/nonfinite；失败仍保留一行 |
| `failure_reason` | 失败理由；成功留空 |
| `psnr` | [0,1] sRGB 的逐图 dB；实际口径见 protocol |
| `ssim` | 实际 RGB SSIM；不可随意混用不同实现 |
| `luma_l1` | 显示 sRGB code values 的 Rec709 加权误差，可选 |
| `delta_e76` | D65 Lab ΔE76，可选，不能填 ΔE00 |
| `chroma_l1` | 实际定义的 Lab chroma 误差，可选 |
| `hue_degrees` | 有效像素最短 hue 角误差，可选 |
| `hue_valid_fraction` | hue 指标有效像素比例，范围0–1 |
| `output_endpoint_fraction` | 最终显示输出端点通道占比0–1，可选 |
| `linear_below_zero_fraction` | chroma 前线性阶段通道越界率，可选 |
| `linear_above_one_fraction` | chroma 前线性阶段通道越界率，可选 |
| `prediction_path` | 实际预测文件位置 |
| `raw_metrics_path` | 对应原始 JSON 结果位置 |

## region_metrics.csv

| 字段 | 说明 |
| --- | --- |
| `eval_id` | 评估身份 |
| `sample_id` | 稳定图像 ID |
| `region` | face/skin/person/sky/background/highlight/other |
| `roi_definition_id` | 固定区域定义和 mask 身份 |
| `status` | ok/absent/invalid/too_small/unknown |
| `area_fraction` | 区域有效像素面积比例0–1；软 mask 规则另注明 |
| `weight_sum` | 实际计算使用的权重之和 |
| `psnr` | 同一有效 ROI 的加权 RGB MSE 转 PSNR |
| `luma_l1` | 同一 ROI 加权误差 |
| `delta_e76` | 区域 Lab ΔE76，可选 |
| `pred_mean_luma` | 预测区域 sRGB code-value luma 均值 |
| `target_mean_luma` | GT 区域对应均值 |
| `pred_mean_Lstar` | 预测区域 Lab L* 均值 |
| `target_mean_Lstar` | GT 区域 Lab L* 均值 |
| `pred_mean_chroma` | 预测区域 C* 均值 |
| `target_mean_chroma` | GT 区域 C* 均值 |
| `notes` | ROI来源/权重/面积阈值在协议中；异常在此补充 |

## runtime_samples.csv

| 字段 | 说明 |
| --- | --- |
| `eval_id` | 评估身份 |
| `measurement_id` | 单次计时稳定 ID |
| `scope` | tm_only/segmentation_plus_tm/local_operator_only/other |
| `sample_id` | 实际输入样本 ID |
| `device` | 设备完整型号 |
| `height` | 输入高 |
| `width` | 输入宽 |
| `batch_size` | 实际 batch |
| `dtype` | fp32/fp16/bf16/int8 等 |
| `compile_mode` | eager/编译器版本和配置 ID |
| `warmup_count` | 该组实际预热次数 |
| `repeat_index` | 本组重复计时序号 |
| `latency_ms` | 本次同步计时耗时；非平均数 |
| `peak_allocated_mib` | 该组测量峰值 allocated MiB，可选；范围在协议写清 |
| `peak_reserved_mib` | 该组测量峰值 reserved MiB，可选 |
| `status` | ok/oom/failed |
| `notes` | 同步、内存重置和计时范围的补充 |

## case_index.csv

| 字段 | 说明 |
| --- | --- |
| `case_id` | 案例 ID；同一案例各方法保持一致 |
| `sample_id` | 稳定图像 ID |
| `eval_id` | 该行对应模型/评估；每案例每方法一行 |
| `selected_by` | fixed_random/worst_delta/best_delta/known_issue |
| `selection_rule` | 排序指标、随机种子、选择哪一方法的结果等 |
| `input_display_path` | 统一显示输入图路径 |
| `target_display_path` | 对齐 GT 显示图路径 |
| `prediction_path` | 该方法原尺寸预测路径 |
| `float_prediction_path` | 导出前 float 显示输出，可选 |
| `roi_boxes_json` | 相同坐标的 ROI JSON；xyxy，左闭右开，以全尺寸像素计 |
| `semantics_path` | S2 软语义原始数组，可选 |
| `confidence_path` | S2 confidence 原始数组，可选 |
| `diagnostic_npz_path` | 该方法原始控制/中间图，可选 |
| `observation` | 可定位的改善/退化/不确定，不只写好/坏 |

## missing_items.csv

| 字段 | 说明 |
| --- | --- |
| `item` | 缺失项名称 |
| `affected_eval_ids` | 受影响 ID；分号分隔 |
| `status` | unknown/not_run/unavailable/not_applicable |
| `reason` | 原因 |
| `effect_on_analysis` | 限制哪些结论 |
| `owner_and_next_action` | 负责人和可补方式，无则说明 |

