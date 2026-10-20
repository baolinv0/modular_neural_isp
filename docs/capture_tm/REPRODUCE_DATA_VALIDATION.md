# 数据复现文档：本轮实测记录

2026-10-05。对应 [复现手册](REPRODUCE_DATA.md)。审查/执行时算法代码为 `60901974df9424fdcb1d48a6cc4fb565a007ebad`；本次提交只增加文档、入口、重组 ZIP 的忽略规则及修正分片清单，**不改变采集、AE、HDR 融合、TM 或训练目标**。

环境：Python 3.12.14、PyTorch 2.5.1+cpu、NumPy 1.26.4、OpenEXR 3.3.3；CPU 单线程生成/验证/训练。以下是本轮实际执行的证据，与先前的全仓 314 项回归记录分开。

## 已执行的检查

| 检查 | 结果与范围 |
|---|---|
| Git 分片清单 | 发现 70 个路径漏 `.zip`；仅修正路径，全部分片字节数与原清单相同 |
| 真实分片重组 | `python artifacts/reconstruct_archive.py` 成功；41,759,233 字节，原 SHA-256 匹配 |
| ZIP 完整性 | `python -m zipfile -t ...zip` 通过，实际解压成功 |
| 旧 RGB checkpoint 推理 | 当前代码载入原 seed0 A11/S11，在 `joint_0032`、采集 seed7 输出有限 `[3,32,32]`；AE 选择 11/9，分别 1/3 次 RGB 采集；`acquisition=None` |
| 物理与接口专项测试 | 111 passed in 18.40s，无失败或跳过；命令见下 |
| 新 Bayer 数据 | 12 场景、16×16；4/4/4 split；Apple 12 候选、Samsung 24 计划；默认 RGGB、rolling=0、readout=1 ms；噪声种子 0 |
| RAW 独立重算 | 从导出 DN 重算 RGB/F0/前端；`valid=true`、`raw_composition_verified=true` |
| 新数据八组训练 | seed0；warmup=1、epochs=1；每个活动模块正式 4 次更新；全部输出所选 checkpoint 与评测 |
| 新 Bayer checkpoint 推理 | 首个测试场景 `joint_0008`、采集 seed37：A11 选择9、1帧 `[1,16,16]` RAW；S11 选择11、3帧；输出均有限 `[3,16,16]` |
| 包内张量 | Apple RAW `[1,12,1,1,16,16]`；Samsung RAW `[1,24,3,1,16,16]`；目标均 `[3,16,16]` |

专项测试的实际命令：

```bash
python -m pytest -q tests/test_capture_tm_acquisition.py \
  tests/test_capture_tm_pipeline.py tests/test_capture_tm_pipeline_sources.py \
  tests/test_capture_tm_learned_tone.py
```

新数据生成、验证、训练和推理使用手册路径 B 的相同参数，只有输出根目录换成全新的执行目录。旧模型同时经独立 reviewer 的临时提取推理与主执行者完整解压推理核对。

## 八组重跑数值

| 组 | 本次测试 J | 相对原 Bayer 小实验记录的差值 |
|---|---:|---:|
| A00 | 0.0564294655341655 | 0 |
| A10 | 0.0959255067864433 | 0 |
| A01 | 0.056090257596224546 | 0 |
| A11 | 0.09559432393871248 | 0 |
| S00 | 0.05383792473003268 | 0 |
| S10 | 0.05221378384158015 | 0 |
| S01 | 0.05352389079052955 | 0 |
| S11 | 0.05190749582834542 | 0 |

比较对象是 [data_pipeline_validation_results.json](data_pipeline_validation_results.json) 的八个组均值。本环境下数值逐项相等；不据此保证跨平台、跨依赖版本、所有中间张量或 checkpoint 文件逐字节一致。这是一轮训练、四个测试场景的小实验；Apple 学习组退化仍然存在，不是画质验收通过。

## 审查后澄清

物理合同与复现路径由两路独立只读审查核对。文档显式区分：

- 公式里的曝光起点与 manifest 的曝光中点，rolling 行偏移与尾部 readout。
- 固定 GT 在光学/像元面积后、CFA 前；不承诺源视频本身无模糊/噪声或充分时间采样。
- virtual gain 不改变采集；实际增益不增加光子；工程默认参数不是设备标定。
- 旧 RGB 24 模型与新 Bayer 数据；scene v1 manifest 与采集包 v2 manifest。
- 不存在的 visibility GT、未实现的完整主观/鬼影指标，以及至少两个噪声重复才有输出方差。
- 外部 recipe 的素材占位路径和 train-only 示例，不能冒充可直接运行的完整训练数据。

## 未在本轮执行或证明

未重新跑旧 RGB 40 场景的 24 组完整训练；未跑扩大到 40 场景的 Bayer 配置；未取得新的外部 HDR/真实手机素材；未重跑完整 314 项套件；未完成真机标定、持续 AE、非零 rolling 的真实场景画质实验或 visibility/遮挡验证。现有专项测试对相关软件分支的覆盖，不能替代这些数据与设备证据。

本轮结论是：**已发布实验包能按校正说明恢复，最小 Bayer 数据—训练—推理路径可按文档复现。** 下一轮研究仍须解决旧实验揭示的策略集中、饱和与外观目标冲突。
