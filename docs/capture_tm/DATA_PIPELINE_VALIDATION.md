# Bayer 数据 pipeline 验证记录

2026-10-05：完整回归 **314 passed in 289.51s**；12 场景数据生成、独立 RAW 重算验证、八组训练和两条真实训练 checkpoint 的模拟 RAW 推理均已完成。

这是软件和相对辐射模型的验证。没有真实手机标定，也没有据此证明 AE–TM 联合训练提高画质。既有实验的饱和与策略集中问题见 [JOINT_RESULTS.md](JOINT_RESULTS.md)，本次数据入口没有修改其目标函数或消除这些失败模式。

结构化结果见 [data_pipeline_validation_results.json](data_pipeline_validation_results.json)，使用说明见 [DATA_PIPELINE.md](DATA_PIPELINE.md)。

## 环境与回归

Python 3.12.14，PyTorch 2.5.1+cpu，NumPy 1.26.4，OpenEXR 3.3.3。使用仓库自带 ISP 权重，CPU 单线程执行数据和训练 pilot。

```bash
python -m pytest -q
```

314 个测试包含此前 195 个回归和新增 119 个测试。新增检查覆盖四种 Bayer 相位、光子数与模拟增益分工、饱和前时间积分、PSF 与像元面积处理、逐行曝光及读出、动态素材支持域、固定参考位置、跨 split 来源身份、相对路径迁移、随机种子稳定性、失败后的原子发布、数据包语义、八组训练和 checkpoint 绑定。

素材成功 fixture 包括 float NPY/NPZ、序列数组和 native OpenEXR；OpenEXR 检查保留大于 1 的线性值。float TIFF 的适配和拒绝检查已实现，但当前环境未执行真实 TIFF 成功 fixture。不会把该可选格式列为实测解码成功。

## 数据包与独立验证

```bash
python -m capture_tm.pipeline_cli demo \
  --output runs/capture_tm/raw-pipeline-demo --scenes 12 --size 16 \
  --scheme both --noise-seeds 0 --seed 2026 --threads 1
python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/raw-pipeline-demo/acquisition/manifest.json \
  --output runs/capture_tm/raw-pipeline-demo/validation.json
```

| 检查项 | 实际结果 |
|---|---|
| 场景划分 | 4 train / 4 val / 4 test；每组含暗静态、暗运动、背光、普通条件 |
| Apple | 12 个单次曝光候选，共 12 个场景记录 |
| Samsung | 24 个顺序三帧候选，共 12 个场景记录 |
| 原生观测 | 12-bit RGGB DN，16×16；每候选一次噪声重复 |
| 参考与目标 | 两方案共用 t=0 光学/像元面积处理后的参考及独立固定外观目标 |
| 采集调度 | 全部行曝光与 1ms readout 满足采集窗口和非重叠约束；历史预览先完成 |
| RAW 重算 | 独立从所有导出 DN 重建 RGB/F0，核对 images、reliability、missing、radiance_mse；通过 |
| 标定状态 | assumed_engineering；相对辐射和全局标量噪声 |

RAW 重算不重新采集源场景，也不调用当前学习 TM 生成目标。训练 loader 检查 schema 和语义；独立 validator 才执行完整 RAW 重算。这两个检查的范围不同。

## 八组训练与推理

```bash
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/raw-pipeline-demo/acquisition/manifest.json \
  --output runs/capture_tm/raw-pipeline-study --scheme both \
  --epochs 1 --warmup 1 --seeds 0 --noise-seeds 0 \
  --threads 1 --prepare-threads 1 --candidate-chunk-size 4
```

所有组直接读取归档候选，训练时不重新采集或融合。回归测试将这些函数替换为抛错函数，验证了该接口隔离；也检查冻结模块参数保持不变、活动模块获得更新。实际 pilot 的八个 selected checkpoint 均与数据包中的 sensor、acquisition 和有序 plans 完全一致。

| 组别 | 正式 AE/TM 更新次数 | 验证选择 epoch | 测试 J |
|---|---:|---:|---:|
| A00 | 0 / 0 | 0 | 0.056429 |
| A10 | 4 / 0 | 0 | 0.095926 |
| A01 | 0 / 4 | 1 | 0.056090 |
| A11 | 4 / 4 | 1 | 0.095594 |
| S00 | 0 / 0 | 0 | 0.053838 |
| S10 | 4 / 0 | 0 | 0.052214 |
| S01 | 0 / 4 | 1 | 0.053524 |
| S11 | 4 / 4 | 1 | 0.051907 |

更新次数不含共同 AE 预热。J 沿用既有目标，越低越好。这是一训练种子、一个 epoch、四测试场景的链路 pilot；不能用它作统计画质结论。Apple 的学习 AE/联合组明显差于基线，不能隐去。

实际载入 A11/S11 的 `selected.pt`，在固定首个测试场景 `joint_0008`、采集 seed=37 上执行 `run_simulated(noisy=True)`：A11 选择候选 9，仅采集一帧；S11 选择候选 11，仅采集三帧。每帧原生 RAW 为 [1,16,16]，成片均为有限的 [3,16,16] 张量。推理只使用过去预览，不枚举最终候选或读取固定目标。

可复现该调用：

```python
from pathlib import Path
import torch
from capture_tm.pipeline_sources import iter_source_scenes
from capture_tm.joint_algorithm import JointCaptureAlgorithm

torch.set_num_threads(1)
demo = Path("runs/capture_tm/raw-pipeline-demo")
study = Path("runs/capture_tm/raw-pipeline-study")
scene = next(s for _, s in iter_source_scenes(demo / "sources/manifest.json")
             if s.split == "test")
for scheme, group in (("apple", "A11"), ("samsung", "S11")):
    algorithm = JointCaptureAlgorithm.from_checkpoint(
        study / scheme / "seed_0" / group / "selected.pt")
    result = algorithm.run_simulated(scene, seed=37, noisy=True)
    assert torch.isfinite(result["output"]).all()
    print(group, result["request"]["selected_index"], len(result["captures"]))
```

## 独立审查与修复

审查分为采集物理和数据/训练接口两路。具体问题以先失败的回归复现后修复：

- 非对称 PSF 原实现会被 `conv2d` 的互相关约定镜像；改为点响应卷积并增加脉冲检查。
- 原始同图换成 NPY/NPZ、改辐射倍率和 source_id 可能跨 split；增加变换前解码内容身份，保留来源/依赖分组。
- 增加实际 EV、计划/profile、预览曝光 state 和时钟的一致性检查；显式拒绝非有限最终时钟。
- checkpoint 的 sensor 列表/元组序列化保持数据包原始表示；RAW 推理恢复精确 acquisition 和 plans，不能静默退回旧 RGB 模式。
- 文档明确 rolling shutter 下跨行 PSF 与逐行时间积分不能普遍交换，并注明当前候选库固定 12/24 个。

仍然是双线性去马赛克和固定全局平移/运动拒绝融合基线；去马赛克噪声相关性只近似处理。源素材自身的噪声、模糊、饱和和插帧误差不会被修复。真实设备排序、风格合理性及联合画质收益需要新的标定和独立真实测试。
