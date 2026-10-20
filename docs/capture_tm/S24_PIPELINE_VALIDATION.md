# S24 计划前的合成管线验证记录

2026-10-05；算法基线 `09aa3afbe6be3f3c2a8f23670713a2ec4aaf582f`。本轮新增文档，不改变算法或历史目标。本记录中的12场景是解析模拟场景，**不是用户下载的 S24 图像**。

环境：Python3.12.14、PyTorch2.5.1+cpu，`cuda_available=false`；生成与重算单线程。

| 实际执行 | 结果 |
|---|---|
| acquisition/pipeline/source/learned-tone专项 | **111 passed in 25.70s**，无失败/跳过 |
| 新 Bayer demo | 12场景、32×32、train/val/test各4；两方案、noise0/1 |
| 计划库 | Apple12单帧动作；Samsung24三帧计划 |
| 采集 | RGGB、rolling0、readout1ms、post_optics、无额外PSF；工程假设传感器 |
| 导出RAW独立重算 | `valid=true`，`raw_composition_verified=true` |
| S24实际下载核验 | 未执行：当前工作区未找到用户下载的数据 |
| S24合成与新训练 | 未执行：专用导入/目标/代价及大数据接口尚待实现 |

## 本轮真实命令

以下命令在仓库根目录、现有Python环境运行。`s24_plan_validation_20261005`为本轮新输出目录名，**名称不代表其中有S24内容**。

```bash
python -m pytest -q tests/test_capture_tm_acquisition.py \
  tests/test_capture_tm_pipeline.py tests/test_capture_tm_pipeline_sources.py \
  tests/test_capture_tm_learned_tone.py

python -m capture_tm.pipeline_cli demo \
  --output /workspace/scratch/4e2b872b5c6b/s24_plan_validation_20261005 \
  --scheme both --scenes 12 --size 32 --noise-seeds 0,1 --threads 1

python -m capture_tm.pipeline_cli validate \
  --manifest /workspace/scratch/4e2b872b5c6b/s24_plan_validation_20261005/acquisition/manifest.json \
  --output /workspace/scratch/4e2b872b5c6b/s24_plan_validation_20261005/validation.json \
  --threads 1
```

关键JSON另保存为 [s24_plan_pipeline_validation.json](s24_plan_pipeline_validation.json)。validator自身报告schema为version1，被检验采集包为version2，两者不能混为一项。

## 证据解释

专项覆盖CFA/相位、积分/满阱/增益、读出、RAW导出、来源与split、目标一致和virtual-gain相关软件合同。新包重算支持“保存RAW、固定前端/融合和保存图像一致”。

它们不证明真实S24传感器物理准确、降噪图是无偏真值、style-0已对齐、源高光未裁切或训练能提升画质。旧J下Apple饱和退化仍是已知问题；本轮未重新训练，也没有改写该结论。

下一步的S24检查/训练验收详见 [S24_EXPERIMENT_PLAN.md](S24_EXPERIMENT_PLAN.md)。
