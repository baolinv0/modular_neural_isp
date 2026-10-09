# AE–TM 实验产物

这里保存原 **RGB 模拟域**的 40 场景实验：24 个验证集选定模型、源数据、训练/评测日志、图片及运行记录。它不是后来新增的 Bayer 候选数据包，也不代表真实手机画质改善。

41,759,233 字节的 ZIP 以 70 个分片存于 Git。原清单路径遗漏 `.zip` 的问题已校正，实际重组和 ZIP 完整性检查通过；原始分片内容和 SHA-256 未改变。

在仓库根目录运行：

```bash
python3 artifacts/reconstruct_archive.py
python3 -m zipfile -t artifacts/ae_tm_single_hdr_experiments_20261005.zip
test ! -e runs/capture_tm/recovered-rgb-study && \
  python3 -m zipfile -e artifacts/ae_tm_single_hdr_experiments_20261005.zip \
  runs/capture_tm/recovered-rgb-study
```

脚本检查清单中的大小和 SHA-256，会覆盖同名重组 ZIP，不改动原分片。完整环境、模型推理、重训和 Bayer 数据生成步骤见 [复现手册](../docs/capture_tm/REPRODUCE_DATA.md)，结果见 [实验报告](../docs/capture_tm/JOINT_RESULTS.md)。
