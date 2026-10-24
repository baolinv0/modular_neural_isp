# TM 算法库实现与独立 review 记录

> 本页为 v1 历史记录；当前组合架构和预训练分割模型接入的实现、验证见 [v2 review](tm_library_v2_review.md)。

日期：2026-10-05。基础版本：`e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3`，工作分支：`feat/tm-algorithm-library`。

本次交付是可训练、可评测的研究实现。原 `photofinishing` 源码和 checkpoint 文件保持原样；库通过封装复用 Gain、GTM、chroma 和 gamma。五类候选替换局部 TM，原 baseline 单独保留。具体接口见 [设计](superpowers/specs/2026-10-05-tm-library-design.md)，使用方式见 [README](../tm_library/README.md)。

## 分工和机制验收

数据、公共模型框架、A–E 各算子、训练工具及文档分别由独立实现 agent 完成。另设算法 reviewer、数据/训练 reviewer、全库 reviewer；reviewer 不编辑实现，问题交回实现方修复，再进行复核。

| 候选 | 实际机制 | 定义性验证 |
|---|---|---|
| A `region_curves` | 正增量单调曲线库与逐像素 softmax 混合 | 单调性、端点、归一化权重、专家/区域响应与梯度 |
| B `spatial_grid` | 真实 3D bilateral grid、学习 guidance、三线性 slicing | 深度坐标敏感性、切片梯度、五个局部参数与门控端点 |
| C `gain_residual` | 零初始化低分辨率标量 EV、guided upsample、二次限幅 | 初始恒等、过冲限幅、RGB 比例保持及实际优化 |
| D `exposure_fusion` | 单图合成曝光、Laplacian 图像金字塔、Gaussian 权重金字塔 | 重建、单候选选择、多尺度结果区别于逐点加权 |
| E `base_detail` | log 亮度边缘感知分解、独立 base 与 detail 控制 | detail 控制确实改变结果，边缘、常量区域与重建行为 |

S0 忽略语义；S1 从共享特征训练辅助语义头，推理不需要标签；S2 接收外部软语义图与置信度。缺失标注不会作为负类监督，零置信度会屏蔽语义条件。算法结构不限定人像，实际适用范围由数据和实验决定。

## 发现并关闭的问题

| 问题 | 修复与复核 |
|---|---|
| 饱和 RGB 颜色触发上游 chroma 直方图的 NaN 梯度 | 库内 LUT 子类保留完全相同的 histogram/sqrt forward，只对下溢至零的直方图 bin 使用有限的零导数。原始代码、state keys、初始化和 RNG 顺序不变。 |
| 接受 128 EV 导致 float32 增益溢出 | `max_ev` 限制到 `[0,16]`，含拒绝非法配置和边界渲染回归。 |
| 没有语义图时非法 confidence 被忽略 | 显式模式始终校验置信度形状、有限值和范围。 |
| Python manifest builder 隐含声明已对齐 | API 和 CLI 均要求调用方显式断言 `target_aligned`。 |
| 仅语义/仅区域损失可能使无标注验证集恒为零 | 必须有正权重的全局图像损失，独立于语义有效性。 |
| 矩形图独立随机 90° 旋转导致 batch 无法堆叠 | 矩形只使用 0°/180°，方形保留四方向；预检读取未增强形状。 |
| CUDA `grid_sample` backward 与强制确定性不兼容 | CPU 使用严格确定性；非 CPU 使用 `warn_only=True` 并记录设置，不承诺 CUDA bit-exact。 |

三位独立 reviewer 的最终结论均以其检查范围为限：没有剩余阻断项。算法复核还验证了 float32/float64 histogram forward 精确相等、原始 baseline 输出零容差相等，以及初始化/RNG 兼容性。

## 执行证据

集成验收使用独立 synthetic train/val 划分，不下载真实训练数据。交付代码的最终完整测试（Python 3.12，CPU，OMP/MKL 各 1 线程）：

| 环境 | 命令 | 结果 |
|---|---|---|
| PyTorch `2.5.1+cpu` | `python -m pytest -q` | **160 passed**，97.45 秒 |
| PyTorch `2.14.1+cpu` | `python -m pytest -q` | **160 passed**，100.57 秒 |

两套环境并行执行，耗时不是模型性能基准。`git diff --cached --check` 通过。

- 八个 YAML 配置均从命令行完成两轮训练、验证选优、评测和无 GT 推理：baseline、A–E，以及 C 的 S1/S2 示例。
- 每个配置使用 4 张训练、2 张验证、32×32 fixture；推理 manifest 移除 target 字段，均输出 2 张 PNG、诊断 NPZ 和 metadata。
- 测试覆盖真实 optimizer 更新、冻结 backbone 的短程拟合、CPU 连续训练与 epoch 边界恢复的 state/history/RNG 一致性。
- 独立 reviewer 另测 S1 batch=2、有/无标注、标签与置信度变化不影响推理像素及诊断图、严格 checkpoint key 校验，以及 scene/camera/region 汇总。
- 全库 reviewer 完成 48 组极端输入实际反向传播：六条模型路径 × 3D LUT 开/关 × 黑/白/红/棋盘，batch=2、奇数尺寸。均未出现非有限输出或梯度。
- 算法 reviewer 复现原 NaN 问题后验证 baseline+A–E 的随机/预训练权重、纯蓝 3×5 输入均可反传。
- checkpoint 迁移边界另外复核：`last.pt` 保存非递归的完整最佳 checkpoint 快照，迁移输出目录且后续无改善时 `best.pt` 仍可恢复训练。独立三轮实验确认最佳 optimizer 状态不随后续训练变化、快照深度恒定，并从迁移后的 best 精确重放模型/history/scheduler/RNG。

可复验命令：

```bash
python -m pip install -r requirements-tm-library.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

完整 CLI 示例与 checkpoint/resume 命令见 README。CPU CI 工作流随代码提交；本记录只声明本地实际执行结果，不把尚未返回的远程 CI 当作通过。

## 证据边界

synthetic 指标仅检查软件链路，不用于给 A–E 排名或证明超过原模型。尚未开展真实相机数据训练、跨相机泛化、主观影调/肤色评价、视频时序、GPU 验证或移动 SoC profiling。

本库未包含商用语义分割器、ISP/NPU SDK、Direct Link 调度、量化部署或视频时序模块。当前 PyTorch pipeline 按 gain/base 依赖串行执行；它为后续算法消融提供统一接口，不能当作已实现硬件并行的证据。单图合成曝光不会创造新的采集信息。五个候选是公开范式启发的研究实现，不是厂商私有算法或专利的逐项复刻。
