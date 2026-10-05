# Apple 单帧 / Samsung HDR：物理数据合成与训练

本分支新增完整的数据入口：**显式线性素材导入 → Bayer 物理采集 → 可迁移数据包 → 独立验证 → 既有八组 AE/TM 训练 → checkpoint 推理**。

所有默认数值是工程假设。数据域为相对传感器辐射，不能据此声称已标定 Apple/Samsung 手机。两项专利用于指导采集与渲染分工，网络和合成数据方法属于本工程的设计。

## 快速运行

在仓库根目录、已安装 `capture_tm/requirements.txt` 的 Python 环境执行：

```bash
# 12个场景，三组source split，每组包含暗静态/暗运动/背光/普通四条件。
python -m capture_tm.pipeline_cli demo \
  --output runs/capture_tm/raw-demo --scenes 12 --size 16 \
  --scheme both --noise-seeds 0 --seed 2026 --threads 1

# 独立从导出的RAW重建RGB及固定融合，检查数据包一致性。
python -m capture_tm.pipeline_cli validate \
  --manifest runs/capture_tm/raw-demo/acquisition/manifest.json \
  --output runs/capture_tm/raw-demo/validation.json

# 两个方案各四组；直接读取同一数据包，不重采集或重融合训练输入。
python -m capture_tm.joint_cli \
  --manifest runs/capture_tm/raw-demo/acquisition/manifest.json \
  --output runs/capture_tm/raw-study --scheme both \
  --epochs 1 --warmup 1 --seeds 0 --noise-seeds 0 \
  --threads 1 --prepare-threads 1 --candidate-chunk-size 4
```

这是CPU协议检查配置，不用于评价手机画质。扩大实验时增加场景、噪声重复和训练种子，所有四组共用相同源划分、候选数据、目标和选模协议。

八组分别为 A00/S00（规则AE、冻结TM）、A10/S10（学习AE、冻结TM）、A01/S01（规则AE、学习TM）、A11/S11（学习AE、学习TM）。HDR融合F0在四组间固定。

## 接入自己的线性HDR素材

以 `configs/capture_tm/source_recipe.example.json` 为模板，提供明确的layout、时间戳、source_id和split。例子中的文件路径是待替换的本地素材，不是仓库自带数据。

```bash
python -m capture_tm.pipeline_cli import \
  --recipe my_sources.json --output runs/capture_tm/imported-scenes
python -m capture_tm.pipeline_cli generate \
  --manifest runs/capture_tm/imported-scenes/manifest.json \
  --acquisition-profile configs/capture_tm/acquisition_bayer.json \
  --output runs/capture_tm/imported-acquisition \
  --scheme both --noise-seeds 0,1 --seed 2026 --threads 1
```

训练的 `--noise-seeds` 必须等于数据包记录的基础种子列表，`--render-ev` 必须等于数据包的外观意图；更改两者需要重新生成数据。所有输出使用新目录，生成器不会覆盖已有数据包。

| 源格式 | 处理约定 |
|---|---|
| float NPY/NPZ | 核心依赖即可；NPZ帧键为`frames`，mask键为`mask` |
| float EXR | 可选`OpenEXR>=3`；单part、非deep、明确R/G/B通道；native解码实际验证过 |
| float TIFF | 可选`imageio[tifffile]`及相应codec；解码必须保持float；当前未运行真实TIFF成功fixture |
| RawGen / RL-3A / DNG代理 | 沿用既有`capture_tm import-*`命令产生v1 manifest，再用新`generate`；代理来源与限制保留 |

新recipe入口支持`input_domain=sensor_linear_relative_radiance`或`linear_xyz`；XYZ需要显式`xyz_to_sensor`。PNG/JPEG不在这个float-HDR入口内；旧命令仍支持其已定义的RGB16代理。编码图不能自动充当未裁切辐射真值。

`input_path`可以是单帧或完整时序数组；`frame_paths`按显式列表读取序列。layout支持CHW/HWC/TCHW/THWC，序列列表提供时间轴。时间戳必须显式提供、严格递增；单帧也提供`[0.0]`。整段素材共享一次`radiance_scale`，不会逐帧归一化。曝光窗超出动态素材时域时拒绝，不用复制边界帧补齐。

裁剪、曝光变体、reference和同一片段的其他来源应共享source_id或dependency_ids。导入前检查来源、依赖与路径跨split；另在变换和亮度缩放之前记录原始解码内容身份，防止同图换容器、改亮度后跨split。自动身份检查不能替代正确的原始episode分组。

## 采集模型和参考位置

```text
线性相对场景
→ 可选PSF（仅pre_optics）
→ 像元面积平均
→ 逐行线性曝光窗积分
→ CFA选择
→ Poisson光子采样 / full-well限制
→ 电子读噪 / analog gain
→ ADC噪声 / black level / 量化与裁切
→ 原生Bayer DN
→ 双线性去马赛克camera RGB
→ 固定WB/CCM；HDR时先固定F0融合
→ TM输入与单独的virtual gain条件
```

实现先一次对整段素材施加光学处理和像元面积平均，再按native sensor各行的实际曝光窗积分。全局快门的共同时间窗可与线性空间处理交换；rolling shutter各行时间权重不同，不能普遍交换跨行PSF与时间积分。任何full-well/ADC饱和都在积分之后。非对称PSF按点响应卷积约定，不使用镜像的互相关核。

- `post_optics`适合已经过镜头成像的实拍素材；同时指定PSF会报错。
- `pre_optics`用于理想渲染源，可指定三通道奇数尺寸PSF；未指定时声明理想delta PSF。
- `spatial_downsample`为整数像元面积平均，H/W必须整除且native Bayer尺寸为偶数。训练输入H/W至少12。
- CFA支持RGGB/BGGR/GRBG/GBRG。噪声加在实际CFA采样之后；保留负的black-subtracted读噪声，去马赛克后才接前端。
- raw variance用black-subtracted归一化传感器单位的平方表示。RGB variance使用插值权重的平方；去马赛克引入的协方差没有完整建模。

**固定目标位于：光学和像元面积处理之后、有限快门运动积分与传感器噪声之前、t=0。** 经过固定WB/CCM形成`sharp_reference`，独立Reinhard shoulder+sRGB OETF形成`target`。所有曝光候选和两种方案共用该目标。不是把已经裁掉高光的EV0照片直接当GT，也不让当前学习TM生成自己的GT。

单帧动作是(t,ga)，最终数据固定digital_gain=1。快门决定光子数和运动积分；analog gain不新增光子。ISO需要先映射设备实际增益与噪声参数，不能仅把ISO当亮度倍率。

Samsung三帧为顺序曝光。共同采集窗口为[-50,+50]ms，nominal中心为(-1/30,0,+1/30)s。为rolling shutter和readout预留空间时，整组候选的快门按共同因子缩小，保持bracket比值；实际动作全部存入数据包与checkpoint。rolling各行中心从`center-rs/2`到`center+rs/2`，末行曝光结束后再计readout。预览的全部readout必须早于最终曝光开始。

## 数据包字段

`manifest.json`为version2、kind=`capture_tm_acquisition_dataset`，包含sensor、acquisition、基础噪声种子、固定外观意图，以及按apple/samsung组织的plans和records。record路径相对manifest，整个目录可以移动。采集参数和标定限制随包保存。

每个scene的`.pt`用`torch.load(...,weights_only=True)`读取：

| 字段 | 形状/含义 |
|---|---|
| raw_dn | [R,K,F,1,H,W]，包括black的量化Bayer DN |
| raw_noise_variance / raw_saturation_mask | 同形状，分别为未裁切variance proxy与full-well/ADC mask |
| images / reliability / missing | [R,K,3,H,W] / [R,K,1,H,W] / [R,K,3,H,W]，固定前端/融合产物 |
| capture_ev / radiance_mse | [R,K]，真实采集尺度或HDR公共尺度、相对固定参考误差 |
| capture_metadata | [R][K][F]：实际动作、曝光窗、读出结束、噪声种子、来源与profile |
| previews / state | [3,3,H,W] / [3,3]，历史实际观测与log2(t/tref),log2(ga),log2(gd) |
| sharp_reference / target | [3,H,W]，独立且动作不变的线性与display目标 |
| subject_mask | 可选[1,H,W]，固定参考坐标的区域标签 |

R为噪声重复，K为候选计划，F为1或3。每个scene逐一处理，之前的场景张量不保留；候选数据包仍需容纳单scene的全部候选。当前CLI固定12/24个候选。大分辨率场景应先构造统一裁剪或减少噪声重复，不能随意改每个候选的归一化。

生成随机种子绑定scene/source/split/repeat/candidate/frame；改输出目录或manifest行顺序不改变同一scene的采集噪声。两种方案共享同一历史预览。数据发布采用临时目录，完成全包schema检查后再rename；失败不留下正式manifest。

## 验证和训练边界

训练loader验证profile、计划、dtype/shape、EV、固定目标、来源划分和预览/最终时钟，不在训练时重融合。独立`validate`命令另外从导出的Bayer DN重新去马赛克，依靠观测数据重算F0和前端，逐候选核对images/reliability/missing/radiance_mse；不读取源场景重新采集。

AE网络只读取previews、state和计划特征。目标、未来场景、RAW真值噪声方差与候选成片不进入推理AE输入。训练的离散质量反馈属于固定环境下的目标，不能称为普遍唯一的最佳AE标签。

默认噪声是独立CFA Poisson/read/ADC模型，未含设备row noise、dark current、crosstalk和dual-conversion-gain。噪声标定目前为全局标量，设备ISO曲线需要另行接入。F0为观测驱动的全局整数平移/运动拒绝研究基线；RGB观测噪声估计未完整传播去马赛克协方差。源视频本身的噪声、模糊、裁切或插帧误差不被修复。

八组CPU运行与物理回归测试证明接口、软件链路和限制条件，真实设备排序准确性、AE/TM联合收益及目标风格合理性仍需独立真实测试。

验证结果与审查修复记录见[DATA_PIPELINE_VALIDATION.md](DATA_PIPELINE_VALIDATION.md)。
