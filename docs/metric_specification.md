# 骨骼图像增强质量评价平台：第一阶段指标规范

**文件名：** `metric_specification.md`  
**规范版本：** 1.1.0  
**适用阶段：** 第一阶段（最小可用评价闭环）  
**状态：** 已确认，作为第一阶段实现与测试基线

## 1. 目的

本规范冻结第一阶段评价平台的指标定义、ROI 依赖、计算顺序、异常处理、聚合方式和结果输出规则。

第一阶段的目标不是增加尽可能多的指标，而是在完全一致的图像条件、ROI 和计算定义下，对原始图像及多个增强算法结果进行客观、可重复、可追溯的评价。

本规范覆盖以下核心指标：

1. Background-based CNR
2. Local CNR
3. Average Gradient
4. Background Noise
5. SSIM
6. Saturation Ratio

本规范同时约束相对变化量、ROI 聚合、灰度范围、无效结果和 CSV 输出。

## 2. 第一阶段实现边界

第一阶段优先完成以下闭环：

- 原图和增强算法结果管理
- 图像一致性检查
- 矩形 ROI 创建、修改、删除、命名、显示和隐藏
- Bone ROI 与 Surrounding ROI 显式配对
- 多个 Background ROI
- 本规范规定的六类指标
- 对原图和所有增强结果进行批量评价
- 总体指标表
- ROI 详细结果表
- CSV 导出
- 项目保存与重新打开

第一阶段只保存一份“当前有效评价结果”。可以保留历史记录、schema migration 和 repository 接口，但不要求实现复杂的评价历史管理、数据库或通用持久化框架。

第一阶段不实现综合总分，也不依据单个指标自动判断“最佳算法”。

## 3. 公平评价原则

所有参与评价的图像必须满足：

- 使用同一张原始图像作为参考。
- 图像宽度和高度完全一致。
- 像素坐标一一对应。
- 使用同一组 ROI 及同一组 Bone–Surrounding 配对。
- 使用相同的灰度数据类型和项目灰度范围。
- 使用相同的指标版本和参数。
- 不对不同算法分别进行归一化、直方图拉伸、缩放或配准。
- 指标计算前不自动平滑、锐化或降噪。

任何不满足像素级对应关系的图像不得参与 SSIM 和 ROI 对比。程序不得静默缩放图像。

## 4. 图像和数值约定

### 4.1 第一阶段支持范围

第一阶段正式支持：

- 单通道 `uint8` 灰度图，项目灰度范围 `[0, 255]`
- 单通道 `uint16` 灰度图，项目灰度范围 `[0, 65535]`

同一项目中的所有图像必须使用相同数据类型。彩色图像、浮点图像及带 alpha 通道图像在第一阶段视为不兼容输入，不进行静默转换。

### 4.2 项目灰度范围

记项目灰度下限为 \(L_{min}\)，灰度上限为 \(L_{max}\)，动态范围为：

\[
L=L_{max}-L_{min}
\]

指标计算使用项目声明的固定灰度范围，而不是每张图像自身的最小值和最大值。

图像中若出现超出项目灰度范围的像素，验证失败并阻止评价。

### 4.3 计算精度

- 读取后保留原始像素值。
- 指标计算前转换为 `float64`，但不缩放到 `[0,1]`。
- 内部结果使用 `float64` 保存。
- 界面默认显示 4 位小数。
- CSV 至少保留 8 位有效数字，避免界面舍入值进入后续统计。

### 4.4 标准差约定

除 SSIM 内部计算外，本规范中的 ROI 标准差均指样本标准差：

\[
s=\sqrt{\frac{1}{n-1}\sum_{k=1}^{n}(x_k-\bar{x})^2}
\]

即实现中使用 `ddof=1`。ROI 像素数小于 2 时，标准差及依赖它的指标不可计算。

## 5. ROI 数据约定

### 5.1 ROI 类型

第一阶段支持以下类型：

- `weak_bone`：弱骨骼区域
- `strong_bone`：强骨骼区域
- `surrounding`：骨骼周围区域
- `background`：无人体有效结构的背景区域

### 5.2 ROI 形状与坐标

第一阶段只支持矩形 ROI。

- 坐标原点位于原图左上角。
- `x` 向右增加，`y` 向下增加。
- 几何数据为整数像素坐标。
- ROI 使用半开区间 `[x, x + width)` 和 `[y, y + height)`。
- `width > 0` 且 `height > 0`。
- ROI 必须完全位于原图范围内。
- 保存原始像素坐标，不保存界面缩放坐标。

### 5.3 Bone–Surrounding 配对

每个参与 CNR 计算的 `weak_bone` 或 `strong_bone` ROI 必须通过稳定 ID 显式关联一个 `surrounding` ROI。

程序不得根据 ROI 名称、创建顺序或空间距离自动猜测配对关系。

一个 Surrounding ROI 可以被多个 Bone ROI 引用，但界面应明确显示这种共享关系。

### 5.4 ROI 重叠

第一阶段允许 ROI 在几何上重叠，但应产生非阻断警告，供用户检查。以下重叠尤其需要提示：

- Bone ROI 与其配对的 Surrounding ROI 重叠
- Background ROI 与任意 Bone ROI 重叠
- Background ROI 与人体结构区域明显重叠

程序不自动裁剪或修改重叠 ROI。

## 6. 公共 ROI 统计量

对每张图像和每个 ROI，首先计算并保存：

- `pixel_count`
- `mean_intensity`
- `std_intensity`
- `min_intensity`
- `max_intensity`

这些统计量既用于指标计算，也作为 ROI 表的基础数据输出。

设某个 Bone ROI 的均值和样本标准差为 \(\mu_b, s_b\)，其配对 Surrounding ROI 的均值和样本标准差为 \(\mu_s, s_s\)。

## 7. Background Noise

### 7.1 单个 Background ROI

对第 \(i\) 个 Background ROI：

\[
Noise_i=s_i
\]

其中 \(s_i\) 为该 ROI 内像素的样本标准差。

### 7.2 多个 Background ROI 的合并噪声

为避免不同背景位置的平均亮度差被误当作随机噪声，多个 Background ROI 不直接拼接后计算总体标准差，而采用 ROI 内方差的合并估计：

\[
\sigma_{bg}=\sqrt{
\frac{\sum_{i=1}^{m}(n_i-1)s_i^2}
{\sum_{i=1}^{m}(n_i-1)}
}
\]

其中：

- \(m\) 为有效 Background ROI 数量
- \(n_i\) 为第 \(i\) 个 Background ROI 的像素数
- \(s_i\) 为第 \(i\) 个 Background ROI 的样本标准差

`background_noise_pooled` 作为总体表中的 Background Noise 主值。

### 7.3 有效条件

- 至少存在一个有效 Background ROI。
- 合并分母必须大于 0。
- 单个 ROI 像素数小于 2 时，该 ROI 无效。

Background Noise 越低通常表示噪声控制越好，但必须结合骨骼可辨识度和清晰度指标解释。

## 8. Background-based CNR

### 8.1 定义

Background-based CNR 使用背景噪声作为分母：

\[
CNR_{bg}=\frac{|\mu_b-\mu_s|}{\sigma_{bg}}
\]

该指标回答：在项目背景噪声水平下，Bone ROI 与其局部周围区域的平均灰度差是否足够明显。

### 8.2 有效条件

- Bone ROI 有有效的 Surrounding ROI 配对。
- Bone ROI 和 Surrounding ROI 均至少包含 1 个像素。
- 至少存在一个有效 Background ROI。
- \(\sigma_{bg}>\varepsilon\)。

第一阶段规定：

\[
\varepsilon=10^{-12}
\]

若 \(\sigma_{bg}\le\varepsilon\)，结果标记为 `undefined_zero_background_noise`，不得返回无穷大或用 0 替代。

### 8.3 输出

对每个有效 Bone–Surrounding 配对保存：

- `cnr_background`
- `bone_mean`
- `surrounding_mean`
- `background_noise_pooled`
- `contrast_absolute = |mu_b - mu_s|`

## 9. Local CNR

### 9.1 定义

Local CNR 只使用当前 Bone ROI 和对应 Surrounding ROI 的局部方差，不依赖 Background ROI：

\[
\sigma_{local}=\sqrt{\frac{s_b^2+s_s^2}{2}}
\]

\[
CNR_{local}=\frac{|\mu_b-\mu_s|}{\sigma_{local}}
\]

该定义对 Bone 和 Surrounding 两个区域的方差给予相同权重，避免较大的 ROI 仅因像素数量更多而完全主导局部噪声估计。

Local CNR 回答：在该骨骼区域及其直接周围的局部灰度波动下，两者是否容易区分。

### 9.2 有效条件

- Bone ROI 有有效的 Surrounding ROI 配对。
- Bone ROI 和 Surrounding ROI 均至少包含 2 个像素。
- \(\sigma_{local}>\varepsilon\)。

若 \(\sigma_{local}\le\varepsilon\)，结果标记为 `undefined_zero_local_variance`。

### 9.3 输出

对每个有效 Bone–Surrounding 配对保存：

- `cnr_local`
- `bone_mean`
- `bone_std`
- `surrounding_mean`
- `surrounding_std`
- `local_noise_combined`
- `contrast_absolute`

### 9.4 两类 CNR 的主次关系

两类 CNR 必须始终同时计算、保存和导出，任何一种都不能覆盖另一种。

评价配置包含：

```json
{
  "primary_cnr": "background"
}
```

允许值为：

- `background`
- `local`

第一阶段默认值为 `background`，用于选择 Background-based CNR 或 Local CNR 作为主 CNR 类型。总体主 CNR 的 ROI 聚合范围固定为 Weak Bone ROI，即：

- `primary_cnr = weak_bone_mean_cnr_background`，或
- `primary_cnr = weak_bone_mean_cnr_local`

`all_bone_mean_cnr_*` 和 `strong_bone_mean_cnr_*` 作为辅助结果保留。主 CNR 类型设置不改变底层计算，也不删除另一类 CNR。后续可依据真实实验结果修改默认类型。

## 10. Average Gradient

### 10.1 定义

对二维灰度图像或矩形 ROI，使用向右和向下的相邻像素差：

\[
G(i,j)=\sqrt{
\frac{
[I(i,j+1)-I(i,j)]^2+[I(i+1,j)-I(i,j)]^2
}{2}}
\]

Average Gradient 定义为所有同时具有右邻像素和下邻像素的位置的平均值：

\[
AG=\frac{1}{(H-1)(W-1)}
\sum_{i=0}^{H-2}\sum_{j=0}^{W-2}G(i,j)
\]

其中 \(H\) 和 \(W\) 是当前全图或矩形 ROI 的高度和宽度。

### 10.2 计算范围

第一阶段计算并保存：

- `average_gradient_global`
- 每个 `weak_bone` ROI 的 `average_gradient_roi`
- 每个 `strong_bone` ROI 的 `average_gradient_roi`

可同时计算 Surrounding 和 Background ROI 的 AG 作为详细数据，但不作为第一阶段总体表核心列。

### 10.3 边界规则

- ROI 内 AG 只使用 ROI 内部相邻像素。
- 不读取 ROI 外的像素参与局部 AG。
- ROI 的宽度或高度小于 2 时，AG 不可计算。

Average Gradient 越高通常表示灰度变化和边缘响应更强，但也可能来源于噪声放大，因此必须结合 Background Noise 分析。

总体清晰度主指标固定使用 `weak_bone_mean_average_gradient`。`average_gradient_global` 和 `strong_bone_mean_average_gradient` 作为辅助结果保留。这样可避免大面积背景或原本已经较清晰的强骨骼区域主导总体清晰度判断。

## 11. SSIM

### 11.1 定位

SSIM 是增强结果相对于原图的结构保持约束指标，不是“增强效果越高越好”的综合指标。

原图与自身比较时 SSIM 为 1，这不表示原图是最佳增强结果。

### 11.2 第一阶段计算范围

- 对每个增强结果与原图计算全图 SSIM。
- 原图行固定记录 `ssim = 1.0`。
- 第一阶段不计算 ROI SSIM。

由于全身骨显像图像中背景通常占比较大，全图 SSIM 可能受到大量稳定背景像素影响，从而弱化骨骼局部结构变化在总体数值中的体现。因此，全图 SSIM 在本项目中仅作为结构保持约束，不能独立代表骨骼区域的结构保持质量。后续阶段可增加 Bone ROI SSIM，第一阶段暂不实现。

### 11.3 固定参数

为避免库默认参数变化导致结果漂移，第一阶段固定采用：

- `data_range = Lmax - Lmin`
- `gaussian_weights = true`
- `sigma = 1.5`
- `use_sample_covariance = false`
- `K1 = 0.01`
- `K2 = 0.03`
- `win_size = 11`
- `channel_axis = null`

边界处理由所选 SSIM 实现完成，但实现库和版本必须写入评价结果的运行元数据。

若图像任一维小于 11，第一阶段 SSIM 标记为不可计算，不自动改变窗口大小。

### 11.4 输出

- `ssim`
- `ssim_distance_from_original = 1 - ssim`

SSIM 不计算“相对原图变化百分比”。界面和 CSV 显示绝对值，以及可选的 `1 - SSIM`。

## 12. Saturation Ratio

### 12.1 目的

Saturation Ratio 用于检测脊柱、骨盆等 Strong Bone ROI 在增强后是否出现过多接近灰度上限的像素，从而提示可能的高亮饱和和细节压缩。

该指标只针对 `strong_bone` ROI 计算。

### 12.2 阈值

评价配置保存饱和阈值比例：

```json
{
  "saturation_threshold_ratio": 0.98
}
```

默认阈值为项目灰度范围的 98%：

\[
T_{sat}=L_{min}+0.98(L_{max}-L_{min})
\]

满足下式的像素视为接近饱和：

\[
I(x,y)\ge T_{sat}
\]

由于原始整数像素与浮点阈值直接比较：

- `uint8`、范围 `[0,255]` 时，等价于像素值 `>= 250`
- `uint16`、范围 `[0,65535]` 时，等价于像素值 `>= 64225`

评价结果必须保存实际使用的 `T_sat` 和阈值比例。

### 12.3 定义

对某个 Strong Bone ROI：

\[
SaturationRatio=
\frac{N(I\ge T_{sat})}{N_{ROI}}
\]

结果原始范围为 `[0,1]`。界面可用百分比显示。

### 12.4 输出与解释

对每个 Strong Bone ROI 保存：

- `saturation_ratio`
- `saturated_pixel_count`
- `pixel_count`
- `saturation_threshold_ratio`
- `saturation_threshold_value`

Saturation Ratio 越高表示接近灰度上限的像素比例越高，通常需要警惕过增强。但高亮骨骼在原图中可能已经存在一定比例的高灰度像素，因此不能只看增强图的绝对值，必须同时比较原图基线。

第一阶段不设置统一的“过增强合格线”，只展示绝对值和相对原图的变化。

## 13. ROI 指标聚合

### 13.1 保留单 ROI 数据

每个 ROI 和每个 Bone–Surrounding 配对的原始结果必须完整保存。聚合结果不能替代单 ROI 数据。

### 13.2 聚合原则

同一类别多个 ROI 的指标采用“有效 ROI 指标的算术平均”，不按 ROI 像素数量加权：

\[
MetricMean=\frac{1}{K}\sum_{k=1}^{K}Metric_k
\]

这样可以避免面积较大的脊柱或骨盆 ROI 掩盖较小的手腕、脚踝 ROI。

保存以下聚合项：

- `weak_bone_mean_cnr_background`
- `weak_bone_mean_cnr_local`
- `strong_bone_mean_cnr_background`
- `strong_bone_mean_cnr_local`
- `all_bone_mean_cnr_background`
- `all_bone_mean_cnr_local`
- `weak_bone_mean_average_gradient`
- `strong_bone_mean_average_gradient`
- `strong_bone_mean_saturation_ratio`
- `background_noise_pooled`

聚合时只包含状态为 `valid` 的 ROI 结果，并同时保存：

- 有效 ROI 数量
- 无效 ROI 数量
- 被排除的 ROI ID 及原因

若有效 ROI 数量为 0，聚合结果为空，不返回 0。

## 14. 相对原图变化

### 14.1 一般指标

对于 CNR、Average Gradient 和 Background Noise，算法结果 \(M_a\) 相对原图 \(M_o\) 的变化百分比为：

\[
\Delta\%=\frac{M_a-M_o}{|M_o|}\times100\%
\]

若 \(|M_o|\le\varepsilon\)，相对变化不可计算，状态标记为 `undefined_zero_baseline`。不得用极小数替换分母来制造巨大百分比。

变化百分比只表示数值方向：

- CNR 增加通常表示可辨识度提高。
- Average Gradient 增加可能表示结构更清晰，也可能表示噪声增强。
- Background Noise 增加表示背景波动增强，通常是不利变化。

### 14.2 Saturation Ratio

Saturation Ratio 主要输出相对原图的百分点变化：

\[
\Delta SR_{pp}=(SR_a-SR_o)\times100
\]

例如从 2% 增加到 5%，记录为 `+3.0 percentage points`，而不是只显示 `+150%`。

若原图 \(SR_o>\varepsilon\)，可额外保存普通相对变化百分比，但界面默认显示百分点变化。若原图为 0，则普通相对百分比为空。

### 14.3 SSIM

SSIM 不计算相对原图变化百分比，只保存 `ssim` 和 `1 - ssim`。

### 14.4 原图行

总体表中原图的变化列显示 `—`，不得显示 `0%` 来暗示该值参与算法改善判断。

## 15. 指标方向与界面解释

| 指标 | 界面方向 | 解释限制 |
|---|---|---|
| Background-based CNR | 通常越大越好 | 必须结合噪声、饱和和结构保持 |
| Local CNR | 通常越大越好 | 局部方差降低也可能抬高数值，应结合局部图像与 AG |
| Average Gradient | 通常越大表示变化更强 | 可能由有效边缘或噪声共同导致 |
| Background Noise | 通常越小越好 | 过度平滑也可能降低噪声并损失细节 |
| SSIM | 结构保持约束 | 不能独立代表增强质量 |
| Saturation Ratio | 通常越低越安全 | 必须相对原图解释，高亮区域可能天然接近上限 |

界面和自动导出不得把上述任一指标单独表述为综合优劣结论。

## 16. 评价前验证

### 16.1 阻断错误

以下问题阻止对应图像进入正式评价：

- 文件不存在或无法解码
- 非单通道灰度图
- 尺寸与原图不一致
- 数据类型与原图不一致
- 像素超出项目灰度范围
- ROI 越界
- 原图不存在

以下问题阻止对应指标计算，但不一定阻止其他指标：

- 没有有效 Background ROI：Background Noise 和 Background-based CNR 不可计算
- Bone ROI 没有 Surrounding 配对：该 Bone ROI 的两类 CNR 不可计算
- ROI 尺寸不足：依赖标准差或梯度的指标不可计算
- 没有 Strong Bone ROI：Saturation Ratio 无结果，但不影响其他指标

### 16.2 警告

以下情况产生警告但可以继续：

- ROI 存在重叠
- 方案名称重复
- 某些 Bone ROI 未配对但其他 ROI 可正常评价
- Saturation Ratio 明显增加
- 图像文件内容自上次评价后发生变化

方案名称重复时应要求用户区分，CSV 同时保存稳定方案 ID，避免结果混淆。

## 17. 批量评价顺序

每次评价按以下顺序执行：

1. 加载原图、算法图像、评价配置和 ROI。
2. 进行项目级和图像级一致性检查。
3. 对每张有效图像提取公共 ROI 统计量。
4. 计算各 Background ROI 噪声及合并 Background Noise。
5. 计算每个 Bone–Surrounding 配对的 Background-based CNR。
6. 计算每个 Bone–Surrounding 配对的 Local CNR。
7. 计算全图及 Bone ROI Average Gradient。
8. 计算 Strong Bone ROI Saturation Ratio。
9. 计算算法图像相对于原图的全图 SSIM。
10. 计算 ROI 分类聚合值。
11. 计算允许计算的相对原图变化。
12. 保存当前评价结果、配置快照和输入摘要。

一个算法图像失败时，其他合格算法继续评价。失败方案保留在结果表中，并显示失败状态和原因，不使用 0 填充。

## 18. 总体表字段

第一阶段总体表至少包含：

| 字段 | 含义 |
|---|---|
| `scheme_name` | 原图或算法方案名称 |
| `primary_cnr` | 当前配置选择的 Weak Bone Mean CNR |
| `primary_cnr_change_percent` | 主 CNR 相对原图变化 |
| `weak_bone_mean_cnr_background` | Weak Bone ROI 的 Background-based CNR 均值 |
| `weak_bone_mean_cnr_local` | Weak Bone ROI 的 Local CNR 均值 |
| `all_bone_mean_cnr_background` | 全部有效 Bone ROI 的 Background-based CNR 辅助均值 |
| `all_bone_mean_cnr_local` | 全部有效 Bone ROI 的 Local CNR 辅助均值 |
| `strong_bone_mean_cnr_background` | Strong Bone ROI 的 Background-based CNR 辅助均值 |
| `strong_bone_mean_cnr_local` | Strong Bone ROI 的 Local CNR 辅助均值 |
| `weak_bone_mean_average_gradient` | Weak Bone ROI 的 AG 主指标 |
| `weak_bone_mean_average_gradient_change_percent` | Weak Bone Mean AG 相对原图变化 |
| `average_gradient_global` | 全图 Average Gradient 辅助指标 |
| `strong_bone_mean_average_gradient` | Strong Bone ROI 的 AG 辅助指标 |
| `background_noise_pooled` | 合并背景噪声 |
| `background_noise_change_percent` | 相对原图变化 |
| `ssim` | 相对原图的全图 SSIM |
| `strong_bone_saturation_mean` | Strong Bone ROI 饱和比例均值 |
| `saturation_change_pp` | 相对原图百分点变化 |
| `status` | `valid`、`partial` 或 `failed` |
| `message` | 缺失指标、警告或失败原因 |

即使界面将其中一个 CNR 作为主列，也应允许用户查看另一个 CNR。

## 19. ROI 表字段

第一阶段 ROI 表至少保存：

- `scheme_id`
- `scheme_name`
- `roi_id`
- `roi_name`
- `roi_type`
- `paired_roi_id`
- `paired_roi_name`
- `pixel_count`
- `mean_intensity`
- `std_intensity`
- `min_intensity`
- `max_intensity`
- `average_gradient_roi`
- `cnr_background`
- `cnr_local`
- `saturation_ratio`
- `saturated_pixel_count`
- `status`
- `message`

不适用于某 ROI 类型的字段留空。例如 Background ROI 的 CNR 字段为空，而不是 0。

## 20. CSV 导出

第一阶段至少导出：

### 20.1 `metrics.csv`

每个方案一行，保存总体指标、变化量、状态和运行元数据引用。

### 20.2 `roi_metrics.csv`

每个“方案 × ROI”一行，保存 ROI 统计量和适用指标。

对于 Bone–Surrounding 配对的 CNR，结果记录在 Bone ROI 行，并保存对应 Surrounding ROI 的 ID 和名称。

### 20.3 `validation.csv`

保存每张图像和每个评价条件的验证结果：

- 检查项
- 严重级别
- 是否通过
- 对象 ID
- 说明

### 20.4 CSV 通用规则

- 编码使用 UTF-8 with BOM。
- 小数点固定使用 `.`。
- 缺失数值留空。
- 无效原因写入状态和消息列。
- 导出文件保存指标规范版本、项目 ID 和评价 ID。

## 21. 结果可重复性元数据

每次评价结果至少保存：

- `metric_spec_version`
- `evaluation_timestamp`
- `project_id`
- `evaluation_id`
- 原图文件哈希
- 每个算法图像文件哈希
- ROI 配置哈希或完整 ROI 快照
- 项目灰度范围
- `primary_cnr`
- `saturation_threshold_ratio`
- SSIM 固定参数
- 程序版本
- Python 版本
- NumPy、OpenCV、scikit-image 版本

第一阶段不要求复杂 migration 系统，但项目文件和结果文件必须包含 `schema_version`。遇到未知的更高版本时，应拒绝写入并提示版本不兼容，避免损坏项目。

## 22. 结果失效规则

发生以下任一变化时，已有评价结果标记为过期：

- 原图文件内容变化
- 任一算法图像内容变化
- 添加、删除或替换算法图像
- ROI 创建、删除、移动、缩放或类型变化
- Bone–Surrounding 配对变化
- 项目灰度范围变化
- 主指标之外的任何计算参数变化
- 指标规范版本变化
- 饱和阈值变化

仅修改方案显示名称时不必重新计算指标，但导出结果必须更新名称。

更改 `primary_cnr` 只影响展示时，不要求重新计算，因为两类 CNR 始终已经保存。

## 23. 第一阶段测试基线

实现前必须为指标准备可手工验证或解析验证的合成图像测试。

### 23.1 Background Noise

- 常量背景：噪声为 0。
- 已知像素序列：与 `ddof=1` 样本标准差一致。
- 多背景 ROI：合并结果必须符合本规范的合并方差公式。

### 23.2 两类 CNR

- 已知均值与方差的 Bone、Surrounding、Background ROI。
- 零背景方差时 Background-based CNR 应无效。
- Bone 和 Surrounding 均为常量时 Local CNR 应无效。
- 改变 Background ROI 只应影响 Background-based CNR，不应影响 Local CNR。
- 改变 Bone 或 Surrounding 局部方差应影响 Local CNR。

### 23.3 Average Gradient

- 常量图：AG 为 0。
- 线性水平渐变：结果与公式解析值一致。
- ROI 计算不得使用 ROI 外像素。
- 宽或高小于 2 的 ROI 应无效。

### 23.4 SSIM

- 图像与自身比较为 1。
- 局部修改后小于 1。
- 不同尺寸图像被验证层拒绝。
- 8-bit 和 16-bit 测试使用各自固定 `data_range`。

### 23.5 Saturation Ratio

- 无阈值以上像素时为 0。
- 全部达到阈值时为 1。
- 恰好位于阈值的像素计入饱和像素。
- 8-bit 默认阈值应使 `249` 不计入、`250` 计入。
- 原图比例为 0 时，只输出百分点变化，普通相对百分比为空。

### 23.6 项目级测试

- 保存项目后重新打开，ROI 坐标和配对关系不变。
- 新增算法无需修改指标代码。
- 替换图像后旧结果失效。
- 重命名算法不改变内部 ID 和已有数值结果。
- 所有方案读取同一份 ROI 配置。
- CSV 数值与内存中的评价结果一致。

## 24. 第一阶段非目标

本规范暂不定义：

- Entropy
- PSNR
- NIQE
- BRISQUE
- Edge Count
- Difference Map
- 雷达图
- 综合评分
- 自动排名
- 自动文字评价
- ROI SSIM
- 多边形 ROI
- 图像自动配准
- 不同尺寸图像的重采样评价

上述能力后续增加时，必须使用新的指标规范版本，并保持第一阶段结果可追溯。

## 25. 第一阶段实现决策摘要

1. 同时保存 Background-based CNR 和 Local CNR。
2. Background-based CNR 使用多个 Background ROI 的合并内部方差。
3. Local CNR 使用 Bone 和对应 Surrounding ROI 方差的均方合成。
4. 默认主 CNR 类型暂设为 Background-based CNR，总体主值使用 Weak Bone Mean；All Bone Mean 和 Strong Bone Mean 作为辅助结果。
5. Strong Bone ROI 使用默认 98% 灰度上限阈值计算 Saturation Ratio。
6. 饱和变化默认以百分点而非相对百分比展示。
7. 总体清晰度主指标使用 Weak Bone Mean Average Gradient；Global AG 和 Strong Bone Mean AG 作为辅助结果。
8. Average Gradient、噪声和 CNR 均保留单 ROI 数据及不加权 ROI 均值。
9. SSIM 只作为结构保持约束，并固定全部关键参数；全图 SSIM 可能受大面积稳定背景影响，第一阶段不实现 Bone ROI SSIM。
10. 无效指标使用空值、状态和原因表示，绝不使用 0 冒充。
11. 第一阶段优先实现完整评价闭环，不让复杂 migration、历史系统或 repository 抽象阻塞开发。
