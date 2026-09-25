# Bone Image Enhancement Quality Assessment Tool

骨骼图像增强质量评价平台是一个面向“光电图像处理课程设计”的本地科研实验工具，用统一 ROI、统一灰度范围和统一指标定义比较原图及多个增强算法结果。

当前版本遵循 [指标规范 1.1.0](docs/metric_specification.md)，重点保证正确性、可重复性、实验公平性和后续扩展能力。

## 已实现

- 新建、保存和重新打开自包含项目
- 导入或替换唯一原图
- 添加、重命名、替换和删除任意数量的增强方案
- 尺寸、位深、通道、文件哈希和 ROI 边界检查
- 在原图上绘制并编辑矩形 ROI
- Weak Bone、Strong Bone、Surrounding、Background 四类 ROI
- Bone–Surrounding 显式配对
- Background-based CNR 和 Local CNR
- Weak Bone Mean CNR 主指标
- Weak Bone Mean Average Gradient 主清晰度指标
- Background Noise
- 全图 SSIM 结构保持约束
- Strong Bone Saturation Ratio
- 批量评价、总体表、辅助指标表、ROI 表
- `metrics.csv`、`roi_metrics.csv`、`validation.csv` 和完整 JSON 导出

## 安装与运行

要求 Python 3.11 或更高版本。Tkinter 通常随 Windows Python 一起安装。

```powershell
git clone https://github.com/Homechicken-crab/bone-image-enhancement-quality-assessment.git
cd bone-image-enhancement-quality-assessment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m bone_iqa
```

安装完成后，也可以运行 `run.ps1`，或者直接双击 `run.bat`。

使用源码直接启动：

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m bone_iqa
```

## 推荐操作顺序

1. 新建一个空项目目录。
2. 导入原始灰度图。
3. 添加若干算法结果并查看一致性检查。
4. 先创建 Surrounding ROI 和 Background ROI。
5. 创建 Weak/Strong Bone ROI，并为每个 Bone ROI 选择对应的 Surrounding ROI。
6. 点击“开始评价”。
7. 检查总体表、辅助指标表、ROI 表和验证信息。
8. 导出 CSV。

## 公平性约束

程序不会自动缩放、配准、归一化，也不会把彩色图静默转换成灰度图。第一阶段只接受同尺寸、同位深的原生单通道 `uint8` 或 `uint16` 图像。

## 指标解释原则

- Background-based CNR 和 Local CNR 同时保存；默认主指标是 Weak Bone Mean Background-based CNR。
- 清晰度主指标是 Weak Bone Mean Average Gradient，全图 AG 仅作为辅助结果。
- Background Noise 使用 Background ROI 的合并内部方差估计。
- Saturation Ratio 用于提示 Strong Bone ROI 的高亮饱和风险。
- 全图 SSIM 可能受到大面积稳定背景影响，只作为结构保持约束，不作为综合质量分数。

平台不计算简单加权总分，也不自动宣称某种算法“最佳”。完整定义见 [指标规范](docs/metric_specification.md)。

## 当前限制

- 第一阶段 ROI 仅支持矩形。
- Bone ROI SSIM 尚未实现。
- 暂不提供综合评分、自动排名、图表和 Excel 导出。
- 项目暂时只保存最新一次评价结果。

## 测试

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

## 项目状态

当前为第一阶段 `v0.1.0`。后续计划包括局部视觉对比、同步缩放、图表、Excel、规则化文字摘要和更多辅助指标。
