# Changelog

## v0.1.1

- 重写 Windows `run.bat`，使用 ASCII 与 CRLF，并在失败时保留窗口和退出码。
- 新项目默认使用 Weak Bone Mean Local CNR；已有项目配置保持不变。
- 创建 Bone ROI 时自动选择唯一或最近的 Surrounding ROI，并在列表中显示配对关系。
- Background ROI 增加唯一灰度数和常量区域警告；零背景方差不再产生伪造 CNR。
- 新增只分析原图、需人工接受的可解释 ROI 自动推荐助手。
- 新增 Background、Strong Bone、Weak Bone 和相邻 Surrounding 候选及自动配对。
- 新增 `valid_with_warnings`，将核心评价缺失与辅助检查警告分开。
- 结果表增加明确原因和主 CNR 切换建议。
- 增加常量背景、自动配对、推荐边界、持久化和公平性回归测试。
