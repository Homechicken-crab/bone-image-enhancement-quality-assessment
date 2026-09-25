from __future__ import annotations

from pathlib import Path

from .images import file_sha256, load_grayscale
from .metrics import EPSILON, roi_statistics
from .models import Project, ValidationIssue
from .project_store import ProjectStore


def validate_project(store: ProjectStore) -> list[ValidationIssue]:
    project = store.project
    issues: list[ValidationIssue] = []
    if project.original is None:
        return [ValidationIssue("error", "missing_original", project.id, "尚未导入原图")]

    original = project.original
    _validate_record(store, original, issues, None)
    original_array = None
    if not any(issue.severity == "error" and issue.object_id == original.id for issue in issues):
        try:
            original_array = load_grayscale(store.resolve(original.relative_path))
        except Exception:
            original_array = None
    for algorithm in project.algorithms:
        _validate_record(store, algorithm, issues, original)

    if not project.algorithms:
        issues.append(ValidationIssue("warning", "missing_algorithms", project.id, "尚未添加增强算法结果"))
    seen_names: set[str] = set()
    for algorithm in project.algorithms:
        if algorithm.display_name in seen_names:
            issues.append(ValidationIssue("warning", "duplicate_scheme_name", algorithm.id, f"方案名称重复：{algorithm.display_name}"))
        seen_names.add(algorithm.display_name)

    for roi in project.rois:
        if roi.width <= 0 or roi.height <= 0:
            issues.append(ValidationIssue("error", "invalid_roi_size", roi.id, f"ROI“{roi.name}”尺寸必须大于零"))
        if roi.x < 0 or roi.y < 0 or roi.x + roi.width > original.width or roi.y + roi.height > original.height:
            issues.append(ValidationIssue("error", "roi_out_of_bounds", roi.id, f"ROI“{roi.name}”超出原图边界"))
        if roi.type in {"weak_bone", "strong_bone"}:
            pair = next((item for item in project.rois if item.id == roi.paired_surrounding_roi_id), None)
            if pair is None or pair.type != "surrounding":
                issues.append(ValidationIssue("warning", "missing_surrounding_pair", roi.id, f"Bone ROI“{roi.name}”没有有效的 Surrounding 配对"))
        if roi.type == "background" and original_array is not None and roi.width > 0 and roi.height > 0:
            region = original_array[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
            stats = roi_statistics(region)
            if stats["std_intensity"] is None or float(stats["std_intensity"]) <= EPSILON or int(stats["unique_pixel_count"] or 0) < 2:
                issues.append(ValidationIssue(
                    "warning", "constant_background_roi", roi.id,
                    f"背景 ROI“{roi.name}”几乎为常量区域（std={stats['std_intensity'] or 0:.6g}，唯一灰度数={stats['unique_pixel_count']}），无法有效估计背景噪声。建议重新选择包含真实背景波动但不含人体结构的区域。",
                ))

    if not any(roi.type == "weak_bone" for roi in project.rois):
        issues.append(ValidationIssue("warning", "missing_weak_bone", project.id, "尚未创建 Weak Bone ROI，主 CNR 和主清晰度指标不可计算"))
    if not any(roi.type == "background" for roi in project.rois):
        issues.append(ValidationIssue("warning", "missing_background", project.id, "尚未创建 Background ROI，背景噪声和 Background-based CNR 不可计算"))
    if not any(roi.type == "strong_bone" for roi in project.rois):
        issues.append(ValidationIssue("warning", "missing_strong_bone", project.id, "尚未创建 Strong Bone ROI，Saturation Ratio 不可计算"))
    for index, first in enumerate(project.rois):
        for second in project.rois[index + 1 :]:
            if _overlap(first, second):
                sensitive = (
                    (first.type == "background" and second.type in {"weak_bone", "strong_bone"})
                    or (second.type == "background" and first.type in {"weak_bone", "strong_bone"})
                    or first.paired_surrounding_roi_id == second.id
                    or second.paired_surrounding_roi_id == first.id
                )
                if sensitive:
                    issues.append(ValidationIssue("warning", "sensitive_roi_overlap", first.id, f"ROI“{first.name}”与“{second.name}”重叠，请确认该区域选择合理"))
    return issues


def _overlap(first, second) -> bool:
    return not (
        first.x + first.width <= second.x
        or second.x + second.width <= first.x
        or first.y + first.height <= second.y
        or second.y + second.height <= first.y
    )


def _validate_record(store: ProjectStore, record, issues: list[ValidationIssue], original) -> None:
    path = store.resolve(record.relative_path)
    if not path.exists():
        issues.append(ValidationIssue("error", "missing_file", record.id, f"图像文件不存在：{path}"))
        return
    try:
        image = load_grayscale(path)
    except Exception as exc:
        issues.append(ValidationIssue("error", "invalid_image", record.id, str(exc)))
        return
    actual_hash = file_sha256(path)
    if actual_hash != record.sha256:
        issues.append(ValidationIssue("error", "file_changed", record.id, "项目内图像文件在导入后发生变化"))
    height, width = image.shape
    if width != record.width or height != record.height or str(image.dtype) != record.dtype:
        issues.append(ValidationIssue("error", "metadata_mismatch", record.id, "图像内容与项目元数据不一致"))
    if original is not None:
        if width != original.width or height != original.height:
            issues.append(ValidationIssue("error", "size_mismatch", record.id, "图像尺寸与原图不一致，程序不会自动缩放"))
        if str(image.dtype) != original.dtype:
            issues.append(ValidationIssue("error", "dtype_mismatch", record.id, "图像位深与原图不一致，程序不会自动转换"))
