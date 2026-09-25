from __future__ import annotations

from pathlib import Path

from .images import file_sha256, load_grayscale
from .models import Project, ValidationIssue
from .project_store import ProjectStore


def validate_project(store: ProjectStore) -> list[ValidationIssue]:
    project = store.project
    issues: list[ValidationIssue] = []
    if project.original is None:
        return [ValidationIssue("error", "missing_original", project.id, "尚未导入原图")]

    original = project.original
    _validate_record(store, original, issues, None)
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
