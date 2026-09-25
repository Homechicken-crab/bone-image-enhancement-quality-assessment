from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

import numpy as np
import PIL

from . import METRIC_SPEC_VERSION, __version__
from .images import load_grayscale
from .metrics import (
    EPSILON,
    average_gradient,
    cnr_background,
    cnr_local,
    pooled_background_noise,
    relative_change,
    roi_statistics,
    saturation_ratio,
    structural_similarity,
)
from .models import ImageRecord, ROI
from .project_store import ProjectStore
from .validation import validate_project


def _mean_valid(values: Iterable[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def _crop(image: np.ndarray, roi: ROI) -> np.ndarray:
    return image[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]


def _metric_status(row: dict[str, Any], validation_warnings: list[str]) -> tuple[str, str]:
    missing_core: list[str] = []
    warnings = list(validation_warnings)
    if row.get("weak_bone_mean_cnr_background") is None and row.get("weak_bone_mean_cnr_local") is None:
        missing_core.append("Weak Bone CNR（Local 与 Background 均不可计算）")
    if row.get("weak_bone_mean_average_gradient") is None:
        missing_core.append("Weak Bone Average Gradient")
    if row.get("ssim") is None:
        missing_core.append("SSIM")
    if missing_core:
        return "partial", "缺少核心指标：" + "；".join(missing_core)
    if row.get("background_noise_pooled") is None:
        warnings.append("未获得有效 Background Noise")
    elif float(row["background_noise_pooled"]) <= EPSILON:
        warnings.append("背景 ROI 方差为 0，Background CNR 不可计算")
    if row.get("strong_bone_saturation_mean") is None:
        warnings.append("未创建有效 Strong Bone ROI，Saturation Ratio 未计算")
    if row.get("weak_bone_mean_cnr_background") is None:
        warnings.append("Background CNR 不可计算")
    if row.get("weak_bone_mean_cnr_local") is None:
        warnings.append("Local CNR 不可计算")
    warnings = list(dict.fromkeys(item for item in warnings if item))
    return ("valid_with_warnings", "；".join(warnings)) if warnings else ("valid", "")


def _evaluate_image(
    image: np.ndarray,
    record: ImageRecord,
    rois: list[ROI],
    gray_min: int,
    gray_max: int,
    threshold_ratio: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    roi_by_id = {roi.id: roi for roi in rois}
    stats_by_id: dict[str, dict[str, Any]] = {}
    roi_rows: list[dict[str, Any]] = []

    for roi in rois:
        region = _crop(image, roi)
        stats = roi_statistics(region)
        stats_by_id[roi.id] = stats
        row = {
            "scheme_id": record.id,
            "scheme_name": record.display_name,
            "roi_id": roi.id,
            "roi_name": roi.name,
            "roi_type": roi.type,
            "paired_roi_id": roi.paired_surrounding_roi_id or "",
            "paired_roi_name": "",
            **stats,
            "average_gradient_roi": average_gradient(region),
            "cnr_background": None,
            "cnr_background_reason": "",
            "cnr_local": None,
            "cnr_local_reason": "",
            "saturation_ratio": None,
            "saturation_reason": "不适用于该 ROI 类型",
            "saturated_pixel_count": None,
            "saturation_threshold_value": None,
            "status": "valid",
            "message": "",
        }
        if roi.type == "strong_bone":
            ratio, count, threshold = saturation_ratio(region, gray_min, gray_max, threshold_ratio)
            row["saturation_ratio"] = ratio
            row["saturated_pixel_count"] = count
            row["saturation_threshold_value"] = threshold
            row["saturation_reason"] = "" if ratio is not None else "Strong Bone ROI 为空"
        roi_rows.append(row)

    background_stats = [stats_by_id[roi.id] for roi in rois if roi.type == "background"]
    background_noise = pooled_background_noise(background_stats)

    rows_by_roi = {row["roi_id"]: row for row in roi_rows}
    for roi in rois:
        if roi.type not in {"weak_bone", "strong_bone"}:
            continue
        row = rows_by_roi[roi.id]
        pair = roi_by_id.get(roi.paired_surrounding_roi_id or "")
        if pair is None or pair.type != "surrounding":
            row["status"] = "partial"
            row["message"] = "缺少有效 Surrounding 配对"
            row["cnr_background_reason"] = "Bone ROI 未配置 Surrounding"
            row["cnr_local_reason"] = "Bone ROI 未配置 Surrounding"
            continue
        bone_stats = stats_by_id[roi.id]
        surrounding_stats = stats_by_id[pair.id]
        row["paired_roi_name"] = pair.name
        row["cnr_background"] = cnr_background(
            float(bone_stats["mean_intensity"]),
            float(surrounding_stats["mean_intensity"]),
            background_noise,
        )
        row["cnr_local"] = cnr_local(
            float(bone_stats["mean_intensity"]),
            float(surrounding_stats["mean_intensity"]),
            bone_stats["std_intensity"],
            surrounding_stats["std_intensity"],
        )
        if row["cnr_background"] is None:
            row["cnr_background_reason"] = "背景 ROI 方差为 0" if background_noise is not None and background_noise <= EPSILON else "缺少有效 Background ROI"
        if row["cnr_local"] is None:
            row["cnr_local_reason"] = "Bone 或 Surrounding ROI 局部方差为 0"
        if row["cnr_background"] is None or row["cnr_local"] is None:
            row["status"] = "partial"
            reasons = []
            if row["cnr_background"] is None:
                reasons.append("Background CNR：" + row["cnr_background_reason"])
            if row["cnr_local"] is None:
                reasons.append("Local CNR：" + row["cnr_local_reason"])
            row["message"] = "；".join(reasons)

    weak_rows = [row for row in roi_rows if row["roi_type"] == "weak_bone"]
    strong_rows = [row for row in roi_rows if row["roi_type"] == "strong_bone"]
    all_bone_rows = weak_rows + strong_rows
    summary: dict[str, Any] = {
        "scheme_id": record.id,
        "scheme_name": record.display_name,
        "role": record.role,
        "weak_bone_mean_cnr_background": _mean_valid(row["cnr_background"] for row in weak_rows),
        "weak_bone_mean_cnr_local": _mean_valid(row["cnr_local"] for row in weak_rows),
        "strong_bone_mean_cnr_background": _mean_valid(row["cnr_background"] for row in strong_rows),
        "strong_bone_mean_cnr_local": _mean_valid(row["cnr_local"] for row in strong_rows),
        "all_bone_mean_cnr_background": _mean_valid(row["cnr_background"] for row in all_bone_rows),
        "all_bone_mean_cnr_local": _mean_valid(row["cnr_local"] for row in all_bone_rows),
        "weak_bone_mean_average_gradient": _mean_valid(row["average_gradient_roi"] for row in weak_rows),
        "strong_bone_mean_average_gradient": _mean_valid(row["average_gradient_roi"] for row in strong_rows),
        "average_gradient_global": average_gradient(image),
        "background_noise_pooled": background_noise,
        "background_noise_reason": (
            "缺少有效 Background ROI" if background_noise is None
            else "背景 ROI 方差为 0，不能有效估计噪声" if background_noise <= EPSILON else ""
        ),
        "strong_bone_saturation_mean": _mean_valid(row["saturation_ratio"] for row in strong_rows),
        "ssim": None,
        "ssim_distance_from_original": None,
        "primary_cnr": None,
        "primary_cnr_change_percent": None,
        "weak_bone_mean_average_gradient_change_percent": None,
        "background_noise_change_percent": None,
        "saturation_change_pp": None,
        "weak_bone_cnr_background_reason": "",
        "weak_bone_cnr_local_reason": "",
        "saturation_reason": "" if strong_rows else "未创建 Strong Bone ROI",
    }
    if summary["weak_bone_mean_cnr_background"] is None:
        summary["weak_bone_cnr_background_reason"] = (
            "背景 ROI 方差为 0" if background_noise is not None and background_noise <= EPSILON
            else "Weak Bone 未配对或缺少有效 Background ROI"
        )
    if summary["weak_bone_mean_cnr_local"] is None:
        summary["weak_bone_cnr_local_reason"] = "Weak Bone 未配对或局部方差为 0"
    return summary, roi_rows


def evaluate_project(store: ProjectStore) -> dict[str, Any]:
    project = store.project
    validation = validate_project(store)
    if project.original is None:
        raise ValueError("尚未导入原图")
    blocking_by_id: dict[str, list[str]] = {}
    for issue in validation:
        if issue.severity == "error":
            blocking_by_id.setdefault(issue.object_id, []).append(issue.message)
    if project.original.id in blocking_by_id or project.id in blocking_by_id:
        raise ValueError("原图或项目验证失败：" + "；".join(blocking_by_id.get(project.original.id, []) + blocking_by_id.get(project.id, [])))

    config = project.evaluation_config
    original_array = load_grayscale(store.resolve(project.original.relative_path))
    records = [project.original] + project.algorithms
    summaries: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []

    for record in records:
        if record.id in blocking_by_id:
            summaries.append({
                "scheme_id": record.id,
                "scheme_name": record.display_name,
                "role": record.role,
                "status": "failed",
                "message": "；".join(blocking_by_id[record.id]),
            })
            continue
        image = original_array if record.role == "original" else load_grayscale(store.resolve(record.relative_path))
        summary, details = _evaluate_image(
            image,
            record,
            project.rois,
            config.gray_min,
            config.gray_max,
            config.saturation_threshold_ratio,
        )
        if record.role == "original":
            summary["ssim"] = 1.0
            summary["ssim_distance_from_original"] = 0.0
        else:
            score = structural_similarity(original_array, image, float(config.gray_max - config.gray_min))
            summary["ssim"] = score
            summary["ssim_distance_from_original"] = None if score is None else 1.0 - score
        summaries.append(summary)
        roi_rows.extend(details)

    baseline = next(row for row in summaries if row["role"] == "original")
    primary_key = "weak_bone_mean_cnr_background" if config.primary_cnr == "background" else "weak_bone_mean_cnr_local"
    for row in summaries:
        if row.get("status") == "failed":
            continue
        row["primary_cnr"] = row.get(primary_key)
        if row["role"] != "original":
            row["primary_cnr_change_percent"] = relative_change(row.get("primary_cnr"), baseline.get(primary_key))
            row["weak_bone_mean_average_gradient_change_percent"] = relative_change(
                row.get("weak_bone_mean_average_gradient"), baseline.get("weak_bone_mean_average_gradient")
            )
            row["background_noise_change_percent"] = relative_change(
                row.get("background_noise_pooled"), baseline.get("background_noise_pooled")
            )
            value = row.get("strong_bone_saturation_mean")
            original_value = baseline.get("strong_bone_saturation_mean")
            row["saturation_change_pp"] = None if value is None or original_value is None else (value - original_value) * 100.0

    validation_warnings = [issue.message for issue in validation if issue.severity == "warning"]
    for row in summaries:
        if row.get("status") == "failed":
            continue
        row["status"], row["message"] = _metric_status(row, validation_warnings)
        if config.primary_cnr == "background" and row.get("primary_cnr") is None and row.get("weak_bone_mean_cnr_local") is not None:
            row["message"] = (row.get("message", "") + "；当前 Background CNR 不可用，可切换到 Local CNR。").strip("；")

    evaluation_id = f"eval-{uuid4().hex}"
    result = {
        "schema_version": 1,
        "metric_spec_version": METRIC_SPEC_VERSION,
        "program_version": __version__,
        "evaluation_id": evaluation_id,
        "project_id": project.id,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "input_hashes": {record.id: record.sha256 for record in records},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "ssim_implementation": "bone_iqa.metrics.structural_similarity",
        },
        "validation": [issue.to_dict() for issue in validation],
        "metrics": summaries,
        "roi_metrics": roi_rows,
    }
    store.save_evaluation(result)
    return result


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def export_csv_bundle(store: ProjectStore, destination: Path) -> list[Path]:
    evaluation = store.load_latest_evaluation()
    if evaluation is None:
        raise ValueError("尚无可导出的评价结果")
    destination.mkdir(parents=True, exist_ok=True)
    common = {
        "project_id": evaluation["project_id"],
        "evaluation_id": evaluation["evaluation_id"],
        "metric_spec_version": evaluation["metric_spec_version"],
    }
    metric_rows = [{**common, **row} for row in evaluation["metrics"]]
    roi_rows = [{**common, **row} for row in evaluation["roi_metrics"]]
    validation_rows = [{**common, **row} for row in evaluation["validation"]]
    paths = [destination / "metrics.csv", destination / "roi_metrics.csv", destination / "validation.csv"]
    _write_csv(paths[0], metric_rows)
    _write_csv(paths[1], roi_rows)
    _write_csv(paths[2], validation_rows)
    with (destination / "evaluation.json").open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(evaluation, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    paths.append(destination / "evaluation.json")
    return paths
