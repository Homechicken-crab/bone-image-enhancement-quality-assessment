from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import EPSILON, average_gradient, roi_statistics
from .models import ROI


@dataclass
class ROIRecommendation:
    roi: ROI
    score: float
    quality: str
    explanation: str


@dataclass
class _Candidate:
    x: int
    y: int
    width: int
    height: int
    score: float
    mean: float
    std: float
    gradient: float
    unique: int
    nonzero_fraction: float


def recommend_rois(
    original: np.ndarray,
    max_background: int = 3,
    max_strong: int = 2,
    max_weak: int = 4,
) -> list[ROIRecommendation]:
    """Recommend explainable rectangle ROIs from the original image only.

    Scores combine local intensity, standard deviation, gradient, contrast and
    spatial separation. Recommendations are candidates requiring user review;
    they are not anatomical segmentation or diagnosis.
    """
    image = np.asarray(original)
    if image.ndim != 2 or min(image.shape) < 16:
        return []
    values = image.astype(np.float64)
    height, width = values.shape
    dtype_max = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else max(float(values.max()), 1.0)
    positive = values[values > 0]
    robust_high = float(np.percentile(positive, 95)) if positive.size else dtype_max
    robust_high = max(robust_high, dtype_max * 0.02, 1.0)

    gradient_map = _gradient_map(values)
    grad_scale = max(float(np.percentile(gradient_map, 95)), 1.0)
    bg_w = _bounded_size(width, 0.10)
    bg_h = _bounded_size(height, 0.07)
    bone_w = _bounded_size(width, 0.12)
    bone_h = _bounded_size(height, 0.08)

    background_pool: list[_Candidate] = []
    for x, y in _grid(width, height, bg_w, bg_h):
        border = x < width * 0.28 or x + bg_w > width * 0.72 or y < height * 0.22 or y + bg_h > height * 0.88
        if not border:
            continue
        candidate = _describe(values, gradient_map, x, y, bg_w, bg_h)
        if candidate.std <= EPSILON or candidate.unique < 3 or candidate.mean <= 0:
            continue
        mean_norm = min(candidate.mean / robust_high, 1.0)
        grad_norm = min(candidate.gradient / grad_scale, 1.0)
        if mean_norm > 0.45 or grad_norm > 0.45:
            continue
        variance_quality = min(candidate.std / max(robust_high * 0.03, 1.0), 1.0)
        border_distance = min(x, y, width - (x + bg_w), height - (y + bg_h))
        edge_bonus = 1.0 - min(max(border_distance, 0) / max(min(width, height) * 0.25, 1.0), 1.0)
        candidate.score = 0.35 * (1.0 - mean_norm) + 0.25 * (1.0 - grad_norm) + 0.25 * variance_quality + 0.15 * edge_bonus
        background_pool.append(candidate)
    backgrounds = _select_spread(background_pool, max_background, width, height, min_distance=0.22)

    high_threshold = float(np.percentile(positive, 72)) if positive.size else robust_high
    strong_pool: list[_Candidate] = []
    weak_pool: list[_Candidate] = []
    for x, y in _grid(width, height, bone_w, bone_h):
        candidate = _describe(values, gradient_map, x, y, bone_w, bone_h)
        region = values[y : y + bone_h, x : x + bone_w]
        high_fraction = float(np.mean(region >= high_threshold)) if region.size else 0.0
        mean_norm = min(candidate.mean / robust_high, 1.0)
        grad_norm = min(candidate.gradient / grad_scale, 1.0)
        if candidate.nonzero_fraction >= 0.10 and candidate.unique >= 3:
            candidate.score = 0.50 * mean_norm + 0.30 * high_fraction + 0.20 * grad_norm
            if mean_norm >= 0.32 or high_fraction >= 0.20:
                strong_pool.append(candidate)

        expanded = _expanded_region(values, x, y, bone_w, bone_h)
        surrounding_mean = float(np.mean(expanded)) if expanded.size else candidate.mean
        contrast = min(abs(candidate.mean - surrounding_mean) / robust_high, 1.0)
        middle_intensity = max(0.0, 1.0 - abs(mean_norm - 0.38) / 0.38)
        background_penalty = 1.0 if candidate.nonzero_fraction < 0.08 or candidate.unique < 3 else 0.0
        strong_penalty = max(0.0, (mean_norm - 0.68) / 0.32)
        candidate_weak = _Candidate(**{**candidate.__dict__})
        candidate_weak.score = 0.30 * middle_intensity + 0.35 * grad_norm + 0.25 * contrast + 0.10 * min(candidate.std / robust_high * 8.0, 1.0) - 0.55 * background_penalty - 0.35 * strong_penalty
        if candidate_weak.score > 0.12 and candidate.nonzero_fraction >= 0.08 and candidate.unique >= 3:
            weak_pool.append(candidate_weak)

    strong = _select_spread(strong_pool, max_strong, width, height, min_distance=0.20)
    weak_pool = [candidate for candidate in weak_pool if not any(_iou(candidate, item) > 0.12 for item in strong)]
    weak = _select_spread(weak_pool, max_weak, width, height, min_distance=0.16)

    recommendations: list[ROIRecommendation] = []
    for index, candidate in enumerate(backgrounds, 1):
        roi = ROI.create(f"推荐 Background {index}", "background", candidate.x, candidate.y, candidate.width, candidate.height)
        recommendations.append(ROIRecommendation(roi, candidate.score, "推荐", _background_explanation(candidate)))

    for roi_type, candidates, label in (("strong_bone", strong, "Strong Bone"), ("weak_bone", weak, "Weak Bone")):
        for index, candidate in enumerate(candidates, 1):
            bone = ROI.create(f"推荐 {label} {index}", roi_type, candidate.x, candidate.y, candidate.width, candidate.height)
            surrounding_candidate = _recommend_surrounding(values, gradient_map, candidate, robust_high, grad_scale)
            if surrounding_candidate is not None:
                surrounding = ROI.create(
                    f"{bone.name}-周围", "surrounding", surrounding_candidate.x, surrounding_candidate.y,
                    surrounding_candidate.width, surrounding_candidate.height,
                )
                bone.paired_surrounding_roi_id = surrounding.id
                recommendations.append(ROIRecommendation(
                    surrounding, surrounding_candidate.score, "推荐",
                    "相邻矩形；不与 Bone ROI 重叠；优先保留非恒定、较低梯度且仍含有效信号的区域。",
                ))
            explanation = (
                f"组合评分={candidate.score:.3f}；局部均值={candidate.mean:.3f}；"
                f"标准差={candidate.std:.3f}；平均梯度={candidate.gradient:.3f}；"
                f"非零像素比例={candidate.nonzero_fraction:.1%}。"
            )
            recommendations.append(ROIRecommendation(bone, candidate.score, "推荐", explanation))
    return recommendations


def _bounded_size(length: int, fraction: float) -> int:
    return max(8, min(length, int(round(length * fraction))))


def _grid(width: int, height: int, window_w: int, window_h: int) -> Iterable[tuple[int, int]]:
    step_x = max(4, window_w // 2)
    step_y = max(4, window_h // 2)
    xs = list(range(0, max(width - window_w + 1, 1), step_x))
    ys = list(range(0, max(height - window_h + 1, 1), step_y))
    if xs[-1] != width - window_w:
        xs.append(width - window_w)
    if ys[-1] != height - window_h:
        ys.append(height - window_h)
    for y in ys:
        for x in xs:
            yield x, y


def _gradient_map(values: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(values)
    dy = np.zeros_like(values)
    dx[:, :-1] = values[:, 1:] - values[:, :-1]
    dy[:-1, :] = values[1:, :] - values[:-1, :]
    return np.sqrt((dx * dx + dy * dy) / 2.0)


def _describe(values: np.ndarray, gradient: np.ndarray, x: int, y: int, width: int, height: int) -> _Candidate:
    region = values[y : y + height, x : x + width]
    stats = roi_statistics(region)
    return _Candidate(
        x, y, width, height, 0.0,
        float(stats["mean_intensity"] or 0.0),
        float(stats["std_intensity"] or 0.0),
        float(np.mean(gradient[y : y + height, x : x + width])),
        int(stats["unique_pixel_count"] or 0),
        float(np.mean(region > 0)) if region.size else 0.0,
    )


def _expanded_region(values: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    margin_x = max(2, width // 2)
    margin_y = max(2, height // 2)
    return values[max(0, y - margin_y) : min(values.shape[0], y + height + margin_y), max(0, x - margin_x) : min(values.shape[1], x + width + margin_x)]


def _select_spread(pool: list[_Candidate], count: int, width: int, height: int, min_distance: float) -> list[_Candidate]:
    selected: list[_Candidate] = []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    for candidate in sorted(pool, key=lambda item: item.score, reverse=True):
        cx = candidate.x + candidate.width / 2.0
        cy = candidate.y + candidate.height / 2.0
        if any(np.hypot(cx - (item.x + item.width / 2.0), cy - (item.y + item.height / 2.0)) / diagonal < min_distance for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= count:
            break
    return selected


def _iou(first: _Candidate, second: _Candidate) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _recommend_surrounding(
    values: np.ndarray,
    gradient: np.ndarray,
    bone: _Candidate,
    robust_high: float,
    grad_scale: float,
) -> _Candidate | None:
    height, width = values.shape
    positions = (
        (bone.x - bone.width, bone.y), (bone.x + bone.width, bone.y),
        (bone.x, bone.y - bone.height), (bone.x, bone.y + bone.height),
    )
    candidates: list[_Candidate] = []
    for x, y in positions:
        if x < 0 or y < 0 or x + bone.width > width or y + bone.height > height:
            continue
        candidate = _describe(values, gradient, x, y, bone.width, bone.height)
        if candidate.std <= EPSILON or candidate.unique < 2 or candidate.nonzero_fraction < 0.02:
            continue
        grad_norm = min(candidate.gradient / grad_scale, 1.0)
        mean_distance = min(abs(candidate.mean - bone.mean) / robust_high, 1.0)
        strong_penalty = max(0.0, candidate.mean / robust_high - 0.70)
        candidate.score = 0.45 * (1.0 - grad_norm) + 0.35 * (1.0 - mean_distance) + 0.20 * min(candidate.nonzero_fraction * 2.0, 1.0) - 0.40 * strong_penalty
        candidates.append(candidate)
    return max(candidates, key=lambda item: item.score) if candidates else None


def _background_explanation(candidate: _Candidate) -> str:
    return (
        f"低/中低灰度、低梯度且非恒定；均值={candidate.mean:.3f}，"
        f"标准差={candidate.std:.3f}，唯一灰度数={candidate.unique}，"
        f"平均梯度={candidate.gradient:.3f}。"
    )
