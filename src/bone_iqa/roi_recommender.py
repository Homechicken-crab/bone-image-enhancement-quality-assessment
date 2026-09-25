from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .metrics import EPSILON, roi_statistics
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
    signal_mean: float
    std: float
    gradient: float
    unique: int
    nonzero_fraction: float
    edge_density: float
    high_fraction: float
    strong_overlap: float
    structure_continuity: float
    signal_overlap: float


def recommend_rois(
    original: np.ndarray,
    max_background: int = 3,
    max_strong: int = 2,
    max_weak: int = 4,
) -> list[ROIRecommendation]:
    """Generate reviewable rectangle candidates from the original image only.

    This is an explainable image-processing heuristic, not anatomical
    segmentation. Strong response is estimated from dynamic percentiles;
    Weak Bone candidates are required to contain structure while avoiding the
    dilated strong-response exclusion mask.
    """
    image = np.asarray(original)
    if image.ndim != 2 or min(image.shape) < 16:
        return []
    values = image.astype(np.float64)
    height, width = values.shape
    dtype_max = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else max(float(values.max()), 1.0)
    positive = values[values > 0]
    if positive.size < 8:
        return []
    robust_high = max(float(np.percentile(positive, 95)), dtype_max * 0.02, 1.0)
    weak_low = float(np.percentile(positive, 20))
    weak_high = float(np.percentile(positive, 68))
    high_threshold = float(np.percentile(positive, 80))

    gradient_map = _gradient_map(values)
    positive_gradients = gradient_map[values > 0]
    grad_scale = max(float(np.percentile(positive_gradients, 90)), 1.0)
    edge_threshold = max(float(np.percentile(positive_gradients, 62)), EPSILON)
    edge_mask = gradient_map >= edge_threshold
    strong_mask, strong_exclusion_mask = build_strong_response_masks(values, gradient_map)
    signal_mask = build_effective_signal_mask(values)

    bg_w = _bounded_size(width, 0.10)
    bg_h = _bounded_size(height, 0.07)
    bone_w = _bounded_size(width, 0.12)
    bone_h = _bounded_size(height, 0.08)

    # Background logic intentionally remains conservative and unchanged in spirit.
    background_pool: list[_Candidate] = []
    for x, y in _grid(width, height, bg_w, bg_h):
        border = x < width * 0.28 or x + bg_w > width * 0.72 or y < height * 0.22 or y + bg_h > height * 0.88
        if not border:
            continue
        candidate = _describe(values, gradient_map, edge_mask, strong_mask, high_threshold, x, y, bg_w, bg_h, signal_mask)
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
    backgrounds = _select_spread(background_pool, max_background, width, height, min_distance=0.20, max_iou=0.05)

    strong_pool: list[_Candidate] = []
    weak_pool: list[_Candidate] = []
    for x, y in _grid(width, height, bone_w, bone_h):
        candidate = _describe(values, gradient_map, edge_mask, strong_mask, high_threshold, x, y, bone_w, bone_h, signal_mask)
        signal_norm = min(candidate.signal_mean / robust_high, 1.0)
        grad_norm = min(candidate.gradient / grad_scale, 1.0)

        if candidate.nonzero_fraction >= 0.10 and candidate.unique >= 3 and candidate.strong_overlap >= 0.04:
            candidate.score = 0.45 * signal_norm + 0.30 * candidate.strong_overlap + 0.15 * candidate.high_fraction + 0.10 * grad_norm
            strong_pool.append(candidate)

        exclusion_overlap = float(np.mean(strong_exclusion_mask[y : y + bone_h, x : x + bone_w]))
        if exclusion_overlap > 0.18 or candidate.nonzero_fraction < 0.08 or candidate.unique < 3:
            continue
        moderate_intensity = _moderate_intensity_score(candidate.signal_mean, weak_low, weak_high, robust_high)
        expanded = _expanded_region(values, x, y, bone_w, bone_h)
        surrounding_mean = _ring_mean(expanded, values[y : y + bone_h, x : x + bone_w])
        local_contrast = min(abs(candidate.mean - surrounding_mean) / robust_high, 1.0)
        variance_score = min(candidate.std / max(robust_high * 0.12, 1.0), 1.0)
        background_penalty = max(0.0, (0.18 - candidate.nonzero_fraction) / 0.18)
        strong_penalty = min(exclusion_overlap / 0.18, 1.0) + max(0.0, (candidate.signal_mean - weak_high) / max(robust_high - weak_high, 1.0))
        texture_penalty = max(0.0, (candidate.edge_density - 0.58) / 0.42) * max(0.0, 1.0 - local_contrast)
        candidate.score = (
            0.22 * moderate_intensity
            + 0.24 * grad_norm
            + 0.16 * candidate.edge_density
            + 0.18 * local_contrast
            + 0.15 * candidate.structure_continuity
            + 0.05 * variance_score
            - 0.60 * strong_penalty
            - 0.50 * background_penalty
            - 0.20 * texture_penalty
        )
        if candidate.score > 0.16 and candidate.edge_density >= 0.025:
            weak_pool.append(candidate)

    strong = _select_spread(strong_pool, max_strong, width, height, min_distance=0.17, max_iou=0.10)
    weak_pool = [candidate for candidate in weak_pool if not any(_iou(candidate, item) > 0.08 for item in strong)]
    weak = _select_spread(weak_pool, max_weak, width, height, min_distance=0.13, max_iou=0.12, grid_coverage=True)

    background_profile = _candidate_profile(backgrounds)

    recommendations: list[ROIRecommendation] = []
    for index, candidate in enumerate(backgrounds, 1):
        roi = ROI.create(f"背景{index}", "background", candidate.x, candidate.y, candidate.width, candidate.height)
        recommendations.append(ROIRecommendation(roi, candidate.score, "推荐", _background_explanation(candidate)))

    for roi_type, candidates, label in (("strong_bone", strong, "强骨骼"), ("weak_bone", weak, "弱骨骼")):
        for index, candidate in enumerate(candidates, 1):
            bone = ROI.create(f"{label}{index}", roi_type, candidate.x, candidate.y, candidate.width, candidate.height)
            neighbor = _recommend_surrounding(
                values, gradient_map, edge_mask, strong_mask, high_threshold,
                candidate, robust_high, grad_scale, signal_mask, background_profile,
            )
            bone_quality = "推荐"
            if neighbor is not None:
                surrounding, quality, explanation = neighbor
                surrounding_roi = ROI.create(
                    f"{bone.name}-邻域", "surrounding", surrounding.x, surrounding.y,
                    surrounding.width, surrounding.height,
                )
                bone.paired_surrounding_roi_id = surrounding_roi.id
                recommendations.append(ROIRecommendation(surrounding_roi, surrounding.score, quality, explanation))
            else:
                bone_quality = "需检查：未找到可靠邻域"
            explanation = (
                f"组合评分={candidate.score:.3f}；局部信号均值={candidate.signal_mean:.3f}；"
                f"标准差={candidate.std:.3f}；平均梯度={candidate.gradient:.3f}；"
                f"边缘密度={candidate.edge_density:.1%}；结构连续性={candidate.structure_continuity:.3f}；"
                f"强响应重叠={candidate.strong_overlap:.1%}。"
            )
            recommendations.append(ROIRecommendation(bone, candidate.score, bone_quality, explanation))
    return recommendations


def recommend_surrounding_for_roi(original: np.ndarray, bone_roi: ROI) -> ROIRecommendation | None:
    """Recommend one local non-bone reference rectangle for a supplied Bone ROI."""
    image = np.asarray(original)
    if image.ndim != 2 or bone_roi.width <= 0 or bone_roi.height <= 0:
        return None
    values = image.astype(np.float64)
    positive = values[values > 0]
    if positive.size < 3:
        return None
    dtype_max = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else max(float(values.max()), 1.0)
    robust_high = max(float(np.percentile(positive, 95)), dtype_max * 0.02, 1.0)
    high_threshold = float(np.percentile(positive, 80))
    gradient = _gradient_map(values)
    positive_gradients = gradient[values > 0]
    grad_scale = max(float(np.percentile(positive_gradients, 90)), 1.0)
    edge_threshold = max(float(np.percentile(positive_gradients, 62)), EPSILON)
    edge_mask = gradient >= edge_threshold
    strong_mask, _ = build_strong_response_masks(values, gradient)
    signal_mask = build_effective_signal_mask(values)
    bone = _describe(
        values, gradient, edge_mask, strong_mask, high_threshold,
        bone_roi.x, bone_roi.y, bone_roi.width, bone_roi.height, signal_mask,
    )
    result = _recommend_surrounding(values, gradient, edge_mask, strong_mask, high_threshold, bone, robust_high, grad_scale, signal_mask, None)
    if result is None:
        return None
    candidate, quality, explanation = result
    roi = ROI.create(
        f"{bone_roi.name}-邻域", "surrounding", candidate.x, candidate.y,
        candidate.width, candidate.height,
    )
    return ROIRecommendation(roi, candidate.score, quality, explanation)


def remove_recommendation(
    recommendations: list[ROIRecommendation],
    roi_id: str,
    remove_orphan_neighbor: bool = False,
) -> list[ROIRecommendation]:
    """Remove a candidate and clear dependent pair IDs without leaving dangling links."""
    removed = next((item for item in recommendations if item.roi.id == roi_id), None)
    if removed is None:
        return list(recommendations)
    result = [item for item in recommendations if item.roi.id != roi_id]
    if removed.roi.type == "surrounding":
        for item in result:
            if item.roi.paired_surrounding_roi_id == roi_id:
                item.roi.paired_surrounding_roi_id = None
                item.quality = "需检查"
                item.explanation += " 对应 Surrounding 已删除，当前未配对。"
    elif remove_orphan_neighbor and removed.roi.type in {"weak_bone", "strong_bone"} and removed.roi.paired_surrounding_roi_id:
        pair_id = removed.roi.paired_surrounding_roi_id
        if not any(item.roi.paired_surrounding_roi_id == pair_id for item in result):
            result = [item for item in result if item.roi.id != pair_id]
    return result


def build_strong_response_masks(values: np.ndarray, gradient: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return a coarse strong-response mask and its dilated exclusion mask.

    All connected high-response groups are retained; no largest-component
    assumption is made, so dual-projection images remain supported.
    """
    image = np.asarray(values, dtype=np.float64)
    gradient_map = _gradient_map(image) if gradient is None else np.asarray(gradient, dtype=np.float64)
    positive = image[image > 0]
    if positive.size < 3:
        empty = np.zeros(image.shape, dtype=bool)
        return empty, empty.copy()
    high_threshold = float(np.percentile(positive, 80))
    gradient_values = gradient_map[image > 0]
    gradient_threshold = float(np.percentile(gradient_values, 55)) if gradient_values.size else 0.0
    high = image >= high_threshold
    neighbors = _neighbor_count(high)
    strong = high & ((neighbors >= 2) | ((gradient_map >= gradient_threshold) & (neighbors >= 1)))
    radius = max(1, min(4, int(round(min(image.shape) * 0.012))))
    return strong, _dilate(strong, radius)


def build_effective_signal_mask(values: np.ndarray) -> np.ndarray:
    """Build a conservative body/signal mask while retaining every component.

    The mask is based on a low positive percentile, local support and a small
    morphological closing.  It deliberately does not keep only the largest
    connected component, because whole-body scintigraphy may contain two
    projections or several separated limbs.
    """
    image = np.asarray(values, dtype=np.float64)
    positive = image[image > 0]
    if positive.size < 3:
        return np.zeros(image.shape, dtype=bool)
    low = float(np.percentile(positive, 10))
    middle = float(np.percentile(positive, 50))
    threshold = max(low + 0.08 * (middle - low), EPSILON)
    initial = image >= threshold
    supported = initial & (_neighbor_count(initial) >= 3)
    radius = max(1, min(3, int(round(min(image.shape) * 0.008))))
    closed = _erode(_dilate(supported, radius), radius)
    return closed | supported


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


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    result = np.zeros(mask.shape, dtype=np.uint8)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dx == 1 and dy == 1:
                continue
            result += padded[dy : dy + height, dx : dx + width]
    return result


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant")
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = expanded
    return result


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.astype(bool, copy=True)
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        shrunk = np.ones_like(result)
        for dy in range(3):
            for dx in range(3):
                shrunk &= padded[dy : dy + result.shape[0], dx : dx + result.shape[1]]
        result = shrunk
    return result


def _describe(
    values: np.ndarray,
    gradient: np.ndarray,
    edge_mask: np.ndarray,
    strong_mask: np.ndarray,
    high_threshold: float,
    x: int,
    y: int,
    width: int,
    height: int,
    signal_mask: np.ndarray | None = None,
) -> _Candidate:
    region = values[y : y + height, x : x + width]
    stats = roi_statistics(region)
    nonzero = region[region > 0]
    return _Candidate(
        x=x, y=y, width=width, height=height, score=0.0,
        mean=float(stats["mean_intensity"] or 0.0),
        signal_mean=float(np.mean(nonzero)) if nonzero.size else 0.0,
        std=float(stats["std_intensity"] or 0.0),
        gradient=float(np.mean(gradient[y : y + height, x : x + width])),
        unique=int(stats["unique_pixel_count"] or 0),
        nonzero_fraction=float(np.mean(region > 0)) if region.size else 0.0,
        edge_density=float(np.mean(edge_mask[y : y + height, x : x + width])),
        high_fraction=float(np.mean(region >= high_threshold)) if region.size else 0.0,
        strong_overlap=float(np.mean(strong_mask[y : y + height, x : x + width])),
        structure_continuity=_structure_continuity(region, edge_mask[y : y + height, x : x + width]),
        signal_overlap=float(np.mean(signal_mask[y : y + height, x : x + width])) if signal_mask is not None else float(np.mean(region > 0)),
    )


def _structure_continuity(region: np.ndarray, edge_region: np.ndarray) -> float:
    """Score coherent, connected edge structure above similarly strong noise."""
    if region.size < 4 or not np.any(edge_region):
        return 0.0
    dx = np.zeros_like(region, dtype=np.float64)
    dy = np.zeros_like(region, dtype=np.float64)
    dx[:, :-1] = region[:, 1:] - region[:, :-1]
    dy[:-1, :] = region[1:, :] - region[:-1, :]
    selected = edge_region & ((dx != 0) | (dy != 0))
    if np.count_nonzero(selected) < 2:
        return 0.0
    angles = np.arctan2(dy[selected], dx[selected])
    direction_consistency = float(np.hypot(np.mean(np.cos(2.0 * angles)), np.mean(np.sin(2.0 * angles))))
    connected = float(np.mean(_neighbor_count(edge_region)[edge_region] >= 1))
    return min(max(0.65 * direction_consistency + 0.35 * connected, 0.0), 1.0)


def _candidate_profile(candidates: list[_Candidate]) -> tuple[float, float, float] | None:
    if not candidates:
        return None
    return (
        float(np.median([item.mean for item in candidates])),
        float(np.median([item.std for item in candidates])),
        float(np.median([item.gradient for item in candidates])),
    )


def _moderate_intensity_score(value: float, low: float, high: float, robust_high: float) -> float:
    if value <= 0:
        return 0.0
    if value < low:
        return max(0.0, value / max(low, EPSILON))
    if value <= high:
        return 1.0
    return max(0.0, 1.0 - (value - high) / max(robust_high - high, 1.0))


def _expanded_region(values: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    margin_x = max(2, width // 2)
    margin_y = max(2, height // 2)
    return values[max(0, y - margin_y) : min(values.shape[0], y + height + margin_y), max(0, x - margin_x) : min(values.shape[1], x + width + margin_x)]


def _ring_mean(expanded: np.ndarray, center: np.ndarray) -> float:
    if expanded.size <= center.size:
        return float(np.mean(expanded)) if expanded.size else 0.0
    return float((np.sum(expanded, dtype=np.float64) - np.sum(center, dtype=np.float64)) / (expanded.size - center.size))


def _select_spread(
    pool: list[_Candidate],
    count: int,
    width: int,
    height: int,
    min_distance: float,
    max_iou: float,
    grid_coverage: bool = False,
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    remaining = list(pool)
    diagonal = max(float(np.hypot(width, height)), 1.0)
    while remaining and len(selected) < count:
        eligible: list[tuple[float, _Candidate]] = []
        for candidate in remaining:
            cx = candidate.x + candidate.width / 2.0
            cy = candidate.y + candidate.height / 2.0
            if selected:
                distances = [np.hypot(cx - (item.x + item.width / 2.0), cy - (item.y + item.height / 2.0)) / diagonal for item in selected]
                nearest_distance = min(distances)
                if nearest_distance < min_distance or any(_iou(candidate, item) > max_iou for item in selected):
                    continue
            else:
                nearest_distance = 0.0
            coverage_adjustment = 0.0
            if grid_coverage:
                cell = (min(2, int(3 * cx / max(width, 1))), min(2, int(3 * cy / max(height, 1))))
                used_cells = {
                    (min(2, int(3 * (item.x + item.width / 2.0) / max(width, 1))),
                     min(2, int(3 * (item.y + item.height / 2.0) / max(height, 1))))
                    for item in selected
                }
                coverage_adjustment = 0.10 if cell not in used_cells else -0.08
            eligible.append((candidate.score + 0.30 * nearest_distance + coverage_adjustment, candidate))
        if not eligible:
            break
        chosen = max(eligible, key=lambda item: item[0])[1]
        selected.append(chosen)
        remaining.remove(chosen)
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
    edge_mask: np.ndarray,
    strong_mask: np.ndarray,
    high_threshold: float,
    bone: _Candidate,
    robust_high: float,
    grad_scale: float,
    signal_mask: np.ndarray,
    background_profile: tuple[float, float, float] | None,
) -> tuple[_Candidate, str, str] | None:
    height, width = values.shape
    w, h = bone.width, bone.height
    positions = (
        (bone.x - w, bone.y, 0.00), (bone.x + w, bone.y, 0.00),
        (bone.x, bone.y - h, 0.00), (bone.x, bone.y + h, 0.00),
        (bone.x - w, bone.y - h, 0.08), (bone.x + w, bone.y - h, 0.08),
        (bone.x - w, bone.y + h, 0.08), (bone.x + w, bone.y + h, 0.08),
    )
    candidates: list[tuple[_Candidate, float, float]] = []
    for x, y, distance_penalty in positions:
        if x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        candidate = _describe(values, gradient, edge_mask, strong_mask, high_threshold, x, y, w, h, signal_mask)
        if candidate.std <= EPSILON or candidate.unique < 2 or candidate.nonzero_fraction < 0.02 or candidate.signal_overlap < 0.08:
            continue
        if candidate.strong_overlap > 0.25 or candidate.high_fraction > 0.45 or candidate.edge_density > 0.58:
            continue
        ratio = candidate.mean / max(bone.mean, EPSILON)
        intensity_relation = max(0.0, 1.0 - abs(ratio - 0.55) / 0.55) if 0.0 < ratio < 1.05 else 0.0
        lower_gradient = max(0.0, 1.0 - candidate.gradient / max(bone.gradient, grad_scale * 0.10, EPSILON))
        low_edge = max(0.0, 1.0 - candidate.edge_density / max(bone.edge_density, 0.10))
        effective_signal = min(candidate.nonzero_fraction / 0.35, 1.0)
        nonconstant = min(candidate.std / max(robust_high * 0.04, 1.0), 1.0)
        background_similarity = _background_similarity(candidate, background_profile, robust_high, grad_scale)
        candidate.score = (
            0.27 * intensity_relation
            + 0.20 * lower_gradient
            + 0.14 * low_edge
            + 0.17 * effective_signal
            + 0.14 * candidate.signal_overlap
            + 0.08 * nonconstant
            - 0.55 * candidate.strong_overlap
            - 0.45 * background_similarity
            - distance_penalty
        )
        candidates.append((candidate, ratio, background_similarity))
    if not candidates:
        return None
    candidate, ratio, background_similarity = max(candidates, key=lambda item: item[0].score)
    quality = "推荐" if (
        ratio < 0.95
        and candidate.strong_overlap < 0.10
        and candidate.edge_density < 0.30
        and candidate.gradient < bone.gradient
        and candidate.signal_overlap >= 0.30
        and background_similarity < 0.55
    ) else "需检查"
    explanation = (
        f"相邻非骨参考候选；均值/Bone均值={ratio:.3f}，平均梯度={candidate.gradient:.3f}，"
        f"边缘密度={candidate.edge_density:.1%}，高响应比例={candidate.high_fraction:.1%}，"
        f"强响应重叠={candidate.strong_overlap:.1%}，有效信号重叠={candidate.signal_overlap:.1%}，"
        f"背景相似度惩罚={background_similarity:.3f}。不再奖励与 Bone 灰度接近。"
    )
    return candidate, quality, explanation


def _background_similarity(
    candidate: _Candidate,
    profile: tuple[float, float, float] | None,
    robust_high: float,
    grad_scale: float,
) -> float:
    low_signal = 1.0 - candidate.signal_overlap
    if profile is None:
        appearance = (
            max(0.0, 1.0 - candidate.mean / max(robust_high * 0.35, 1.0))
            + max(0.0, 1.0 - candidate.gradient / max(grad_scale * 0.35, 1.0))
        ) / 2.0
    else:
        mean, std, gradient = profile
        mean_match = max(0.0, 1.0 - abs(candidate.mean - mean) / max(robust_high * 0.25, 1.0))
        std_match = max(0.0, 1.0 - abs(candidate.std - std) / max(robust_high * 0.08, 1.0))
        gradient_match = max(0.0, 1.0 - abs(candidate.gradient - gradient) / max(grad_scale * 0.30, 1.0))
        appearance = (mean_match + std_match + gradient_match) / 3.0
    return min(max(0.65 * appearance + 0.35 * low_signal, 0.0), 1.0)


def _background_explanation(candidate: _Candidate) -> str:
    return (
        f"低/中低灰度、低梯度且非恒定；均值={candidate.mean:.3f}，"
        f"标准差={candidate.std:.3f}，唯一灰度数={candidate.unique}，"
        f"平均梯度={candidate.gradient:.3f}。"
    )
