from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPSILON = 1e-12


def sample_std(values: np.ndarray) -> float | None:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size < 2:
        return None
    return float(np.std(flat, ddof=1))


def roi_statistics(values: np.ndarray) -> dict[str, float | int | None]:
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return {
            "pixel_count": 0,
            "unique_pixel_count": 0,
            "mean_intensity": None,
            "std_intensity": None,
            "min_intensity": None,
            "max_intensity": None,
        }
    return {
        "pixel_count": int(flat.size),
        "unique_pixel_count": int(np.unique(flat).size),
        "mean_intensity": float(np.mean(flat)),
        "std_intensity": sample_std(flat),
        "min_intensity": float(np.min(flat)),
        "max_intensity": float(np.max(flat)),
    }


def pooled_background_noise(stats: Iterable[dict[str, float | int | None]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for item in stats:
        count = int(item.get("pixel_count") or 0)
        std = item.get("std_intensity")
        if count >= 2 and std is not None:
            numerator += (count - 1) * float(std) ** 2
            denominator += count - 1
    if denominator <= 0:
        return None
    return math.sqrt(numerator / denominator)


def cnr_background(bone_mean: float, surrounding_mean: float, background_noise: float | None) -> float | None:
    if background_noise is None or background_noise <= EPSILON:
        return None
    return abs(float(bone_mean) - float(surrounding_mean)) / float(background_noise)


def cnr_local(
    bone_mean: float,
    surrounding_mean: float,
    bone_std: float | None,
    surrounding_std: float | None,
) -> float | None:
    if bone_std is None or surrounding_std is None:
        return None
    local_noise = math.sqrt((float(bone_std) ** 2 + float(surrounding_std) ** 2) / 2.0)
    if local_noise <= EPSILON:
        return None
    return abs(float(bone_mean) - float(surrounding_mean)) / local_noise


def average_gradient(image: np.ndarray) -> float | None:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        return None
    dx = values[:-1, 1:] - values[:-1, :-1]
    dy = values[1:, :-1] - values[:-1, :-1]
    return float(np.mean(np.sqrt((dx * dx + dy * dy) / 2.0)))


def saturation_ratio(values: np.ndarray, gray_min: float, gray_max: float, threshold_ratio: float = 0.98) -> tuple[float | None, int, float]:
    flat = np.asarray(values).ravel()
    threshold = float(gray_min) + float(threshold_ratio) * (float(gray_max) - float(gray_min))
    if flat.size == 0:
        return None, 0, threshold
    count = int(np.count_nonzero(flat >= threshold))
    return count / int(flat.size), count, threshold


def relative_change(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or abs(float(baseline)) <= EPSILON:
        return None
    return (float(value) - float(baseline)) / abs(float(baseline)) * 100.0


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def _gaussian_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    radius = kernel.size // 2
    height, width = values.shape
    padded_y = np.pad(values, ((radius, radius), (0, 0)), mode="reflect")
    vertical = np.zeros_like(values)
    for offset, weight in enumerate(kernel):
        vertical += weight * padded_y[offset : offset + height, :]
    padded_x = np.pad(vertical, ((0, 0), (radius, radius)), mode="reflect")
    result = np.zeros_like(values)
    for offset, weight in enumerate(kernel):
        result += weight * padded_x[:, offset : offset + width]
    return result


def structural_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
    data_range: float,
    win_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float | None:
    """Wang-style Gaussian SSIM with fixed population covariance parameters."""
    x = np.asarray(reference, dtype=np.float64)
    y = np.asarray(candidate, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        return None
    if min(x.shape) < win_size or data_range <= 0:
        return None
    kernel = _gaussian_kernel(win_size, sigma)
    ux = _gaussian_filter(x, kernel)
    uy = _gaussian_filter(y, kernel)
    uxx = _gaussian_filter(x * x, kernel)
    uyy = _gaussian_filter(y * y, kernel)
    uxy = _gaussian_filter(x * y, kernel)
    vx = np.maximum(uxx - ux * ux, 0.0)
    vy = np.maximum(uyy - uy * uy, 0.0)
    vxy = uxy - ux * uy
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    score_map = numerator / denominator
    pad = win_size // 2
    valid = score_map[pad:-pad, pad:-pad]
    if valid.size == 0:
        return None
    return float(np.mean(valid))
