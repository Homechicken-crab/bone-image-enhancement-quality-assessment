from __future__ import annotations

from typing import Any

from .models import ROI


ROI_TEMPLATE_VERSION = "ROI Template v1"
ROI_TEMPLATE_SIZE = (500, 800)  # width, height

_TEMPLATE_ROWS = (
    ("弱骨骼1", "weak_bone", 43, 190, 9, 60, "邻域1"),
    ("邻域1", "surrounding", 53, 190, 9, 60, None),
    ("弱骨骼2", "weak_bone", 164, 625, 9, 70, "邻域2"),
    ("邻域2", "surrounding", 154, 625, 9, 70, None),
    ("弱骨骼3", "weak_bone", 283, 190, 9, 60, "邻域3"),
    ("邻域3", "surrounding", 294, 190, 9, 60, None),
    ("弱骨骼4", "weak_bone", 415, 625, 9, 70, "邻域4"),
    ("邻域4", "surrounding", 406, 625, 9, 70, None),
    ("强骨骼1", "strong_bone", 121, 165, 18, 45, None),
    ("强骨骼2", "strong_bone", 367, 245, 18, 50, None),
    ("背景1", "background", 10, 180, 20, 140, None),
    ("背景2", "background", 472, 180, 20, 140, None),
)


def build_roi_template() -> list[ROI]:
    """Return a fresh, deterministic-layout copy of ROI Template v1."""
    by_name: dict[str, ROI] = {}
    for name, roi_type, x, y, width, height, _pair_name in _TEMPLATE_ROWS:
        by_name[name] = ROI.create(name, roi_type, x, y, width, height)
    for name, _roi_type, _x, _y, _width, _height, pair_name in _TEMPLATE_ROWS:
        if pair_name:
            by_name[name].paired_surrounding_roi_id = by_name[pair_name].id
            by_name[pair_name].paired_surrounding_roi_id = by_name[name].id
    return [by_name[name] for name, *_ in _TEMPLATE_ROWS]


def roi_template_snapshot(rois: list[ROI]) -> list[dict[str, Any]]:
    """Serialize ROI semantics without volatile IDs for cache comparison."""
    by_id = {roi.id: roi for roi in rois}
    rows = []
    for roi in rois:
        pair = by_id.get(roi.paired_surrounding_roi_id or "")
        rows.append({
            "name": roi.name,
            "type": roi.type,
            "x": roi.x,
            "y": roi.y,
            "width": roi.width,
            "height": roi.height,
            "paired_surrounding_name": pair.name if pair else "",
        })
    return rows


def roi_configuration_payload(rois: list[ROI]) -> list[dict[str, Any]]:
    return roi_template_snapshot(rois)


def is_template_v1(rois: list[ROI], width: int, height: int) -> bool:
    if (width, height) != ROI_TEMPLATE_SIZE:
        return False
    expected = roi_template_snapshot(build_roi_template())
    return roi_template_snapshot(rois) == expected


def template_version_for(rois: list[ROI], width: int | None = None, height: int | None = None) -> str:
    if width is not None and height is not None and is_template_v1(rois, width, height):
        return ROI_TEMPLATE_VERSION
    return f"{ROI_TEMPLATE_VERSION} (modified)" if rois else "manual"
