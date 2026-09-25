from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


ROIType = Literal["weak_bone", "strong_bone", "surrounding", "background"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImageRecord:
    id: str
    role: Literal["original", "algorithm"]
    display_name: str
    relative_path: str
    sha256: str
    width: int
    height: int
    dtype: str
    channels: int = 1
    source_path: str = ""
    imported_at: str = field(default_factory=utc_now)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ImageRecord":
        return cls(**value)


@dataclass
class ROI:
    id: str
    name: str
    type: ROIType
    x: int
    y: int
    width: int
    height: int
    paired_surrounding_roi_id: str | None = None
    visible: bool = True
    notes: str = ""

    @classmethod
    def create(
        cls,
        name: str,
        roi_type: ROIType,
        x: int,
        y: int,
        width: int,
        height: int,
        paired_surrounding_roi_id: str | None = None,
    ) -> "ROI":
        return cls(
            id=f"roi-{uuid4().hex[:12]}",
            name=name,
            type=roi_type,
            x=int(x),
            y=int(y),
            width=int(width),
            height=int(height),
            paired_surrounding_roi_id=paired_surrounding_roi_id,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ROI":
        return cls(**value)

    def geometry(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass
class EvaluationConfig:
    gray_min: int = 0
    gray_max: int = 255
    primary_cnr: Literal["background", "local"] = "local"
    saturation_threshold_ratio: float = 0.98

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationConfig":
        return cls(**value)


@dataclass
class Project:
    id: str
    name: str
    schema_version: int = 1
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    original: ImageRecord | None = None
    algorithms: list[ImageRecord] = field(default_factory=list)
    rois: list[ROI] = field(default_factory=list)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    latest_evaluation_id: str | None = None

    @classmethod
    def create(cls, name: str) -> "Project":
        return cls(id=f"project-{uuid4().hex}", name=name)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Project":
        return cls(
            id=value["id"],
            name=value["name"],
            schema_version=value.get("schema_version", 1),
            created_at=value.get("created_at", utc_now()),
            updated_at=value.get("updated_at", utc_now()),
            original=ImageRecord.from_dict(value["original"]) if value.get("original") else None,
            algorithms=[ImageRecord.from_dict(v) for v in value.get("algorithms", [])],
            rois=[ROI.from_dict(v) for v in value.get("rois", [])],
            evaluation_config=EvaluationConfig.from_dict(value.get("evaluation_config", {})),
            latest_evaluation_id=value.get("latest_evaluation_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationIssue:
    severity: Literal["error", "warning", "info"]
    code: str
    object_id: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)
