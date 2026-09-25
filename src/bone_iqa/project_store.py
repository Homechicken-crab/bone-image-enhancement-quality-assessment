from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .images import dtype_range, image_metadata
from .models import ImageRecord, Project, ROI, utc_now


PROJECT_FILE = "project.json"


class ProjectError(RuntimeError):
    pass


def _atomic_json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


class ProjectStore:
    def __init__(self, root: Path, project: Project):
        self.root = root.resolve()
        self.project = project

    @classmethod
    def create(cls, root: Path, name: str) -> "ProjectStore":
        root = root.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / PROJECT_FILE).exists():
            raise ProjectError("目标目录中已经存在项目")
        for relative in ("images/original", "images/algorithms", "evaluations", "exports"):
            (root / relative).mkdir(parents=True, exist_ok=True)
        store = cls(root, Project.create(name.strip() or root.name))
        store.save()
        return store

    @classmethod
    def open(cls, root_or_file: Path) -> "ProjectStore":
        path = root_or_file.resolve()
        project_path = path if path.name == PROJECT_FILE else path / PROJECT_FILE
        if not project_path.exists():
            raise ProjectError("未找到 project.json")
        with project_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream)
        version = int(raw.get("schema_version", 1))
        if version > 1:
            raise ProjectError(f"项目 schema_version={version} 高于当前程序支持版本")
        return cls(project_path.parent, Project.from_dict(raw))

    def save(self) -> None:
        self.project.updated_at = utc_now()
        _atomic_json_write(self.root / PROJECT_FILE, self.project.to_dict())

    def resolve(self, relative_path: str) -> Path:
        result = (self.root / relative_path).resolve()
        try:
            result.relative_to(self.root)
        except ValueError as exc:
            raise ProjectError("项目文件路径越出项目目录") from exc
        return result

    def _copy_image(self, source: Path, relative_directory: Path, image_id: str) -> tuple[str, dict[str, object]]:
        source = source.resolve()
        if not source.is_file():
            raise ProjectError("选择的图像文件不存在")
        metadata = image_metadata(source)
        extension = source.suffix.lower() or ".img"
        relative = relative_directory / f"{image_id}{extension}"
        destination = self.root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return relative.as_posix(), metadata

    def set_original(self, source: Path, display_name: str = "原图") -> ImageRecord:
        image_id = "original"
        relative, metadata = self._copy_image(source, Path("images/original"), image_id)
        record = ImageRecord(
            id=image_id,
            role="original",
            display_name=display_name.strip() or "原图",
            relative_path=relative,
            source_path=str(source.resolve()),
            **metadata,
        )
        self.project.original = record
        gray_min, gray_max = dtype_range(record.dtype)
        self.project.evaluation_config.gray_min = gray_min
        self.project.evaluation_config.gray_max = gray_max
        self.project.latest_evaluation_id = None
        self.save()
        return record

    def add_algorithm(self, source: Path, display_name: str) -> ImageRecord:
        name = display_name.strip()
        if not name:
            raise ProjectError("方案名称不能为空")
        if any(item.display_name == name for item in self.project.algorithms):
            raise ProjectError("方案名称已存在")
        image_id = f"alg-{uuid4().hex[:12]}"
        relative, metadata = self._copy_image(source, Path(f"images/algorithms/{image_id}"), "result")
        record = ImageRecord(
            id=image_id,
            role="algorithm",
            display_name=name,
            relative_path=relative,
            source_path=str(source.resolve()),
            **metadata,
        )
        self.project.algorithms.append(record)
        self.project.latest_evaluation_id = None
        self.save()
        return record

    def replace_algorithm(self, algorithm_id: str, source: Path) -> ImageRecord:
        record = self.get_algorithm(algorithm_id)
        relative, metadata = self._copy_image(source, Path(f"images/algorithms/{record.id}"), "result")
        record.relative_path = relative
        record.source_path = str(source.resolve())
        record.sha256 = str(metadata["sha256"])
        record.width = int(metadata["width"])
        record.height = int(metadata["height"])
        record.dtype = str(metadata["dtype"])
        record.channels = int(metadata["channels"])
        record.imported_at = utc_now()
        self.project.latest_evaluation_id = None
        self.save()
        return record

    def get_algorithm(self, algorithm_id: str) -> ImageRecord:
        for item in self.project.algorithms:
            if item.id == algorithm_id:
                return item
        raise ProjectError("未找到算法方案")

    def rename_algorithm(self, algorithm_id: str, new_name: str) -> None:
        name = new_name.strip()
        if not name:
            raise ProjectError("方案名称不能为空")
        if any(item.id != algorithm_id and item.display_name == name for item in self.project.algorithms):
            raise ProjectError("方案名称已存在")
        self.get_algorithm(algorithm_id).display_name = name
        evaluation_path = self.root / "evaluations/latest.json"
        if self.project.latest_evaluation_id and evaluation_path.exists():
            try:
                with evaluation_path.open("r", encoding="utf-8") as stream:
                    evaluation = json.load(stream)
                for section in ("metrics", "roi_metrics"):
                    for row in evaluation.get(section, []):
                        if row.get("scheme_id") == algorithm_id:
                            row["scheme_name"] = name
                _atomic_json_write(evaluation_path, evaluation)
            except (OSError, ValueError, TypeError):
                # Name synchronization is optional; force reevaluation if it cannot be done safely.
                self.project.latest_evaluation_id = None
        self.save()

    def remove_algorithm(self, algorithm_id: str) -> None:
        record = self.get_algorithm(algorithm_id)
        self.project.algorithms = [item for item in self.project.algorithms if item.id != algorithm_id]
        self.project.latest_evaluation_id = None
        self.save()
        # Do not destructively remove imported files; orphan cleanup can be added later.
        _ = record

    def upsert_roi(self, roi: ROI) -> None:
        if not roi.name.strip():
            raise ProjectError("ROI 名称不能为空")
        if roi.type in {"weak_bone", "strong_bone"} and not roi.paired_surrounding_roi_id:
            roi.paired_surrounding_roi_id = nearest_surrounding_roi_id(self.project.rois, roi)
        existing = next((index for index, item in enumerate(self.project.rois) if item.id == roi.id), None)
        if existing is None:
            self.project.rois.append(roi)
        else:
            self.project.rois[existing] = roi
        self.project.latest_evaluation_id = None
        self.save()

    def remove_roi(self, roi_id: str) -> None:
        self.project.rois = [item for item in self.project.rois if item.id != roi_id]
        for item in self.project.rois:
            if item.paired_surrounding_roi_id == roi_id:
                item.paired_surrounding_roi_id = None
        self.project.latest_evaluation_id = None
        self.save()

    def clear_rois(self) -> None:
        """Remove every saved ROI and invalidate the latest evaluation."""
        self.project.rois.clear()
        self.project.latest_evaluation_id = None
        self.save()

    def save_evaluation(self, evaluation: dict[str, Any]) -> Path:
        path = self.root / "evaluations/latest.json"
        _atomic_json_write(path, evaluation)
        self.project.latest_evaluation_id = str(evaluation["evaluation_id"])
        self.save()
        return path

    def load_latest_evaluation(self) -> dict[str, Any] | None:
        if not self.project.latest_evaluation_id:
            return None
        path = self.root / "evaluations/latest.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as stream:
            evaluation = json.load(stream)
        if evaluation.get("evaluation_id") != self.project.latest_evaluation_id:
            return None
        return evaluation


def nearest_surrounding_roi_id(rois: list[ROI], bone_roi: ROI) -> str | None:
    """Return the only or center-nearest Surrounding ROI for a Bone ROI."""
    surroundings = [roi for roi in rois if roi.type == "surrounding" and roi.id != bone_roi.id]
    if not surroundings:
        return None
    bone_center = (bone_roi.x + bone_roi.width / 2.0, bone_roi.y + bone_roi.height / 2.0)
    return min(
        surroundings,
        key=lambda roi: (
            (roi.x + roi.width / 2.0 - bone_center[0]) ** 2
            + (roi.y + roi.height / 2.0 - bone_center[1]) ** 2,
            roi.id,
        ),
    ).id
