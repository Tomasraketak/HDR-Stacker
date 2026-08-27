"""
Project files: save a whole editing session and reopen it later.

A project stores which photographs were loaded, the alignment worked out for
each of them, the crop, and every processing setting — so a session that took
an hour of manual nudging can be reopened exactly as it was left.

The format is plain JSON with a version field, so a project written by an older
build stays readable and unknown future keys are ignored rather than fatal.
Image data is never copied into the project: only paths, which keeps the file
tiny and keeps the photographs the single source of truth.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from core.exif_and_analysis import ExposureItem
except ImportError:  # pragma: no cover
    from .exif_and_analysis import ExposureItem

PROJECT_EXTENSION = ".ahdrproj"
PROJECT_FILTER = f"Projekt Astro HDR Stacker (*{PROJECT_EXTENSION});;Všechny soubory (*.*)"

# Bumped only for changes that older builds could not read correctly.
PROJECT_FORMAT_VERSION = 1


class ProjectError(Exception):
    """Raised when a project file cannot be read or is not a project at all."""


@dataclass
class FrameRecord:
    """One photograph plus the per-frame decisions made about it."""
    path: str                 # absolute path as it was when saved
    relpath: str = ""         # path relative to the project file, if possible
    filename: str = ""
    exposure_time: float = 0.0
    shutter_str: str = ""
    calculated_ev: Optional[float] = None
    is_valid: bool = True
    shift_x: float = 0.0
    shift_y: float = 0.0
    width: int = 0
    height: int = 0
    has_exif_time: bool = False

    @classmethod
    def from_item(cls, item: ExposureItem, project_dir: Optional[str]) -> "FrameRecord":
        return cls(
            path=os.path.abspath(item.filepath),
            relpath=_safe_relpath(item.filepath, project_dir),
            filename=item.filename,
            exposure_time=float(item.exposure_time),
            shutter_str=item.shutter_str,
            calculated_ev=item.calculated_ev,
            is_valid=bool(item.is_valid),
            shift_x=float(item.shift_x),
            shift_y=float(item.shift_y),
            width=int(item.width),
            height=int(item.height),
            has_exif_time=bool(item.has_exif_time),
        )

    def resolve(self, project_dir: Optional[str]) -> Optional[str]:
        """
        Finds the photograph on disk.

        The relative path is tried first, so moving or copying the whole folder
        (project file and photographs together) keeps a project working — which
        is the common case when files come off a card onto a different machine.
        """
        if self.relpath and project_dir:
            candidate = os.path.normpath(os.path.join(project_dir, self.relpath))
            if os.path.isfile(candidate):
                return candidate
        if self.path and os.path.isfile(self.path):
            return self.path
        return None


@dataclass
class Project:
    """Everything needed to restore a session."""
    frames: List[FrameRecord] = field(default_factory=list)
    settings: Dict[str, Any] = field(default_factory=dict)
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    roi_active: bool = False
    roi_rect: Optional[Tuple[int, int, int, int]] = None
    roi_size: int = 300
    ev_step: float = 1.0
    preset_name: str = ""
    compare_mode: bool = False
    histogram_visible: bool = False
    format_version: int = PROJECT_FORMAT_VERSION
    app_note: str = "Astro HDR Stacker project"

    def to_json(self) -> str:
        payload = asdict(self)
        payload["frames"] = [asdict(f) for f in self.frames]
        return json.dumps(payload, indent=2, ensure_ascii=False)


def _safe_relpath(path: str, base_dir: Optional[str]) -> str:
    """Relative path, or "" when the two are not comparable (different drives)."""
    if not base_dir:
        return ""
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(base_dir))
    except (ValueError, OSError):
        return ""


def _as_rect(value: Any) -> Optional[Tuple[int, int, int, int]]:
    """Coerces a stored rectangle back into a 4-tuple of ints, or None."""
    if not value or not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        rect = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None
    return rect if rect[2] > 0 and rect[3] > 0 else None


def build_project(
    items: List[ExposureItem],
    settings: Dict[str, Any],
    project_path: Optional[str] = None,
    **extras: Any,
) -> Project:
    """Snapshots the current session into a Project."""
    project_dir = os.path.dirname(os.path.abspath(project_path)) if project_path else None
    known = {f.name for f in Project.__dataclass_fields__.values()}
    return Project(
        frames=[FrameRecord.from_item(it, project_dir) for it in items],
        settings=dict(settings),
        **{k: v for k, v in extras.items() if k in known},
    )


def save_project(project: Project, filepath: str) -> None:
    """
    Writes the project as UTF-8 JSON.

    The write goes to a temporary file that is then moved into place, so an
    interrupted save can never leave a half-written project where a good one
    used to be.
    """
    if not filepath:
        raise ProjectError("Nebyla zadána cesta k projektu.")
    if not os.path.splitext(filepath)[1]:
        filepath += PROJECT_EXTENSION

    directory = os.path.dirname(os.path.abspath(filepath))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    # Recompute the relative paths against wherever the project is being written.
    project_dir = os.path.dirname(os.path.abspath(filepath))
    for frame in project.frames:
        frame.relpath = _safe_relpath(frame.path, project_dir)

    temp_path = filepath + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(project.to_json())
        os.replace(temp_path, filepath)
    except OSError as e:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise ProjectError(f"Projekt se nepodařilo uložit: {e}") from e


def load_project(filepath: str) -> Tuple[Project, List[str]]:
    """
    Reads a project and resolves its photographs.

    Returns (project, missing_paths). Frames whose file cannot be found are
    reported but do NOT stop the load: the rest of the session is still worth
    restoring, and the user can relink or drop the missing ones.
    """
    if not filepath or not os.path.isfile(filepath):
        raise ProjectError(f"Soubor projektu neexistuje: {filepath}")

    try:
        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, UnicodeDecodeError) as e:
        raise ProjectError(f"Projekt se nepodařilo přečíst: {e}") from e
    except json.JSONDecodeError as e:
        raise ProjectError(
            f"Soubor není platný projekt Astro HDR Stacker (chyba na řádku {e.lineno})."
        ) from e

    if not isinstance(data, dict) or "frames" not in data:
        raise ProjectError("Soubor není projekt Astro HDR Stacker.")

    version = int(data.get("format_version", 1) or 1)
    if version > PROJECT_FORMAT_VERSION:
        raise ProjectError(
            f"Projekt byl uložen novější verzí programu (formát {version}, "
            f"tento program umí {PROJECT_FORMAT_VERSION}). Aktualizujte prosím program."
        )

    project_dir = os.path.dirname(os.path.abspath(filepath))
    allowed = set(FrameRecord.__dataclass_fields__.keys())

    frames: List[FrameRecord] = []
    missing: List[str] = []
    for raw in data.get("frames", []):
        if not isinstance(raw, dict):
            continue
        # Unknown keys from a future build are dropped rather than raising.
        frame = FrameRecord(**{k: v for k, v in raw.items() if k in allowed})
        if frame.resolve(project_dir) is None:
            missing.append(frame.path or frame.relpath or frame.filename)
        frames.append(frame)

    settings = dict(data.get("settings") or {})
    # JSON has no tuples: a rectangle written as (x, y, w, h) reads back as a
    # list. Coerce it so callers can compare and unpack it the same way they
    # would a freshly built settings dict.
    if "crop_rect" in settings:
        settings["crop_rect"] = _as_rect(settings["crop_rect"])

    project = Project(
        frames=frames,
        settings=settings,
        crop_rect=_as_rect(data.get("crop_rect")),
        roi_active=bool(data.get("roi_active", False)),
        roi_rect=_as_rect(data.get("roi_rect")),
        roi_size=int(data.get("roi_size", 300) or 300),
        ev_step=float(data.get("ev_step", 1.0) or 1.0),
        preset_name=str(data.get("preset_name", "") or ""),
        compare_mode=bool(data.get("compare_mode", False)),
        histogram_visible=bool(data.get("histogram_visible", False)),
        format_version=version,
    )
    return project, missing


def resolved_paths(project: Project, project_path: Optional[str]) -> List[str]:
    """Existing photograph paths, in the order the project stored them."""
    project_dir = os.path.dirname(os.path.abspath(project_path)) if project_path else None
    found = []
    for frame in project.frames:
        path = frame.resolve(project_dir)
        if path is not None:
            found.append(path)
    return found


def apply_frame_records(project: Project, items: List[ExposureItem],
                        project_path: Optional[str]) -> int:
    """
    Restores per-frame state (shifts, include/exclude) onto freshly loaded items.

    Matching is by resolved path, so it stays correct even when re-inspection
    reorders the list — which it does, since frames are sorted by exposure.
    Returns how many frames were matched.
    """
    project_dir = os.path.dirname(os.path.abspath(project_path)) if project_path else None

    by_path: Dict[str, FrameRecord] = {}
    for frame in project.frames:
        path = frame.resolve(project_dir)
        if path is not None:
            by_path[os.path.normcase(os.path.abspath(path))] = frame

    matched = 0
    for item in items:
        record = by_path.get(os.path.normcase(os.path.abspath(item.filepath)))
        if record is None:
            continue
        item.shift_x = float(record.shift_x)
        item.shift_y = float(record.shift_y)
        item.is_valid = bool(record.is_valid)
        matched += 1
    return matched
