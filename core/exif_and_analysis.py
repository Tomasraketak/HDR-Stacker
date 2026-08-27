"""
EXIF extraction, brightness analysis, and automatic EV-sequence recognition.

Handles both EXIF-based discovery and luminance-based sorting, and — importantly —
preserves per-frame user state (manual shifts, include/exclude) across re-scans.
"""

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import cv2
import numpy as np
from PIL import Image, ExifTags

try:
    from core.postprocess import imread_unicode
except ImportError:  # pragma: no cover - package-relative fallback
    from .postprocess import imread_unicode


@dataclass
class ExposureItem:
    filepath: str
    filename: str
    exposure_time: float          # seconds, e.g. 0.001 for 1/1000 s
    shutter_str: str              # e.g. "1/1000s" or "2.5s"
    iso: Optional[int] = None
    aperture: Optional[float] = None
    ev_bias: Optional[float] = None
    calculated_ev: Optional[float] = None
    mean_luminance: float = 0.0
    is_valid: bool = True
    thumbnail: Optional[np.ndarray] = None
    shift_x: float = 0.0
    shift_y: float = 0.0
    width: int = 0
    height: int = 0
    has_exif_time: bool = False


def format_shutter_speed(sec: float) -> str:
    """Formats an exposure time in seconds as a human-readable shutter speed."""
    if sec is None or not math.isfinite(sec) or sec <= 0:
        return "N/A"
    if sec < 0.8:
        denom = max(1, round(1.0 / sec))
        return f"1/{denom}s"
    if sec < 10:
        return f"{int(sec)}s" if abs(sec - round(sec)) < 1e-6 else f"{sec:.2f}s"
    return f"{int(sec)}s" if abs(sec - round(sec)) < 1e-6 else f"{sec:.1f}s"


def read_image_size(filepath: str) -> Tuple[int, int]:
    """
    Reads pixel dimensions from the file header only — no full decode.

    Used to estimate memory before loading a 45 MP bracket, which is what keeps
    the full-resolution export from exhausting RAM on a laptop.
    """
    try:
        with Image.open(filepath) as im:
            return int(im.width), int(im.height)
    except Exception:
        return 0, 0


def compute_image_luminance(filepath: str, max_dim: int = 400) -> Tuple[float, Optional[np.ndarray]]:
    """Loads a downscaled copy to compute perceptual brightness plus a GUI thumbnail."""
    try:
        img = imread_unicode(filepath, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, None

        h, w = img.shape[:2]
        scale = min(max_dim / float(max(h, w)), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        median_val = float(np.median(gray))
        p75 = float(np.percentile(gray, 75))

        luminance_score = 0.4 * mean_val + 0.3 * median_val + 0.3 * p75
        thumb_rgb = np.ascontiguousarray(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
        return luminance_score, thumb_rgb
    except Exception as e:
        print(f"Error computing luminance for {filepath}: {e}")
        return 0.0, None


def _ratio_to_float(val) -> Optional[float]:
    """Converts the several shapes an EXIF rational can arrive in into a float."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    num = getattr(val, 'numerator', None)
    den = getattr(val, 'denominator', None)
    if num is not None and den:
        return float(num) / float(den)
    if isinstance(val, tuple) and len(val) == 2 and val[1]:
        return float(val[0]) / float(val[1])
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def extract_exif_metadata(filepath: str) -> dict:
    """Extracts exposure time, ISO, aperture and EV bias via Pillow."""
    info = {
        'exposure_time': None,
        'iso': None,
        'aperture': None,
        'ev_bias': None,
        'has_exif': False,
    }
    try:
        with Image.open(filepath) as pil_img:
            exif = pil_img.getexif()
            if not exif:
                return info

            info['has_exif'] = True
            exif_dict = {}
            for tag_id, value in exif.items():
                exif_dict[ExifTags.TAGS.get(tag_id, tag_id)] = value

            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.Makernote):
                try:
                    for k, v in exif.get_ifd(ifd_id).items():
                        exif_dict[ExifTags.TAGS.get(k, k)] = v
                except Exception:
                    pass

            t = _ratio_to_float(exif_dict.get('ExposureTime'))
            if t is None and 'ShutterSpeedValue' in exif_dict:
                # APEX: time = 2^-ShutterSpeedValue
                apex = _ratio_to_float(exif_dict['ShutterSpeedValue'])
                if apex is not None and -30 < apex < 30:
                    t = 2.0 ** (-apex)
            if t is not None and math.isfinite(t) and t > 0:
                info['exposure_time'] = float(t)

            iso_val = exif_dict.get('ISOSpeedRatings', exif_dict.get('PhotographicSensitivity'))
            if isinstance(iso_val, (tuple, list)) and iso_val:
                iso_val = iso_val[0]
            if isinstance(iso_val, (int, float)):
                info['iso'] = int(iso_val)

            info['aperture'] = _ratio_to_float(exif_dict.get('FNumber'))
            info['ev_bias'] = _ratio_to_float(exif_dict.get('ExposureBiasValue'))

    except Exception as e:
        print(f"EXIF parsing error for {filepath}: {e}")

    return info


def inspect_exposure_files(
    filepaths: List[str],
    user_ev_step: float = 1.0,
    preserve: Optional[Dict[str, ExposureItem]] = None,
) -> List[ExposureItem]:
    """
    Inspects image files and builds a sorted, EV-mapped exposure sequence.

    1. Reads EXIF, or measures scene luminance when EXIF is missing.
    2. Orders frames from the shortest (darkest, -EV) to the longest (+EV).
    3. Assigns EV values and synthesises plausible shutter speeds where needed.

    `preserve` maps filepath -> a previously built item whose user-set state
    (manual shifts, include/exclude flag) is carried over. Without it, re-sorting
    the list would silently discard manual alignment work.
    """
    items: List[ExposureItem] = []
    preserve = preserve or {}

    for path in filepaths:
        exif_info = extract_exif_metadata(path)
        luminance, thumb = compute_image_luminance(path)
        exp_time = exif_info['exposure_time']
        w, h = read_image_size(path)

        item = ExposureItem(
            filepath=path,
            filename=os.path.basename(path),
            exposure_time=exp_time if exp_time else 0.0,
            shutter_str=format_shutter_speed(exp_time) if exp_time else "Auto",
            iso=exif_info['iso'],
            aperture=exif_info['aperture'],
            ev_bias=exif_info['ev_bias'],
            mean_luminance=luminance,
            thumbnail=thumb,
            width=w,
            height=h,
            has_exif_time=bool(exp_time),
        )

        prev = preserve.get(path)
        if prev is not None:
            item.shift_x = prev.shift_x
            item.shift_y = prev.shift_y
            item.is_valid = prev.is_valid

        items.append(item)

    if not items:
        return items

    exif_count = sum(1 for it in items if it.has_exif_time)

    if exif_count == len(items) and len(items) > 1:
        # Every frame has a real shutter speed: trust it completely.
        items.sort(key=lambda x: x.exposure_time)
        ref_time = items[len(items) // 2].exposure_time
        for it in items:
            it.calculated_ev = round(math.log2(it.exposure_time / ref_time), 2) if ref_time > 0 else 0.0
    else:
        # Missing or partial EXIF: order by measured scene brightness and
        # synthesise an even EV ladder around the middle frame.
        items.sort(key=lambda x: x.mean_luminance)
        half = (len(items) - 1) / 2.0
        known = [it for it in items if it.has_exif_time]
        base_center = (
            float(np.median([it.exposure_time for it in known])) if known else 1.0 / 125.0
        )

        for idx, it in enumerate(items):
            rel_ev = (idx - half) * float(user_ev_step)
            it.calculated_ev = round(rel_ev, 2)
            if not it.has_exif_time:
                synthesized = base_center * (2.0 ** rel_ev)
                it.exposure_time = synthesized
                it.shutter_str = format_shutter_speed(synthesized)

    return items


def estimate_stack_megapixels(items: List[ExposureItem]) -> float:
    """Megapixels of the largest frame in the stack (0 if unknown)."""
    best = 0.0
    for it in items:
        if it.width > 0 and it.height > 0:
            best = max(best, (it.width * it.height) / 1e6)
    return best
