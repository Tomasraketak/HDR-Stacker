"""
EXIF extraction, brightness analysis, and automatic EV sequence recognition.
Handles both EXIF-based discovery and smart luminance-based sorting and EV assignment.
"""

import os
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ExifTags


@dataclass
class ExposureItem:
    filepath: str
    filename: str
    exposure_time: float  # in seconds, e.g. 0.001 (1/1000s)
    shutter_str: str      # e.g. "1/1000s" or "2.5s"
    iso: Optional[int] = None
    aperture: Optional[float] = None
    ev_bias: Optional[float] = None
    calculated_ev: Optional[float] = None
    mean_luminance: float = 0.0
    is_valid: bool = True
    thumbnail: Optional[np.ndarray] = None
    shift_x: float = 0.0
    shift_y: float = 0.0


def format_shutter_speed(sec: float) -> str:
    """Format exposure time in seconds to human readable string (e.g. 1/1000s, 0.5s, 2s)."""
    if sec <= 0:
        return "N/A"
    if sec < 0.8:
        denom = round(1.0 / sec)
        return f"1/{denom}s"
    elif sec < 10:
        return f"{sec:.2f}s" if sec != round(sec) else f"{int(sec)}s"
    else:
        return f"{sec:.1f}s" if sec != round(sec) else f"{int(sec)}s"


def compute_image_luminance(filepath: str, max_dim: int = 400) -> Tuple[float, Optional[np.ndarray]]:
    """
    Quickly loads downscaled image to calculate average perceptual luminance
    and thumbnail for the GUI.
    """
    try:
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, None
        
        h, w = img.shape[:2]
        scale = min(max_dim / max(h, w), 1.0)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
        
        mean_val = float(np.mean(gray))
        median_val = float(np.median(gray))
        p75 = float(np.percentile(gray, 75))
        
        luminance_score = 0.4 * mean_val + 0.3 * median_val + 0.3 * p75
        
        thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        return luminance_score, thumb_rgb
    except Exception as e:
        print(f"Error computing luminance for {filepath}: {e}")
        return 0.0, None


def extract_exif_metadata(filepath: str) -> dict:
    """Extracts exposure time, ISO, aperture, and EV bias from EXIF using Pillow."""
    info = {
        'exposure_time': None,
        'iso': None,
        'aperture': None,
        'ev_bias': None,
        'has_exif': False
    }
    try:
        with Image.open(filepath) as pil_img:
            exif = pil_img.getexif()
            if not exif:
                return info

            info['has_exif'] = True
            
            exif_dict = {}
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                exif_dict[tag_name] = value

            for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.Makernote):
                try:
                    ifd = exif.get_ifd(ifd_id)
                    for k, v in ifd.items():
                        tag_name = ExifTags.TAGS.get(k, k)
                        exif_dict[tag_name] = v
                except Exception:
                    pass

            if 'ExposureTime' in exif_dict:
                val = exif_dict['ExposureTime']
                if isinstance(val, (int, float)):
                    info['exposure_time'] = float(val)
                elif hasattr(val, 'numerator') and hasattr(val, 'denominator'):
                    info['exposure_time'] = float(val.numerator) / float(val.denominator) if val.denominator != 0 else None
                elif isinstance(val, tuple) and len(val) == 2 and val[1] != 0:
                    info['exposure_time'] = float(val[0]) / float(val[1])

            if 'ISOSpeedRatings' in exif_dict:
                iso_val = exif_dict['ISOSpeedRatings']
                if isinstance(iso_val, tuple):
                    info['iso'] = int(iso_val[0])
                elif isinstance(iso_val, (int, float)):
                    info['iso'] = int(iso_val)
            elif 'PhotographicSensitivity' in exif_dict:
                info['iso'] = int(exif_dict['PhotographicSensitivity'])

            if 'FNumber' in exif_dict:
                fn = exif_dict['FNumber']
                if isinstance(fn, (int, float)):
                    info['aperture'] = float(fn)
                elif hasattr(fn, 'numerator') and hasattr(fn, 'denominator'):
                    info['aperture'] = float(fn.numerator) / float(fn.denominator) if fn.denominator != 0 else None

            if 'ExposureBiasValue' in exif_dict:
                eb = exif_dict['ExposureBiasValue']
                if isinstance(eb, (int, float)):
                    info['ev_bias'] = float(eb)
                elif hasattr(eb, 'numerator') and hasattr(eb, 'denominator'):
                    info['ev_bias'] = float(eb.numerator) / float(eb.denominator) if eb.denominator != 0 else 0.0

    except Exception as e:
        print(f"EXIF parsing error for {filepath}: {e}")

    return info


def inspect_exposure_files(
    filepaths: List[str],
    user_ev_step: float = 1.0,
    base_shutter_center: float = 1.0 / 125.0
) -> List[ExposureItem]:
    """
    Inspects multiple image files:
    1. Extracts EXIF data or calculates scene luminance.
    2. Determines which photo corresponds to which EV/shutter speed.
    3. Orders photos systematically from fastest (darkest / -EV) to longest (brightest / +EV).
    4. Automatically maps EV values and shutter speeds.
    """
    items: List[ExposureItem] = []
    
    for path in filepaths:
        fname = os.path.basename(path)
        exif_info = extract_exif_metadata(path)
        luminance, thumb = compute_image_luminance(path)
        
        exp_time = exif_info['exposure_time']
        
        item = ExposureItem(
            filepath=path,
            filename=fname,
            exposure_time=exp_time if exp_time is not None else 0.0,
            shutter_str=format_shutter_speed(exp_time) if exp_time is not None else "Auto",
            iso=exif_info['iso'],
            aperture=exif_info['aperture'],
            ev_bias=exif_info['ev_bias'],
            mean_luminance=luminance,
            thumbnail=thumb
        )
        items.append(item)

    exif_count = sum(1 for it in items if it.exposure_time > 0)
    
    if exif_count == len(items) and len(items) > 1:
        # All have valid EXIF exposure times: sort by exposure time ascending
        items.sort(key=lambda x: x.exposure_time)
        
        # Calculate relative EV relative to the median exposure
        ref_time = items[len(items) // 2].exposure_time
        for it in items:
            if it.exposure_time > 0 and ref_time > 0:
                rel_ev = math.log2(it.exposure_time / ref_time)
                it.calculated_ev = round(rel_ev, 2)
            else:
                it.calculated_ev = 0.0
    else:
        # Sort by image luminance from darkest to brightest
        items.sort(key=lambda x: x.mean_luminance)
        
        n = len(items)
        half = (n - 1) / 2.0
        
        for idx, it in enumerate(items):
            rel_ev_step = (idx - half) * user_ev_step
            it.calculated_ev = round(rel_ev_step, 2)
            
            if it.exposure_time <= 0:
                calculated_shutter = base_shutter_center * (2.0 ** rel_ev_step)
                it.exposure_time = calculated_shutter
                it.shutter_str = format_shutter_speed(calculated_shutter)

    return items
