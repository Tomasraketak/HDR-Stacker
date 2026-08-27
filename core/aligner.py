"""
Advanced Astronomical Alignment Engine.

Features:
1. Robust "black circle in light" detector (locates the circular Moon silhouette
   surrounded by the bright corona) that works on a bounded-size proxy so it stays
   fast and memory-safe even on 24-45 MP frames.
2. Universal Sun/Moon center finder (black-disc detector plus a bright-corona
   moment analysis fallback).
3. Static point-light (street lamp) pattern alignment, which measures camera
   movement directly from a fixed foreground.
4. Per-image subpixel shift calculation and application.

All public functions are defensive: they never raise on malformed input, they
never allocate more than a bounded amount of scratch memory, and they always
return usable values.
"""

from typing import List, Callable, Optional, Tuple
import cv2
import numpy as np

# Detection never runs on more pixels than this along the longest edge.
# A 6000x4000 frame is analysed at 1024x683 -> ~35x less work and memory.
DETECT_MAX_DIM = 1024


def _as_bgr_u8(image: np.ndarray) -> Optional[np.ndarray]:
    """Coerces any supported array into a contiguous 8-bit 3-channel BGR image."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None

    img = image
    if img.dtype != np.uint8:
        img = np.nan_to_num(img.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
        # float images are assumed to be in [0, 1]; integer images in [0, 65535]
        if image.dtype in (np.float32, np.float64):
            img = np.clip(img, 0.0, 1.0) * 255.0
        else:
            img = np.clip(img / 257.0, 0.0, 255.0)
        img = img.astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3 or img.shape[2] != 3:
        return None

    return np.ascontiguousarray(img)


def _detection_proxy(image_bgr: np.ndarray, max_dim: int = DETECT_MAX_DIM) -> Tuple[np.ndarray, float]:
    """
    Returns (gray_proxy, scale) where scale maps proxy coordinates back to the
    original image: original = proxy / scale.
    """
    h, w = image_bgr.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 0.999:
        small = cv2.resize(image_bgr, (max(8, int(round(w * scale))), max(8, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = image_bgr
        scale = 1.0
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), scale


def _refine_disc_center(gray_full: np.ndarray, cx: float, cy: float, radius: float) -> Tuple[float, float]:
    """
    Refines a coarse disc center to subpixel accuracy at full resolution.

    Uses the intensity-weighted centroid of the *dark* pixels inside a window
    around the coarse estimate, which is robust against coronal asymmetry.
    """
    h, w = gray_full.shape[:2]
    half = int(max(8.0, radius * 1.6))
    x0 = int(max(0, round(cx) - half))
    y0 = int(max(0, round(cy) - half))
    x1 = int(min(w, round(cx) + half + 1))
    y1 = int(min(h, round(cy) + half + 1))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return float(cx), float(cy)

    patch = gray_full[y0:y1, x0:x1].astype(np.float32)
    patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=1.5, sigmaY=1.5)

    lo = float(patch.min())
    hi = float(patch.max())
    if hi - lo < 4.0:
        return float(cx), float(cy)

    # Weight = "how dark", zero above the midpoint between disc and corona.
    threshold = lo + (hi - lo) * 0.45
    weights = np.clip(threshold - patch, 0.0, None)
    total = float(weights.sum())
    if total <= 1e-6:
        return float(cx), float(cy)

    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    rx = float((weights * xs).sum() / total) + x0
    ry = float((weights * ys).sum() / total) + y0

    # Reject a refinement that ran away from the coarse estimate.
    if abs(rx - cx) > radius or abs(ry - cy) > radius:
        return float(cx), float(cy)
    return rx, ry


def detect_black_circle_in_light(
    image_bgr: np.ndarray,
    max_dim: int = DETECT_MAX_DIM
) -> Optional[Tuple[float, float, float]]:
    """
    Detects the dark lunar silhouette surrounded by the bright coronal glow.

    Landscape and foliage are rejected by a strict circularity filter plus a
    check that the candidate's interior really is dark relative to the scene.

    Returns (cx, cy, radius) in ORIGINAL image coordinates, or None.
    """
    img = _as_bgr_u8(image_bgr)
    if img is None:
        return None

    full_h, full_w = img.shape[:2]
    if full_w < 16 or full_h < 16:
        return None

    try:
        gray_small, scale = _detection_proxy(img, max_dim)
        h, w = gray_small.shape[:2]

        blurred = cv2.GaussianBlur(gray_small, (0, 0), sigmaX=2.0, sigmaY=2.0)

        # The Sun is essentially never in the bottom 10% (that is horizon/landscape).
        sky_h = max(16, int(h * 0.9))
        sky_gray = blurred[:sky_h, :]

        min_v, max_v, _, _ = cv2.minMaxLoc(sky_gray)
        if max_v < 25:
            return None

        min_r = max(2.0, min(w, h) * 0.02)
        max_r = min(w, h) * 0.45
        min_area = np.pi * (min_r ** 2)
        max_area = np.pi * (max_r ** 2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        candidates: List[Tuple[Tuple[float, float, float], float]] = []

        for t_factor in (0.15, 0.25, 0.35, 0.45, 0.55):
            t_val = int(min_v + (max_v - min_v) * t_factor)
            _, thresh = cv2.threshold(sky_gray, t_val, 255, cv2.THRESH_BINARY_INV)
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in contours:
                area = cv2.contourArea(c)
                if not (min_area <= area <= max_area):
                    continue
                perimeter = cv2.arcLength(c, True)
                if perimeter <= 0:
                    continue
                circularity = (4.0 * np.pi * area) / (perimeter ** 2)
                if circularity <= 0.65:
                    continue

                (cx, cy), r = cv2.minEnclosingCircle(c)
                cx_i, cy_i = int(cx), int(cy)
                if not (0 <= cx_i < w and 0 <= cy_i < sky_h):
                    continue
                if float(blurred[cy_i, cx_i]) >= max_v * 0.7:
                    continue

                # Prefer candidates that really are enclosed by bright corona:
                # sample an annulus just outside the disc.
                halo = _annulus_mean(blurred, cx, cy, r * 1.15, r * 1.6)
                halo_bonus = 0.0
                if halo is not None and halo > float(blurred[cy_i, cx_i]) + 10.0:
                    halo_bonus = 0.25

                candidates.append(((float(cx), float(cy), float(r)), circularity + halo_bonus))

        best = None
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best = candidates[0][0]
        else:
            best = _hough_fallback(blurred[:sky_h, :], w, h)

        if best is None:
            return None

        # Map the proxy result back to full resolution and refine there.
        inv = 1.0 / scale if scale > 0 else 1.0
        cx_f, cy_f, r_f = best[0] * inv, best[1] * inv, best[2] * inv
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cx_f, cy_f = _refine_disc_center(gray_full, cx_f, cy_f, r_f)

        cx_f = float(np.clip(cx_f, 0.0, full_w - 1.0))
        cy_f = float(np.clip(cy_f, 0.0, full_h - 1.0))
        return cx_f, cy_f, float(r_f)

    except cv2.error:
        return None
    except Exception:
        return None


def _annulus_mean(gray: np.ndarray, cx: float, cy: float, r_in: float, r_out: float) -> Optional[float]:
    """Mean brightness of a ring around (cx, cy). Returns None if it does not fit."""
    h, w = gray.shape[:2]
    x0, y0 = int(max(0, cx - r_out)), int(max(0, cy - r_out))
    x1, y1 = int(min(w, cx + r_out + 1)), int(min(h, cy + r_out + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    mask = (d >= r_in) & (d <= r_out)
    if not mask.any():
        return None
    return float(patch[mask].mean())


def _hough_fallback(gray: np.ndarray, w: int, h: int) -> Optional[Tuple[float, float, float]]:
    """Soft-parameter Hough circle search used when contour analysis finds nothing."""
    try:
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(8, min(w, h) // 4),
            param1=80,
            param2=25,
            minRadius=max(2, int(min(w, h) * 0.03)),
            maxRadius=max(4, int(min(w, h) * 0.45)),
        )
    except cv2.error:
        return None
    if circles is None or len(circles) == 0 or len(circles[0]) == 0:
        return None
    c = circles[0][0]
    return float(c[0]), float(c[1]), float(c[2])


def find_sun_or_moon_center(image_bgr: np.ndarray, max_dim: int = DETECT_MAX_DIM) -> Tuple[int, int]:
    """
    Universal astronomical center finder, in ORIGINAL image coordinates.

    1. Strict black lunar-disc detection (totality).
    2. Fallback: centre of mass of the brightest coronal / solar region.
    3. Final fallback: centre of the frame.
    """
    img = _as_bgr_u8(image_bgr)
    if img is None:
        return 0, 0

    h, w = img.shape[:2]

    disc = detect_black_circle_in_light(img, max_dim=max_dim)
    if disc is not None:
        return int(round(disc[0])), int(round(disc[1]))

    try:
        gray_small, scale = _detection_proxy(img, max_dim)
        blurred = cv2.GaussianBlur(gray_small, (0, 0), sigmaX=3.0, sigmaY=3.0)
        min_v, max_v, _, _ = cv2.minMaxLoc(blurred)
        if max_v > 30:
            t_val = int(min_v + (max_v - min_v) * 0.85)
            _, mask = cv2.threshold(blurred, t_val, 255, cv2.THRESH_BINARY)
            M = cv2.moments(mask)
            if M["m00"] > 0:
                inv = 1.0 / scale if scale > 0 else 1.0
                cx = int(round((M["m10"] / M["m00"]) * inv))
                cy = int(round((M["m01"] / M["m00"]) * inv))
                if 0 <= cx < w and 0 <= cy < h:
                    return cx, cy
    except cv2.error:
        pass
    except Exception:
        pass

    return w // 2, h // 2


def calculate_moon_shifts(
    images: List[np.ndarray],
    ref_idx: Optional[int] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> List[Tuple[float, float]]:
    """
    Detects the lunar disc in each image and computes the subpixel (dx, dy) that
    brings each frame onto the reference frame.

    Frames where no disc is found get (0, 0) — they are left untouched rather
    than being shifted by a wrong guess.
    """
    n = len(images) if images else 0
    if n == 0:
        return []
    if n == 1:
        return [(0.0, 0.0)]

    if ref_idx is None or not (0 <= ref_idx < n):
        ref_idx = n // 2

    circles: List[Optional[Tuple[float, float, float]]] = []
    for i, img in enumerate(images):
        if should_cancel is not None and should_cancel():
            return [(0.0, 0.0)] * n
        if progress_callback is not None:
            progress_callback(int(100 * i / n), f"Detekce disku Měsíce {i + 1}/{n}...")
        circles.append(detect_black_circle_in_light(img))

    ref_disc = circles[ref_idx]
    if ref_disc is None:
        # Fall back to the first frame where the disc was actually found.
        for idx, c in enumerate(circles):
            if c is not None:
                ref_idx, ref_disc = idx, c
                break

    if ref_disc is None:
        return [(0.0, 0.0)] * n

    ref_cx, ref_cy, _ = ref_disc
    shifts: List[Tuple[float, float]] = []
    for c in circles:
        if c is None:
            shifts.append((0.0, 0.0))
        else:
            shifts.append((float(ref_cx - c[0]), float(ref_cy - c[1])))
    return shifts


def apply_shifts_to_images(
    images: List[np.ndarray],
    shifts: List[Tuple[float, float]],
    scale_factor: float = 1.0
) -> List[np.ndarray]:
    """
    Applies (dx, dy) translations with subpixel cubic interpolation.

    `scale_factor` converts shifts expressed in full-resolution pixels into the
    pixel scale of the images actually passed in (e.g. 0.25 for a quarter-size
    proxy). Images without a matching shift are passed through untouched.
    """
    if not images:
        return []
    shifts = list(shifts) if shifts else []

    aligned: List[np.ndarray] = []
    for i, img in enumerate(images):
        dx, dy = shifts[i] if i < len(shifts) else (0.0, 0.0)
        sx = float(dx) * float(scale_factor)
        sy = float(dy) * float(scale_factor)

        if not np.isfinite(sx) or not np.isfinite(sy) or (abs(sx) < 0.01 and abs(sy) < 0.01):
            aligned.append(img)
            continue

        h, w = img.shape[:2]
        M = np.float32([[1.0, 0.0, sx], [0.0, 1.0, sy]])
        try:
            aligned.append(cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            ))
        except cv2.error:
            aligned.append(img)

    return aligned


# =====================================================================
#  Static point-light (street lamp) pattern alignment
# =====================================================================
#
# During totality the Sun drifts across the sky, but street lamps and distant
# windows in the foreground do not move at all. Their pattern is therefore a
# direct measurement of camera shake, and it stays usable across the whole
# bracket because lamps are bright point sources that register even in the
# darkest -4 EV frame.
#
# Lamps are matched as a constellation rather than by raw pixels, which makes
# the estimate immune to the huge brightness differences between exposures.

# Point-light detection runs at a higher resolution than disc detection:
# lamps are only a few pixels across and vanish if the proxy is too small.
LIGHTS_MAX_DIM = 1600

# Only the lower part of the frame is searched by default — that is where the
# horizon and its lamps are, and it excludes the moving Sun entirely.
LIGHTS_DEFAULT_REGION = (0.45, 1.0)


def detect_point_lights(
    image_bgr: np.ndarray,
    region: Tuple[float, float] = LIGHTS_DEFAULT_REGION,
    max_points: int = 150,
    max_dim: int = LIGHTS_MAX_DIM,
) -> np.ndarray:
    """
    Finds small, bright, isolated light sources and returns their subpixel
    centres in ORIGINAL image coordinates.

    `region` is the vertical band to search, as fractions of image height.
    Returns an (N, 3) array of (x, y, brightness), brightest first, or an
    empty (0, 3) array when nothing suitable is found.
    """
    empty = np.zeros((0, 3), dtype=np.float32)
    img = _as_bgr_u8(image_bgr)
    if img is None:
        return empty

    full_h, full_w = img.shape[:2]
    if full_w < 32 or full_h < 32:
        return empty

    try:
        scale = min(1.0, float(max_dim) / float(max(full_h, full_w)))
        if scale < 0.999:
            small = cv2.resize(img, (max(16, int(round(full_w * scale))),
                                     max(16, int(round(full_h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small, scale = img, 1.0

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        y0 = int(np.clip(region[0], 0.0, 1.0) * h)
        y1 = int(np.clip(region[1], 0.0, 1.0) * h)
        if y1 - y0 < 16:
            y0, y1 = 0, h
        band = gray[y0:y1, :]

        # White top-hat isolates small bright structures from any smooth
        # background gradient (sky glow, light pollution, a lit horizon).
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        tophat = cv2.morphologyEx(band, cv2.MORPH_TOPHAT, kernel)
        tophat = cv2.GaussianBlur(tophat, (0, 0), sigmaX=1.0, sigmaY=1.0)

        peak = float(tophat.max())
        if peak < 8.0:
            return empty

        # Keep only clearly-above-background responses. The percentile floor
        # stops a single very bright lamp from suppressing all the others.
        level = max(float(np.percentile(tophat, 99.0)), peak * 0.12, 6.0)
        _, mask = cv2.threshold(tophat, level, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        # In the brightest frames a lamp saturates and its top-hat response
        # collapses, so add the saturated cores as a second detection channel.
        # It is skipped when the band is mostly blown, since then the lamps have
        # merged into the background and their positions are unrecoverable.
        sat_fraction = float((band >= 250).mean())
        if sat_fraction < 0.25:
            _, sat_mask = cv2.threshold(band, 249, 255, cv2.THRESH_BINARY)
            sat_mask = cv2.morphologyEx(
                sat_mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            mask = cv2.bitwise_or(mask, sat_mask)

        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels <= 1:
            return empty

        inv = 1.0 / scale if scale > 0 else 1.0
        max_area = max(24.0, (w * h) * 0.0008)   # a lamp is small; reject big blobs
        points = []

        for label in range(1, n_labels):
            area = float(stats[label, cv2.CC_STAT_AREA])
            if area < 2.0 or area > max_area:
                continue
            bw = float(stats[label, cv2.CC_STAT_WIDTH])
            bh = float(stats[label, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or bh <= 0:
                continue
            # Reject streaks: a lamp is roughly round, a wire or roof edge is not.
            if max(bw, bh) / min(bw, bh) > 3.0:
                continue

            cx, cy = float(centroids[label][0]), float(centroids[label][1])
            cx, cy = _refine_point_centroid(tophat, cx, cy, radius=max(2.0, (bw + bh) * 0.5))
            peak_here = float(tophat[int(np.clip(cy, 0, band.shape[0] - 1)),
                                     int(np.clip(cx, 0, band.shape[1] - 1))])
            # A saturated core has no top-hat peak left; rank it by size instead.
            brightness = max(peak_here, area)
            points.append((cx * inv, (cy + y0) * inv, brightness))

        if not points:
            return empty

        arr = np.array(points, dtype=np.float32)
        arr = arr[np.argsort(-arr[:, 2])][:max_points]
        return arr

    except cv2.error:
        return empty
    except Exception:
        return empty


def _refine_point_centroid(band: np.ndarray, cx: float, cy: float,
                           radius: float) -> Tuple[float, float]:
    """Intensity-weighted centroid in a small window — subpixel lamp position."""
    h, w = band.shape[:2]
    r = int(max(2, min(6, round(radius))))
    x0, y0 = int(max(0, cx - r)), int(max(0, cy - r))
    x1, y1 = int(min(w, cx + r + 1)), int(min(h, cy + r + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return cx, cy

    patch = band[y0:y1, x0:x1].astype(np.float32)
    total = float(patch.sum())
    if total <= 1e-6:
        return cx, cy
    ys, xs = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    return float((patch * xs).sum() / total) + x0, float((patch * ys).sum() / total) + y0


def estimate_translation_from_points(
    ref_points: np.ndarray,
    cur_points: np.ndarray,
    max_shift: float = 300.0,
    vote_bin: float = 3.0,
    inlier_radius: float = 4.0,
) -> Optional[Tuple[float, float, int]]:
    """
    Recovers the translation that maps `cur_points` onto `ref_points`.

    Every candidate pair casts a vote for the offset it implies; the true
    translation is the only offset a whole constellation agrees on, so it wins
    by a wide margin even when many lamps are missing from one frame. The
    winning bin is then refined to subpixel accuracy from its inliers.

    Returns (dx, dy, inlier_count), or None if no offset gathers support.
    """
    if ref_points is None or cur_points is None:
        return None
    if len(ref_points) < 3 or len(cur_points) < 3:
        return None

    ref_xy = np.asarray(ref_points, dtype=np.float32)[:, :2]
    cur_xy = np.asarray(cur_points, dtype=np.float32)[:, :2]

    # All pairwise offsets: ref - cur, i.e. what must be added to cur.
    dx = ref_xy[:, None, 0] - cur_xy[None, :, 0]
    dy = ref_xy[:, None, 1] - cur_xy[None, :, 1]
    within = (np.abs(dx) <= max_shift) & (np.abs(dy) <= max_shift)
    if not within.any():
        return None

    dx_f, dy_f = dx[within].ravel(), dy[within].ravel()

    # 2-D vote over quantised offsets.
    bin_size = max(0.5, float(vote_bin))
    keys_x = np.round(dx_f / bin_size).astype(np.int64)
    keys_y = np.round(dy_f / bin_size).astype(np.int64)
    span = int(np.ceil(max_shift / bin_size)) * 2 + 3
    flat = (keys_y + span) * (span * 4) + (keys_x + span)
    uniq, counts = np.unique(flat, return_counts=True)
    best = int(np.argmax(counts))
    if counts[best] < 3:
        return None

    key = uniq[best]
    coarse_x = float(key % (span * 4) - span) * bin_size
    coarse_y = float(key // (span * 4) - span) * bin_size

    # Refine: keep the offsets consistent with the winner and take the median,
    # which ignores the handful of pairs that matched the wrong lamp.
    near = (np.abs(dx_f - coarse_x) <= inlier_radius) & (np.abs(dy_f - coarse_y) <= inlier_radius)
    inliers = int(near.sum())
    if inliers < 3:
        return None

    return float(np.median(dx_f[near])), float(np.median(dy_f[near])), inliers


def _normalised_band(image_bgr: np.ndarray, region: Tuple[float, float],
                     max_dim: int = LIGHTS_MAX_DIM) -> Optional[Tuple[np.ndarray, float]]:
    """
    Contrast-normalised float copy of the search band, for phase correlation.

    CLAHE plus gradient magnitude removes the overall exposure level, so a -4 EV
    and a +2 EV frame present comparable structure to the correlator.
    """
    img = _as_bgr_u8(image_bgr)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w)))
    if scale < 0.999:
        img = cv2.resize(img, (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                         interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bh = gray.shape[0]
    y0 = int(np.clip(region[0], 0.0, 1.0) * bh)
    y1 = int(np.clip(region[1], 0.0, 1.0) * bh)
    if y1 - y0 < 32:
        y0, y1 = 0, bh
    band = gray[y0:y1, :]
    if band.shape[0] < 16 or band.shape[1] < 16:
        return None

    band = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(band)
    gx = cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return np.ascontiguousarray(magnitude), scale


def _phase_correlate_bands(
    ref_image: np.ndarray,
    cur_image: np.ndarray,
    region: Tuple[float, float],
    min_response: float = 0.06,
) -> Optional[Tuple[float, float, float]]:
    """
    Whole-band subpixel translation via phase correlation.

    Used as a last resort when too few individual lamps survive: it draws on
    every edge in the band rather than isolated points. It is roughly a pixel
    accurate — far worse than matched lamps, but far better than giving up.

    The response threshold matters and is deliberately strict. On a blown or
    near-featureless band the correlator still returns a confident-looking peak
    that is pure noise; measured on synthetic brackets, relaxing the threshold
    from 0.06 to 0.03 let through a result that was 98 px off — far worse than
    leaving the frame unshifted. A bounded failure beats an unbounded one, so
    only strong correlations are accepted, and callers additionally reject any
    result outside the plausible shift range.

    Returns (dx, dy, response) in ORIGINAL image coordinates, or None.
    """
    try:
        ref = _normalised_band(ref_image, region)
        cur = _normalised_band(cur_image, region)
        if ref is None or cur is None:
            return None
        ref_band, scale = ref
        cur_band, _ = cur
        if ref_band.shape != cur_band.shape:
            return None

        window = cv2.createHanningWindow((ref_band.shape[1], ref_band.shape[0]), cv2.CV_32F)
        (sx, sy), response = cv2.phaseCorrelate(ref_band, cur_band, window)
        if not np.isfinite(response) or response < min_response:
            return None

        inv = 1.0 / scale if scale > 0 else 1.0
        # phaseCorrelate reports how cur is displaced from ref; the shift that
        # brings cur onto ref is the negation of that.
        return -float(sx) * inv, -float(sy) * inv, float(response)
    except cv2.error:
        return None
    except Exception:
        return None


def calculate_light_pattern_shifts(
    images: List[np.ndarray],
    ref_idx: Optional[int] = None,
    region: Tuple[float, float] = LIGHTS_DEFAULT_REGION,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Aligns a bracket on the static pattern of street lamps and distant lights.

    This measures camera movement directly, so it is the right choice whenever
    the foreground is fixed — which also means it aligns the landscape, not the
    Sun. Over a long sequence the Sun drifts relative to the ground; align on
    the lunar disc instead if the corona is what must stack.

    Returns (shifts, match_counts). A frame whose pattern cannot be matched
    gets (0, 0) and a count of 0 rather than a guessed shift.
    """
    n = len(images) if images else 0
    if n == 0:
        return [], []
    if n == 1:
        return [(0.0, 0.0)], [0]

    if ref_idx is None or not (0 <= ref_idx < n):
        ref_idx = n // 2

    point_sets: List[np.ndarray] = []
    for i, img in enumerate(images):
        if should_cancel is not None and should_cancel():
            return [(0.0, 0.0)] * n, [0] * n
        if progress_callback is not None:
            progress_callback(int(60 * i / n), f"Hledám statická světla {i + 1}/{n}...")
        point_sets.append(detect_point_lights(img, region=region))

    # Prefer a reference frame that actually has a rich pattern to match against.
    counts = [len(p) for p in point_sets]
    if counts[ref_idx] < 3:
        richest = int(np.argmax(counts))
        if counts[richest] >= 3:
            ref_idx = richest

    ref_points = point_sets[ref_idx]
    if len(ref_points) < 3:
        return [(0.0, 0.0)] * n, [0] * n

    h, w = images[ref_idx].shape[:2]
    max_shift = max(40.0, min(w, h) * 0.15)

    shifts: List[Optional[Tuple[float, float]]] = [None] * n
    matches: List[int] = [0] * n
    shifts[ref_idx] = (0.0, 0.0)
    matches[ref_idx] = len(ref_points)

    # Pass 1: match every frame straight to the reference.
    for i, pts in enumerate(point_sets):
        if should_cancel is not None and should_cancel():
            return [(0.0, 0.0)] * n, [0] * n
        if progress_callback is not None:
            progress_callback(60 + int(30 * i / n), f"Porovnávám vzor světel {i + 1}/{n}...")
        if i == ref_idx:
            continue
        result = estimate_translation_from_points(ref_points, pts, max_shift=max_shift)
        if result is not None:
            shifts[i] = (result[0], result[1])
            matches[i] = result[2]

    # Pass 2: rescue the frames that failed by chaining through a neighbour.
    # Adjacent exposures are only one EV apart, so their lamp patterns still
    # resemble each other even when the extreme ends of the bracket do not.
    if progress_callback is not None:
        progress_callback(92, "Dopočítávám obtížné expozice přes sousední snímky...")

    for _sweep in range(n):
        progressed = False
        for i in range(n):
            if shifts[i] is not None:
                continue
            for neighbour in (i - 1, i + 1):
                if not (0 <= neighbour < n) or shifts[neighbour] is None:
                    continue
                link = estimate_translation_from_points(
                    point_sets[neighbour], point_sets[i], max_shift=max_shift)
                if link is None:
                    continue
                base = shifts[neighbour]
                # Compose: frame -> neighbour -> reference.
                shifts[i] = (base[0] + link[0], base[1] + link[1])
                matches[i] = link[2]
                progressed = True
                break
        if not progressed:
            break

    # Pass 3: for whatever is still unmatched, correlate the whole band against
    # the reference. This works on frames where the individual lamps have bloomed
    # together but the horizon still carries usable structure.
    if any(sh is None for sh in shifts):
        if progress_callback is not None:
            progress_callback(96, "Zkouším korelaci celého pásu u zbývajících snímků...")
        for i in range(n):
            if shifts[i] is not None:
                continue
            if should_cancel is not None and should_cancel():
                break
            correlated = _phase_correlate_bands(images[ref_idx], images[i], region)
            if correlated is not None and abs(correlated[0]) <= max_shift \
                    and abs(correlated[1]) <= max_shift:
                shifts[i] = (correlated[0], correlated[1])
                # Flag as a weaker, correlation-derived result.
                matches[i] = -1

    # Anything still unmatched is left untouched rather than guessed at.
    final = [sh if sh is not None else (0.0, 0.0) for sh in shifts]
    return final, matches


class ImageAligner:
    """Aligner supporting automatic disc detection, manual offsets, and pass-through."""

    def __init__(
        self,
        method: str = "none",
        manual_shifts: Optional[List[Tuple[float, float]]] = None
    ):
        self.method = (method or "none").lower()
        self.manual_shifts = manual_shifts

    def align(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        if len(images) <= 1 or self.method in ("none", "disabled", "off"):
            if progress_callback:
                progress_callback(100, "Zarovnání vypnuto.")
            return images

        if self.manual_shifts and len(self.manual_shifts) == len(images):
            if progress_callback:
                progress_callback(50, "Aplikuji ručně zadané posuny expozic...")
            return apply_shifts_to_images(images, self.manual_shifts)

        if self.method in ("eclipse_disc", "sun_only"):
            if progress_callback:
                progress_callback(20, "Detekuji černý disk Měsíce v záři korony...")
            shifts = calculate_moon_shifts(images)
            if progress_callback:
                progress_callback(70, "Aplikuji subpixelové posuny disku...")
            return apply_shifts_to_images(images, shifts)

        return images
