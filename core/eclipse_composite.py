"""
Eclipse sequence composite: the classic "string of suns" image.

A stacked totality frame is the background. Partial-phase frames shot through
a solar filter each contribute one Sun, placed exactly where the Sun really was
at the moment that frame was taken — so the crescents march along the Sun's
true path through the landscape, growing thinner towards totality and fatter
again afterwards.

How placement is made accurate
------------------------------
The background photograph is modelled as a pinhole camera looking at the sky:
focal length in pixels, plus the camera's azimuth, elevation and roll. Those
four unknowns are solved from what the user marks on the totality frame:

  * the Sun's centre, whose true azimuth/altitude at the frame's timestamp is
    known from the ephemeris (2 constraints),
  * two points on the horizon, i.e. on the altitude = 0 line (2 constraints:
    the camera's roll and elevation),
  * the Sun's diameter in pixels, which fixes the focal length on its own
    (1 constraint, redundant when the horizon is marked).

It is solved as a weighted least-squares problem, so marking both the horizon
and the diameter cross-checks one against the other. Every partial frame is
then projected through the same camera using ITS timestamp. Because the
position is anchored at the totality Sun, a camera clock that is off by the
same amount on every frame cancels out almost completely.

The partial frames themselves never need to share the background's framing:
only the Sun is cut out of each one.
"""

import math
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from core.solar_position import sun_position, sun_angular_radius_deg, sun_path, ecliptic_horizontal
    from core.postprocess import imread_unicode
except ImportError:  # pragma: no cover
    from .solar_position import sun_position, sun_angular_radius_deg, sun_path, ecliptic_horizontal
    from .postprocess import imread_unicode

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"

BLEND_MODES = ("lighten", "screen", "normal")
COLOR_MODES = ("original", "unify", "neutral", "golden")
SCALE_MODES = ("auto", "diameter", "horizon")
ORIENTATION_MODES = ("as_shot", "level")

# BGR chroma of a warm, golden Sun (brightest channel = 1).
GOLDEN_CHROMA = (0.50, 0.80, 1.00)

# How far around the fitted disc a partial frame is cut out, in solar radii.
CROP_RADII = 1.6


class CompositeError(ValueError):
    """A composite cannot be built from the current inputs; the message says why."""


# ------------------------------------------------------------------ Time text

def format_time(moment: Optional[datetime]) -> str:
    return moment.strftime(TIME_FORMAT) if moment is not None else ""


def parse_time(text: str) -> Optional[datetime]:
    if not text:
        return None
    for fmt in (TIME_FORMAT, "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


# ------------------------------------------------------------------ Images

def load_image_float(filepath: str, max_dim: Optional[int] = None) -> Optional[np.ndarray]:
    """
    Decodes an 8- or 16-bit image into float32 BGR in [0, 1].

    The stacked totality background is typically this program's own 16-bit
    TIFF, and squeezing it through 8 bits would band the smooth sky gradient.
    """
    img = imread_unicode(filepath, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.dtype == np.uint16:
        out = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        out = img.astype(np.float32) / 255.0
    else:
        out = np.clip(np.nan_to_num(img.astype(np.float32)), 0.0, 1.0)

    if max_dim:
        h, w = out.shape[:2]
        scale = max_dim / float(max(h, w))
        if scale < 1.0:
            out = cv2.resize(out, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                             interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(out)


# ----------------------------------------------------------- Sun detection

def _fit_circle(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Algebraic (Kåsa) least-squares circle through Nx2 points."""
    if points is None or len(points) < 5:
        return None
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    a = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        (d, e, f), *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy = -d / 2.0, -e / 2.0
    r2 = cx * cx + cy * cy - f
    if not np.isfinite(r2) or r2 <= 0:
        return None
    return float(cx), float(cy), float(math.sqrt(r2))


def _robust_circle(points: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """Circle fit that iteratively drops outliers (horn tips, glow, noise)."""
    circle = _fit_circle(points)
    for _ in range(4):
        if circle is None:
            return None
        cx, cy, r = circle
        resid = np.abs(np.hypot(points[:, 0] - cx, points[:, 1] - cy) - r)
        mad = float(np.median(resid)) * 1.4826
        keep = resid <= max(1.0, 2.5 * mad)
        if keep.sum() < 5 or keep.all():
            break
        points = points[keep]
        circle = _fit_circle(points)
    return circle


def _as_float(image: np.ndarray) -> np.ndarray:
    """float32 [0, 1] view of an image; float input is used as is, not copied."""
    if image.dtype == np.float32:
        return image
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return image.astype(np.float32)


def _brightness(img: np.ndarray) -> np.ndarray:
    """Per-pixel brightness as the brightest channel (HSV value).

    Luminance would call a deep-red setting Sun dim and push its gain until the
    red channel clips; the brightest channel keeps every colour comparable.
    """
    return img.max(axis=2) if img.ndim == 3 else img


def _limb_circle(mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Fits the solar limb to the largest bright blob in a binary mask.

    A crescent's outline is the solar limb (convex) plus the lunar limb
    (concave). Only outline points lying on the blob's convex hull belong to
    the solar limb — the lunar bite never touches the hull — so fitting those
    recovers the whole solar disc even from a thin crescent.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 12 or len(contour) < 12:
        return None

    hull = cv2.convexHull(contour)
    pts = contour.reshape(-1, 2).astype(np.float32)
    on_hull = np.array([cv2.pointPolygonTest(hull, (float(px), float(py)), True)
                        for px, py in pts]) < 1.5
    limb = pts[on_hull] if on_hull.sum() >= 8 else pts
    circle = _robust_circle(limb)
    if circle is None:
        return None
    cx, cy, r = circle
    # Contour points sit on the centres of the outermost lit pixels, half a
    # pixel inside the true edge.
    return cx, cy, r + 0.5


def detect_sun_disc(image_bgr: np.ndarray,
                    detect_max_dim: int = 1400) -> Optional[Tuple[float, float, float]]:
    """
    Finds the (partially eclipsed) solar disc in a filtered frame.

    Returns (cx, cy, radius) in the frame's own pixels, or None when there is
    no Sun in it (a black frame, or cloud).
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    img = _as_float(image_bgr)

    h, w = img.shape[:2]
    scale = min(1.0, detect_max_dim / float(max(h, w)))
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    value = cv2.GaussianBlur(_brightness(small), (3, 3), 0)

    peak = float(np.percentile(value, 99.99))
    floor = float(np.median(value))
    if peak < 0.12 or peak - floor < 0.08:
        return None

    mask = (value > floor + 0.45 * (peak - floor)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    # The Sun is the blob with the most light in it, not merely the largest.
    best, best_score = 0, -1.0
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < 3:
            continue
        score = float(value[labels == label].sum())
        if score > best_score:
            best, best_score = label, score
    if best == 0:
        return None

    x, y, bw, bh = stats[best, :4]
    # Refine at full resolution inside a generous window around the blob.
    pad = int(max(bw, bh) * 0.6 / scale) + 6
    x0 = max(0, int(x / scale) - pad)
    y0 = max(0, int(y / scale) - pad)
    x1 = min(w, int((x + bw) / scale) + pad)
    y1 = min(h, int((y + bh) / scale) + pad)
    window = cv2.GaussianBlur(_brightness(img[y0:y1, x0:x1]), (3, 3), 0)

    local_peak = float(np.percentile(window, 99.9))
    local_floor = float(np.percentile(window, 20))
    if local_peak - local_floor < 0.05:
        return None
    local_mask = window > local_floor + 0.45 * (local_peak - local_floor)
    circle = _limb_circle(local_mask)
    if circle is None:
        return None
    cx, cy, r = circle
    if not (2.0 <= r <= max(h, w)):
        return None
    return cx + x0, cy + y0, r


def detect_totality_disc(image_bgr: np.ndarray,
                         detect_max_dim: int = 2000) -> Optional[Tuple[float, float, float]]:
    """
    Finds the black lunar disc inside the bright inner corona of a totality
    frame, even when it is only a couple of dozen pixels across in a wide
    landscape. Returns (cx, cy, radius) or None.

    The inner corona is the brightest thing in the frame and wraps the Moon as
    a ring; the hole in that ring is the lunar limb.
    """
    if image_bgr is None or image_bgr.size == 0:
        return None
    img = _as_float(image_bgr)
    h, w = img.shape[:2]
    scale = min(1.0, detect_max_dim / float(max(h, w)))
    small = cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else img
    value = cv2.GaussianBlur(_brightness(small), (5, 5), 0)
    peak = float(value.max())
    if peak < 0.15:
        return None

    best = None
    for frac in (0.5, 0.35, 0.65, 0.25):
        ring = (value > frac * peak).astype(np.uint8)
        contours, hierarchy = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for idx, contour in enumerate(contours):
            parent = hierarchy[0][idx][3]
            if parent < 0:
                continue            # only holes inside bright regions
            area = cv2.contourArea(contour)
            if area < 12:
                continue
            (hx, hy), hr = cv2.minEnclosingCircle(contour)
            circularity = area / (math.pi * hr * hr + 1e-9)
            if circularity < 0.6:
                continue
            inside = value[int(max(0, hy - hr * 0.5)):int(hy + hr * 0.5) + 1,
                           int(max(0, hx - hr * 0.5)):int(hx + hr * 0.5) + 1]
            darkness = 1.0 - float(inside.mean()) / peak if inside.size else 0.0
            score = circularity * darkness * math.sqrt(area)
            if best is None or score > best[0]:
                best = (score, contour)
        if best is not None:
            break
    if best is None:
        return None

    circle = _robust_circle(best[1].reshape(-1, 2).astype(np.float32))
    if circle is None:
        return None
    cx, cy, r = circle
    cx, cy, r = (cx + 0.5) / scale - 0.5, (cy + 0.5) / scale - 0.5, (r + 0.5) / scale
    # Refine on a window around the disc: remapping rays over a whole 24 Mpx
    # frame would copy it once per ray.
    pad = int(math.ceil(2.2 * r)) + 4
    x0, y0 = max(0, int(cx) - pad), max(0, int(cy) - pad)
    x1, y1 = min(w, int(cx) + pad + 1), min(h, int(cy) + pad + 1)
    window = np.ascontiguousarray(_brightness(img[y0:y1, x0:x1]), dtype=np.float32)
    refined = _refine_dark_limb(window, cx - x0, cy - y0, r)
    if refined is None:
        return cx, cy, r
    return refined[0] + x0, refined[1] + y0, refined[2]


def _refine_dark_limb(value: np.ndarray, cx: float, cy: float, r: float,
                      rays: int = 180) -> Optional[Tuple[float, float, float]]:
    """
    Subpixel lunar limb: along rays from the centre, the point where brightness
    crosses halfway between the dark disc and the bright ring just outside it.

    Thresholding a blurred image pulls the ring inwards and shrinks the disc by
    a pixel or more — a 5 % scale error on a 20 px Moon. Half-maximum crossings
    on the unblurred profile do not have that bias.
    """
    h, w = value.shape[:2]
    inner = value[int(max(0, cy - 0.4 * r)):int(cy + 0.4 * r) + 1,
                  int(max(0, cx - 0.4 * r)):int(cx + 0.4 * r) + 1]
    if inner.size == 0:
        return None
    dark = float(np.median(inner))
    radii = np.linspace(0.3 * r, 1.9 * r, max(40, int(1.6 * r * 6)))
    edge_pts = []
    for theta in np.linspace(0.0, 2.0 * math.pi, rays, endpoint=False):
        xs = cx + radii * math.cos(theta)
        ys = cy + radii * math.sin(theta)
        if xs.min() < 0 or ys.min() < 0 or xs.max() > w - 2 or ys.max() > h - 2:
            continue
        profile = cv2.remap(value, xs.astype(np.float32).reshape(1, -1),
                            ys.astype(np.float32).reshape(1, -1), cv2.INTER_LINEAR).ravel()
        top = int(np.argmax(profile))
        bright = float(profile[top])
        if bright - dark < 0.1:
            continue
        half = dark + 0.5 * (bright - dark)
        below = np.nonzero(profile[:top + 1] < half)[0]
        if below.size == 0:
            continue
        i = int(below[-1])
        if i + 1 >= profile.size:
            continue
        a, b = float(profile[i]), float(profile[i + 1])
        t = (half - a) / (b - a) if b != a else 0.5
        rad = radii[i] + t * (radii[i + 1] - radii[i])
        edge_pts.append((cx + rad * math.cos(theta), cy + rad * math.sin(theta)))
    if len(edge_pts) < rays // 3:
        return None
    return _robust_circle(np.array(edge_pts, dtype=np.float64))


def measure_sun_surface(crop_bgr: np.ndarray, disc: Tuple[float, float, float]
                        ) -> Tuple[float, Tuple[float, float, float]]:
    """
    (surface brightness, BGR chroma) of the lit photosphere inside the disc.

    Brightness is the median brightest-channel value of the lit pixels, away
    from the limb, where limb darkening would pull it down. Chroma is the mean
    colour of the same pixels scaled so its brightest channel is 1.
    """
    cx, cy, r = disc
    h, w = crop_bgr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= (0.9 * r) ** 2
    value = _brightness(crop_bgr)
    if inside.sum() < 4:
        return 1.0, (1.0, 1.0, 1.0)
    vals = value[inside]
    peak = float(np.percentile(vals, 99))
    lit = inside & (value > 0.5 * peak)
    if lit.sum() < 4:
        lit = inside
    level = float(np.median(value[lit]))
    mean = crop_bgr[lit].reshape(-1, 3).mean(axis=0)
    top = float(mean.max())
    chroma = tuple(float(c / top) for c in mean) if top > 1e-6 else (1.0, 1.0, 1.0)
    return max(level, 1e-4), chroma


@dataclass
class SunCutout:
    """A partial-phase Sun cut out of its frame at full resolution."""
    crop: np.ndarray                      # float32 BGR [0, 1]
    disc: Tuple[float, float, float]      # centre and radius inside `crop`
    level: float
    chroma: Tuple[float, float, float]


def cut_out_sun(image_f32: np.ndarray, disc: Tuple[float, float, float]) -> SunCutout:
    """Cuts a square of CROP_RADII solar radii around the disc, padding with black."""
    cx, cy, r = disc
    half = int(math.ceil(CROP_RADII * r)) + 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    size = 2 * half + 1
    h, w = image_f32.shape[:2]
    crop = np.zeros((size, size, 3), np.float32)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + size), min(h, y0 + size)
    if sx1 > sx0 and sy1 > sy0:
        crop[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image_f32[sy0:sy1, sx0:sx1]
    local = (cx - x0, cy - y0, r)
    level, chroma = measure_sun_surface(crop, local)
    return SunCutout(crop=crop, disc=local, level=level, chroma=chroma)


# ------------------------------------------------------------- Camera model

def _direction(az_deg: float, alt_deg: float) -> np.ndarray:
    """Unit vector (east, north, up) for an azimuth / altitude."""
    az, alt = math.radians(az_deg), math.radians(alt_deg)
    return np.array([math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)])


def _az_alt(vec: np.ndarray) -> Tuple[float, float]:
    v = vec / max(1e-12, float(np.linalg.norm(vec)))
    alt = math.degrees(math.asin(max(-1.0, min(1.0, float(v[2])))))
    az = math.degrees(math.atan2(float(v[0]), float(v[1]))) % 360.0
    return az, alt


@dataclass
class SkyCamera:
    """
    Pinhole camera looking at the sky.

    yaw   — azimuth of the optical axis (degrees from north through east)
    pitch — altitude of the optical axis
    roll  — rotation about the axis; positive tilts the horizon down to the right
    focal — focal length in pixels; the principal point is the image centre.
    """
    width: int
    height: int
    focal: float
    yaw: float
    pitch: float
    roll: float

    def basis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _direction(self.yaw, self.pitch)
        yaw = math.radians(self.yaw)
        right0 = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
        up0 = np.cross(right0, forward)
        phi = math.radians(self.roll)
        right = right0 * math.cos(phi) + up0 * math.sin(phi)
        up = up0 * math.cos(phi) - right0 * math.sin(phi)
        return forward, right, up

    @property
    def centre(self) -> Tuple[float, float]:
        return (self.width - 1) / 2.0, (self.height - 1) / 2.0

    def project_vector(self, vec: np.ndarray) -> Optional[Tuple[float, float]]:
        forward, right, up = self.basis()
        z = float(vec @ forward)
        if z <= 1e-6:
            return None
        cx, cy = self.centre
        return (cx + self.focal * float(vec @ right) / z,
                cy - self.focal * float(vec @ up) / z)

    def project(self, az_deg: float, alt_deg: float) -> Optional[Tuple[float, float]]:
        """Pixel position of a sky direction, or None when it is behind the camera."""
        return self.project_vector(_direction(az_deg, alt_deg))

    def ray(self, x: float, y: float) -> np.ndarray:
        forward, right, up = self.basis()
        cx, cy = self.centre
        vec = forward * self.focal + right * (x - cx) - up * (y - cy)
        return vec / float(np.linalg.norm(vec))

    def unproject(self, x: float, y: float) -> Tuple[float, float]:
        """(azimuth, altitude) seen at a pixel."""
        return _az_alt(self.ray(x, y))

    @property
    def pixels_per_degree(self) -> float:
        return self.focal * math.pi / 180.0

    def local_sun(self, az_deg: float, alt_deg: float, radius_deg: float
                  ) -> Optional[Tuple[float, float, float, float]]:
        """
        (x, y, radius_px, zenith_angle_deg) of a solar disc at that direction.

        The radius is measured through the projection, so a Sun near the frame
        corner is drawn with the same slight stretch the lens gave the scenery.
        zenith_angle is the on-image direction of "up in the sky", clockwise
        from image-up — what a crescent shot with a level camera must be
        rotated by to sit correctly in this frame.
        """
        centre = self.project(az_deg, alt_deg)
        top = self.project(az_deg, alt_deg + radius_deg)
        bottom = self.project(az_deg, alt_deg - radius_deg)
        if centre is None or top is None or bottom is None:
            return None
        radius = 0.5 * math.hypot(top[0] - bottom[0], top[1] - bottom[1])
        angle = math.degrees(math.atan2(top[0] - bottom[0], -(top[1] - bottom[1])))
        return centre[0], centre[1], radius, angle


@dataclass
class Calibration:
    """What the user marked on the background frame, in full-resolution pixels."""
    width: int
    height: int
    sun_x: float
    sun_y: float
    sun_diameter: float = 0.0
    horizon: Optional[Tuple[float, float, float, float]] = None
    horizon_altitude: float = 0.0


@dataclass
class CalibrationReport:
    camera: SkyCamera
    sun_residual_px: float
    horizon_rms_px: Optional[float]
    diameter_model_px: float
    diameter_measured_px: float
    focal_from_diameter: Optional[float]
    focal_from_horizon: Optional[float]
    scale_source: str
    warnings: List[str] = field(default_factory=list)


def _horizon_focal_estimate(cal: Calibration, sun_alt: float) -> Optional[float]:
    """Focal length implied by the Sun's pixel height above the horizon line."""
    if cal.horizon is None:
        return None
    x1, y1, x2, y2 = cal.horizon
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1.0:
        return None
    # Perpendicular pixel distance of the Sun from the horizon line.
    dist = abs((x2 - x1) * (y1 - cal.sun_y) - (x1 - cal.sun_x) * (y2 - y1)) / length
    elevation = math.radians(sun_alt - cal.horizon_altitude)
    if elevation < math.radians(1.5):
        return None
    return dist / math.tan(elevation)


def solve_camera(cal: Calibration, sun_az: float, sun_alt: float, sun_radius_deg: float,
                 scale_mode: str = "auto") -> CalibrationReport:
    """
    Solves the background camera from the marked Sun, horizon and diameter.

    Weighted Levenberg–Marquardt over (yaw, pitch, roll, log focal). Weights
    express how precisely each mark can be clicked: the Sun's centre to half a
    pixel, a horizon point to about 1.5 px, the diameter to ~4 %.
    """
    if cal.width <= 0 or cal.height <= 0:
        raise CompositeError("Chybí rozměry pozadí.")
    has_diameter = cal.sun_diameter >= 2.0
    has_horizon = cal.horizon is not None and math.hypot(
        cal.horizon[2] - cal.horizon[0], cal.horizon[3] - cal.horizon[1]) >= 10.0

    f_diam = (cal.sun_diameter / (2.0 * math.tan(math.radians(sun_radius_deg)))
              if has_diameter else None)
    f_hor = _horizon_focal_estimate(cal, sun_alt) if has_horizon else None

    warnings: List[str] = []
    use_diameter = has_diameter
    if scale_mode == "diameter":
        if not has_diameter:
            raise CompositeError("Pro měřítko podle průměru vyznačte průměr Slunce.")
    elif scale_mode == "horizon":
        use_diameter = False
        if f_hor is None:
            if not has_diameter:
                raise CompositeError(
                    "Měřítko z horizontu vyžaduje vyznačený horizont a Slunce alespoň "
                    "1,5° nad ním. Vyznačte i průměr Slunce.")
            warnings.append("Slunce je příliš nízko pro měřítko z horizontu — použit průměr Slunce.")
            use_diameter = True
    if not use_diameter and f_hor is None:
        raise CompositeError("Vyznačte průměr Slunce nebo horizont (se Sluncem nad ním).")

    f0 = f_diam if (use_diameter and f_diam) else f_hor
    if f0 is None or not np.isfinite(f0) or f0 <= 0:
        raise CompositeError("Nelze určit měřítko snímku.")

    sun_vec = _direction(sun_az, sun_alt)
    cx0, cy0 = (cal.width - 1) / 2.0, (cal.height - 1) / 2.0

    sigma_sun, sigma_hor = 0.5, 1.5
    sigma_diam = max(0.6, 0.04 * cal.sun_diameter)
    if scale_mode == "diameter":
        sigma_diam = 0.05          # the diameter is authoritative
    horizon_pts = []
    if has_horizon:
        x1, y1, x2, y2 = cal.horizon
        # Several points along the marked segment weight the line evenly.
        horizon_pts = [(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t) for t in (0.0, 0.5, 1.0)]

    def residuals(p: np.ndarray) -> np.ndarray:
        cam = SkyCamera(cal.width, cal.height, math.exp(p[3]), p[0], p[1], p[2])
        out = []
        pos = cam.project_vector(sun_vec)
        if pos is None:
            out += [1e4, 1e4]
        else:
            out += [(pos[0] - cal.sun_x) / sigma_sun, (pos[1] - cal.sun_y) / sigma_sun]
        if has_horizon:
            for hx, hy in horizon_pts:
                _az, alt = cam.unproject(hx, hy)
                out.append((alt - cal.horizon_altitude) * cam.pixels_per_degree / sigma_hor)
        else:
            out.append(p[2] / 0.05)    # no horizon marked: assume a level camera
        if use_diameter:
            model = 2.0 * cam.focal * math.tan(math.radians(sun_radius_deg))
            out.append((model - cal.sun_diameter) / sigma_diam)
        return np.array(out, dtype=np.float64)

    def initial(roll: float) -> np.ndarray:
        # Point the axis so the Sun lands roughly on its marked pixel.
        yaw = sun_az - math.degrees(math.atan2(cal.sun_x - cx0, f0))
        pitch = sun_alt + math.degrees(math.atan2(cal.sun_y - cy0, f0))
        return np.array([yaw, pitch, roll, math.log(f0)])

    roll0 = 0.0
    if has_horizon:
        x1, y1, x2, y2 = cal.horizon
        if x2 < x1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        roll0 = math.degrees(math.atan2(y2 - y1, x2 - x1))

    best_p, best_cost = None, float("inf")
    for start in (initial(roll0), initial(-roll0)) if roll0 else (initial(0.0),):
        p = start.copy()
        lam = 1e-3
        cost = float(np.sum(residuals(p) ** 2))
        for _ in range(100):
            r = residuals(p)
            jac = np.empty((r.size, 4))
            for k, eps in enumerate((1e-5, 1e-5, 1e-5, 1e-7)):
                dp = np.zeros(4)
                dp[k] = eps
                jac[:, k] = (residuals(p + dp) - r) / eps
            jtj = jac.T @ jac
            grad = jac.T @ r
            improved = False
            for _inner in range(10):
                try:
                    step = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj) + 1e-9), -grad)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                trial = p + step
                trial_cost = float(np.sum(residuals(trial) ** 2))
                if trial_cost < cost:
                    p, cost = trial, trial_cost
                    lam = max(1e-9, lam * 0.3)
                    improved = True
                    break
                lam *= 10
            if not improved or float(np.max(np.abs(step))) < 1e-9:
                break
        if cost < best_cost:
            best_p, best_cost = p, cost

    cam = SkyCamera(cal.width, cal.height, math.exp(best_p[3]),
                    best_p[0] % 360.0, best_p[1], best_p[2])

    pos = cam.project_vector(sun_vec)
    sun_res = math.hypot(pos[0] - cal.sun_x, pos[1] - cal.sun_y) if pos else float("inf")
    horizon_rms = None
    if has_horizon:
        errs = [(cam.unproject(hx, hy)[1] - cal.horizon_altitude) * cam.pixels_per_degree
                for hx, hy in horizon_pts]
        horizon_rms = float(math.sqrt(np.mean(np.square(errs))))
    model_diam = 2.0 * cam.focal * math.tan(math.radians(sun_radius_deg))

    if sun_alt < cal.horizon_altitude:
        warnings.append("Podle výpočtu je Slunce pod horizontem — zkontrolujte čas, "
                        "časové pásmo a polohu.")
    if has_diameter and f_hor is not None and abs(f_diam - f_hor) > 0.12 * f_hor:
        warnings.append(
            f"Průměr Slunce ({cal.sun_diameter:.1f} px) a výška nad horizontem si odporují "
            f"o {abs(f_diam - f_hor) / f_hor * 100:.0f} % — zkontrolujte čas, polohu, "
            "výšku horizontu nebo průměr.")
    if horizon_rms is not None and horizon_rms > 4.0:
        warnings.append(f"Horizont nesedí s modelem (odchylka {horizon_rms:.1f} px).")

    source = ("průměr Slunce + horizont" if use_diameter and has_horizon
              else "průměr Slunce" if use_diameter else "výška Slunce nad horizontem")
    return CalibrationReport(
        camera=cam, sun_residual_px=sun_res, horizon_rms_px=horizon_rms,
        diameter_model_px=model_diam, diameter_measured_px=cal.sun_diameter,
        focal_from_diameter=f_diam, focal_from_horizon=f_hor,
        scale_source=source, warnings=warnings,
    )


# ------------------------------------------------------------ Settings model

@dataclass
class ColorGrade:
    """
    Photo-editor style adjustments. Every value is 0 when neutral, so two
    grades simply add up: the master grade of all Suns plus one frame's own.
    """
    exposure: float = 0.0       # EV
    contrast: float = 0.0       # -100 .. 100, around the subject's own mid level
    midtones: float = 0.0       # -100 .. 100, black and white stay where they are
    saturation: float = 0.0     # -100 (greyscale) .. 100
    temperature: float = 0.0    # -100 (cooler) .. 100 (warmer)
    tint: float = 0.0           # -100 (greener) .. 100 (more magenta)

    def is_neutral(self) -> bool:
        return all(abs(getattr(self, name)) < 1e-6 for name in GRADE_LIMITS)

    def combined(self, other: "ColorGrade") -> "ColorGrade":
        """This grade with `other` added on top, each value kept in range."""
        return ColorGrade(**{name: min(hi, max(lo, getattr(self, name) + getattr(other, name)))
                             for name, (lo, hi) in GRADE_LIMITS.items()})

    @classmethod
    def from_dict(cls, data: Any) -> "ColorGrade":
        grade = cls()
        if isinstance(data, dict):
            for name, (lo, hi) in GRADE_LIMITS.items():
                try:
                    setattr(grade, name, min(hi, max(lo, float(data.get(name, 0.0)))))
                except (TypeError, ValueError):
                    pass
        return grade


# name -> (lowest, highest) value of each adjustment.
GRADE_LIMITS = {
    "exposure": (-6.0, 6.0),
    "contrast": (-100.0, 100.0),
    "midtones": (-100.0, 100.0),
    "saturation": (-100.0, 100.0),
    "temperature": (-100.0, 100.0),
    "tint": (-100.0, 100.0),
}


@dataclass
class PartialFrame:
    """One filtered partial-phase photograph and every decision made about it."""
    path: str
    filename: str = ""
    time: str = ""                  # local camera time, TIME_FORMAT
    time_from_exif: bool = False
    enabled: bool = True
    disc: Optional[Tuple[float, float, float]] = None   # full-res, in its own frame
    auto_brightness: bool = True
    grade: ColorGrade = field(default_factory=ColorGrade)   # on top of the master grade
    offset_x: float = 0.0           # manual nudge, background full-res pixels
    offset_y: float = 0.0
    rotation: float = 0.0           # extra rotation, degrees clockwise
    scale: float = 1.0              # extra size factor for this Sun alone

    @property
    def moment(self) -> Optional[datetime]:
        return parse_time(self.time)


@dataclass
class CompositeSettings:
    """The whole composite — persisted inside the project file."""
    background_path: str = ""
    background_time: str = ""
    utc_offset_hours: float = 2.0
    latitude: float = 0.0
    longitude: float = 0.0
    location_set: bool = False
    sun_x: float = 0.0
    sun_y: float = 0.0
    sun_diameter: float = 0.0
    horizon: Optional[Tuple[float, float, float, float]] = None
    horizon_altitude: float = 0.0
    scale_mode: str = "auto"
    clock_offset_s: float = 0.0     # added to every partial frame's timestamp
    target_level: float = 0.85      # surface brightness every Sun is equalised to
    sun_grade: ColorGrade = field(default_factory=ColorGrade)          # master, all Suns
    background_grade: ColorGrade = field(default_factory=ColorGrade)   # the totality frame
    color_mode: str = "original"
    blend_mode: str = "lighten"
    size_multiplier: float = 1.0
    orientation_mode: str = "as_shot"
    clip_below_horizon: bool = True
    edge_softness: float = 0.04
    show_path: bool = True
    show_ticks: bool = True
    tick_minutes: int = 10
    show_ecliptic: bool = False
    show_calibration: bool = True
    show_markers: bool = True
    frames: List[PartialFrame] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["frames"] = [asdict(f) for f in self.frames]
        return data

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "CompositeSettings":
        """Tolerant of missing and unknown keys, like the project format itself."""
        if not isinstance(data, dict):
            return cls()
        allowed = {f.name for f in fields(cls)} - {"frames"}
        kwargs = {k: v for k, v in data.items() if k in allowed}
        settings = cls(**kwargs)
        settings.horizon = _as_tuple(settings.horizon, 4)
        settings.background_grade = ColorGrade.from_dict(data.get("background_grade"))
        settings.sun_grade = ColorGrade.from_dict(data.get("sun_grade"))
        if "sun_grade" not in data:
            # Older projects set the Suns' brightness as a target level only;
            # the master exposure now carries it.
            try:
                level = float(settings.target_level)
            except (TypeError, ValueError):
                level = 0.0
            if level > 0:
                settings.sun_grade.exposure = round(math.log2(level / 0.85), 3)
            settings.target_level = 0.85
        frame_keys = {f.name for f in fields(PartialFrame)}
        for raw in data.get("frames") or []:
            if isinstance(raw, dict) and raw.get("path"):
                frame = PartialFrame(**{k: v for k, v in raw.items() if k in frame_keys})
                frame.disc = _as_tuple(frame.disc, 3)
                frame.grade = ColorGrade.from_dict(raw.get("grade"))
                if "grade" not in raw and "ev_adjust" in raw:
                    # Older projects: the per-frame EV correction.
                    frame.grade = ColorGrade.from_dict({"exposure": raw.get("ev_adjust")})
                settings.frames.append(frame)
        if settings.blend_mode not in BLEND_MODES:
            settings.blend_mode = "lighten"
        if settings.color_mode not in COLOR_MODES:
            settings.color_mode = "original"
        if settings.scale_mode not in SCALE_MODES:
            settings.scale_mode = "auto"
        if settings.orientation_mode not in ORIENTATION_MODES:
            settings.orientation_mode = "as_shot"
        return settings

    @property
    def background_moment(self) -> Optional[datetime]:
        return parse_time(self.background_time)

    def frame_moment(self, frame: PartialFrame) -> Optional[datetime]:
        moment = frame.moment
        if moment is None:
            return None
        return moment + timedelta(seconds=float(self.clock_offset_s))


def _as_tuple(value: Any, n: int) -> Optional[tuple]:
    if not isinstance(value, (list, tuple)) or len(value) != n:
        return None
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None


# -------------------------------------------------------------- Geometry

@dataclass
class Placement:
    """Where and how one partial Sun lands in the background (full-res px)."""
    index: int
    x: float
    y: float
    radius: float
    rotation: float
    az: float
    alt: float
    base_x: float = 0.0     # the computed position, before the manual offset
    base_y: float = 0.0


def calibrate(settings: CompositeSettings, width: int, height: int) -> CalibrationReport:
    """Solves the background camera from the settings' marks and timestamp."""
    moment = settings.background_moment
    if moment is None:
        raise CompositeError("Zadejte čas snímku úplného zatmění.")
    if not settings.location_set:
        raise CompositeError("Zadejte zeměpisnou polohu pozorování.")
    if settings.sun_x <= 0 and settings.sun_y <= 0:
        raise CompositeError("Vyznačte Slunce na snímku úplného zatmění.")
    az, alt = sun_position(moment, settings.latitude, settings.longitude, settings.utc_offset_hours)
    radius = sun_angular_radius_deg(moment, settings.utc_offset_hours)
    cal = Calibration(width=width, height=height, sun_x=settings.sun_x, sun_y=settings.sun_y,
                      sun_diameter=settings.sun_diameter, horizon=settings.horizon,
                      horizon_altitude=settings.horizon_altitude)
    return solve_camera(cal, az, alt, radius, settings.scale_mode)


def compute_placements(settings: CompositeSettings, camera: SkyCamera) -> List[Optional[Placement]]:
    """One placement per frame (None for disabled, timeless or off-camera frames)."""
    placements: List[Optional[Placement]] = []
    for idx, frame in enumerate(settings.frames):
        moment = settings.frame_moment(frame)
        if not frame.enabled or moment is None:
            placements.append(None)
            continue
        az, alt = sun_position(moment, settings.latitude, settings.longitude,
                               settings.utc_offset_hours)
        radius_deg = sun_angular_radius_deg(moment, settings.utc_offset_hours)
        local = camera.local_sun(az, alt, radius_deg)
        if local is None:
            placements.append(None)
            continue
        x, y, radius, zenith = local
        rotation = frame.rotation + (zenith if settings.orientation_mode == "level" else 0.0)
        placements.append(Placement(
            index=idx, x=x + frame.offset_x, y=y + frame.offset_y,
            radius=radius * settings.size_multiplier * max(0.05, frame.scale),
            rotation=rotation, az=az, alt=alt, base_x=x, base_y=y))
    return placements


def path_polyline(settings: CompositeSettings, camera: SkyCamera, step_minutes: float = 1.0,
                  span_hours: float = 5.0) -> List[Tuple[datetime, float, float]]:
    """(local time, x, y) along the Sun's daily path, only where it is in view."""
    centre = settings.background_moment
    if centre is None:
        return []
    frame_times = [settings.frame_moment(f) for f in settings.frames]
    frame_times = [t for t in frame_times if t is not None]
    start = min([centre] + frame_times) - timedelta(hours=span_hours / 2)
    # Whole minutes, so time ticks fall exactly on samples.
    start = start.replace(second=0, microsecond=0)
    end = max([centre] + frame_times) + timedelta(hours=span_hours / 2)
    margin = max(camera.width, camera.height)
    points = []
    for moment, az, alt in sun_path(start, end, step_minutes, settings.latitude,
                                    settings.longitude, settings.utc_offset_hours):
        pos = camera.project(az, alt)
        if pos is None or not (-margin <= pos[0] <= camera.width + margin
                               and -margin <= pos[1] <= camera.height + margin):
            points.append((moment, float("nan"), float("nan")))
        else:
            points.append((moment, pos[0], pos[1]))
    return points


def ecliptic_polyline(settings: CompositeSettings, camera: SkyCamera) -> List[Tuple[float, float]]:
    """The ecliptic at the totality instant, projected (NaN breaks the line)."""
    moment = settings.background_moment
    if moment is None:
        return []
    margin = max(camera.width, camera.height)
    points = []
    for _lam, az, alt in ecliptic_horizontal(moment, settings.latitude, settings.longitude,
                                             settings.utc_offset_hours, step_deg=0.5):
        pos = camera.project(az, alt)
        if pos is None or not (-margin <= pos[0] <= camera.width + margin
                               and -margin <= pos[1] <= camera.height + margin):
            points.append((float("nan"), float("nan")))
        else:
            points.append(pos)
    return points


def horizon_polyline(settings: CompositeSettings, camera: SkyCamera, samples: int = 120
                     ) -> List[Tuple[float, float]]:
    """The model's horizon (altitude = horizon_altitude) across the frame."""
    points = []
    _az_c, _alt_c = camera.unproject(camera.width / 2.0, camera.height / 2.0)
    left_az, _ = camera.unproject(0, camera.height / 2.0)
    right_az, _ = camera.unproject(camera.width - 1, camera.height / 2.0)
    span = (right_az - left_az) % 360.0
    for i in range(samples + 1):
        az = left_az - 0.1 * span + 1.2 * span * i / samples
        pos = camera.project(az, settings.horizon_altitude)
        points.append(pos if pos is not None else (float("nan"), float("nan")))
    return points


# ----------------------------------------------------------- Colour & gain

def frame_gains(settings: CompositeSettings, cutouts: Sequence[Optional[SunCutout]]
                ) -> List[Tuple[float, Optional[np.ndarray]]]:
    """
    Per frame: (brightness gain, target BGR chroma or None to keep the colour).

    The gain equalises the photosphere's surface brightness to
    `target_level`, then applies the user's EV correction. A colour treatment
    re-tints each pixel's brightness rather than multiplying channels — a deep
    red setting Sun has almost nothing in its blue channel, and amplifying that
    thirtyfold would only amplify noise and JPEG blocks.
    """
    chromas = [np.array(c.chroma) for f, c in zip(settings.frames, cutouts)
               if c is not None and f.enabled]
    if settings.color_mode == "unify" and chromas:
        # The median ignores a single reddened frame near the horizon.
        target = np.median(np.array(chromas), axis=0)
        target = target / max(1e-6, float(target.max()))
    elif settings.color_mode == "neutral":
        target = np.ones(3)
    elif settings.color_mode == "golden":
        target = np.array(GOLDEN_CHROMA)
    else:
        target = None

    gains = []
    for frame, cut in zip(settings.frames, cutouts):
        level = cut.level if cut is not None else 1.0
        base = settings.target_level / level if frame.auto_brightness else 1.0
        exposure = settings.sun_grade.combined(frame.grade).exposure
        gain = float(base * (2.0 ** exposure))
        gains.append((gain, None if target is None else target.astype(np.float32)))
    return gains


# Luma weights (BGR), as in the main post-processing.
_LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)

# Rows per band when grading a large image, to bound the temporaries.
GRADE_BAND_ROWS = 512


def white_balance_gains(temperature: float, tint: float) -> np.ndarray:
    """
    BGR channel multipliers for a temperature / tint shift (-100 .. 100),
    normalised so that the luma of a grey pixel is unchanged.
    """
    t, g = float(temperature) / 100.0, float(tint) / 100.0
    gains = np.array([2.0 ** (-0.5 * t + 0.15 * g),     # blue
                      2.0 ** (-0.3 * g),                # green
                      2.0 ** (0.5 * t + 0.15 * g)],     # red
                     np.float32)
    return gains / float(np.dot(_LUMA_BGR, gains))


def _grade_in_place(img: np.ndarray, grade: ColorGrade, gain: float, pivot: float):
    """
    Grades float32 BGR pixels in place: white balance and `gain` (which already
    includes the exposure), then contrast, mid-tones and saturation.

    The tone curves keep black at black and white at white, and their slope
    stays finite at zero: the black sky around a cut-out Sun, and the Moon
    biting into it, must never turn grey — with the lighten blend a grey Moon
    would show as a pale disc over the landscape.
    """
    wb = white_balance_gains(grade.temperature, grade.tint) * float(gain)
    img *= wb.reshape(1, 1, 3)
    np.clip(img, 0.0, 1.0, out=img)

    # Contrast: piecewise quadratic through (0, 0), (pivot, pivot) and (1, 1)
    # with slope c at the pivot; monotonic for c in [0, 2].
    c = 1.0 + min(100.0, max(-100.0, grade.contrast)) / 100.0
    if abs(c - 1.0) > 1e-4:
        p = min(0.9, max(0.1, float(pivot)))
        weight = np.where(img < p, img / p, (1.0 - img) / (1.0 - p))
        weight *= img - p
        img += (c - 1.0) * weight

    # Mid-tones: y = x + a x (1 - x) lifts or lowers the middle only.
    a = min(100.0, max(-100.0, grade.midtones)) / 100.0
    if abs(a) > 1e-4:
        img += a * img * (1.0 - img)

    s = 1.0 + min(100.0, max(-100.0, grade.saturation)) / 100.0
    if abs(s - 1.0) > 1e-4:
        luma = img @ _LUMA_BGR
        img -= luma[:, :, None]
        img *= s
        img += luma[:, :, None]
    np.clip(img, 0.0, 1.0, out=img)


def apply_grade(image: np.ndarray, grade: ColorGrade, pivot: float = 0.5,
                band_rows: int = GRADE_BAND_ROWS) -> np.ndarray:
    """
    A graded float32 copy of a whole BGR image (the totality background).

    Large images are processed in horizontal bands, so a 45 Mpx export needs
    only the output array plus a band's worth of temporaries.
    """
    out = np.array(image, dtype=np.float32, copy=True)
    if grade.is_neutral():
        return out
    gain = 2.0 ** grade.exposure
    step = max(1, int(band_rows))
    for y0 in range(0, out.shape[0], step):
        _grade_in_place(out[y0:y0 + step], grade, gain, pivot)
    return out


# ------------------------------------------------------------- Rendering

def _horizon_sky_sign(settings: CompositeSettings) -> float:
    """+1 or -1: which side of the marked horizon line the sky is on."""
    x1, y1, x2, y2 = settings.horizon
    side = (x2 - x1) * (settings.sun_y - y1) - (y2 - y1) * (settings.sun_x - x1)
    if abs(side) < 1e-9:
        # Fall back to "above in the image".
        side = (x2 - x1) * (-1e6 - y1)
    return 1.0 if side > 0 else -1.0


def _render_one(out: np.ndarray, cut: SunCutout, place: Placement,
                gain: Tuple[float, Optional[np.ndarray]], grade: ColorGrade,
                settings: CompositeSettings, scale: float, sky_sign: float):
    """
    Warps one cut-out Sun into `out` (float32 BGR, modified in place).

    `gain` already carries the grade's exposure; the rest of `grade` (the
    master grade plus this frame's own) is applied here.
    """
    h, w = out.shape[:2]
    X, Y = place.x * scale, place.y * scale
    radius = place.radius * scale
    src = cut.crop
    scx, scy, sr = cut.disc
    k = radius / max(sr, 1e-6)

    # Shrinking by a large factor through warpAffine alone would alias the limb:
    # pre-shrink with area averaging and leave only a mild warp.
    if k < 0.7:
        pre = max(k, 0.02)
        sh, sw = src.shape[:2]
        nw, nh = max(4, int(round(sw * pre))), max(4, int(round(sh * pre)))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)
        fx, fy = nw / float(sw), nh / float(sh)
        scx, scy = (scx + 0.5) * fx - 0.5, (scy + 0.5) * fy - 0.5
        k = k / ((fx + fy) / 2.0)

    half = int(math.ceil(CROP_RADII * radius)) + 3
    ox, oy = int(math.floor(X)) - half, int(math.floor(Y)) - half
    size = 2 * half + 1
    x0, y0 = max(0, ox), max(0, oy)
    x1, y1 = min(w, ox + size), min(h, oy + size)
    if x1 <= x0 or y1 <= y0:
        return

    m = cv2.getRotationMatrix2D((scx, scy), -place.rotation, k)
    m[0, 2] += (X - ox) - scx
    m[1, 2] += (Y - oy) - scy
    patch = cv2.warpAffine(src, m, (size, size), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    factor, chroma = gain
    if chroma is not None:
        patch = _brightness(patch)[:, :, None] * chroma.reshape(1, 1, 3)
    # Contrast pivots on the Sun's own surface level, so it deepens the limb
    # darkening and sunspots instead of just brightening the whole disc.
    _grade_in_place(patch, grade, factor, pivot=cut.level * factor)

    yy, xx = np.mgrid[oy:oy + size, ox:ox + size].astype(np.float32)
    dist = np.hypot(xx - X, yy - Y)
    edge = max(0.75, settings.edge_softness * radius)
    alpha = np.clip((radius + 0.5 + edge * 0.5 - dist) / edge, 0.0, 1.0)

    if settings.clip_below_horizon and settings.horizon is not None:
        hx1, hy1, hx2, hy2 = [v * scale for v in settings.horizon]
        length = max(1e-6, math.hypot(hx2 - hx1, hy2 - hy1))
        signed = sky_sign * ((hx2 - hx1) * (yy - hy1) - (hy2 - hy1) * (xx - hx1)) / length
        alpha *= np.clip(signed / 1.5 + 0.5, 0.0, 1.0)

    sub = (slice(y0 - oy, y1 - oy), slice(x0 - ox, x1 - ox))
    a = alpha[sub][:, :, None]
    s = patch[sub]
    b = out[y0:y1, x0:x1]
    if settings.blend_mode == "screen":
        blended = 1.0 - (1.0 - b) * (1.0 - s)
    elif settings.blend_mode == "normal":
        blended = s
    else:
        blended = np.maximum(b, s)
    out[y0:y1, x0:x1] = b + a * (blended - b)


def render_composite(background: np.ndarray, settings: CompositeSettings,
                     cutouts: Sequence[Optional[SunCutout]], camera: SkyCamera,
                     scale: float = 1.0,
                     should_cancel: Optional[Callable[[], bool]] = None,
                     background_graded: bool = False) -> np.ndarray:
    """
    Paints every enabled partial Sun onto a copy of the background.

    `background` may be a proxy: `scale` is its size relative to the full
    resolution frame that `camera` and all marks are expressed in. The
    background grade is applied first, unless the caller passes a background
    it has already graded (the editor caches that between slider moves).
    """
    if background_graded:
        out = background.astype(np.float32, copy=True)
    else:
        out = apply_grade(background, settings.background_grade)
    placements = compute_placements(settings, camera)
    gains = frame_gains(settings, cutouts)
    sky_sign = _horizon_sky_sign(settings) if settings.horizon is not None else 1.0
    # Paint in time order, so where two Suns overlap the later one is on top.
    order = sorted((p for p in placements if p is not None),
                   key=lambda p: settings.frame_moment(settings.frames[p.index]))
    for place in order:
        if should_cancel is not None and should_cancel():
            break
        cut = cutouts[place.index] if place.index < len(cutouts) else None
        if cut is None:
            continue
        grade = settings.sun_grade.combined(settings.frames[place.index].grade)
        _render_one(out, cut, place, gains[place.index], grade, settings, scale, sky_sign)
    return np.clip(out, 0.0, 1.0)
