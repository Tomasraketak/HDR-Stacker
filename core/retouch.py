"""
Retouching: removes unwanted things — grass blades in front of the lens, a
bird, a dust spot — from smooth areas such as the sky, so that it looks as if
they had never been there.

The user paints over the object and the painted area is refilled in two layers:

  * the light of the sky: a membrane stretched over the area's surroundings
    (a harmonic fill), which continues every gradient — the glow above the
    horizon included — without a seam;
  * the grain: the fine noise of a nearby clean patch of the same image is
    copied in, so the fill is not suspiciously smooth.

Strokes are vector paths in full-resolution pixels, so the preview (a proxy)
and the export (full size) apply the same retouch at their own resolution.
Each stroke heals the image as the previous strokes left it, in order.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Jacobi sweeps per pyramid level of the membrane fill; each level starts from
# the coarser level's solution, so a few dozen sweeps converge.
MEMBRANE_SWEEPS = 50

_NEIGHBOURS = np.array([[0.0, 0.25, 0.0], [0.25, 0.0, 0.25], [0.0, 0.25, 0.0]], np.float32)


@dataclass
class RetouchStroke:
    """One brush stroke: a path and the brush radius, in full-resolution pixels."""
    points: List[Tuple[float, float]]
    radius: float

    def to_dict(self) -> dict:
        return {"points": [[round(float(x), 2), round(float(y), 2)] for x, y in self.points],
                "radius": round(float(self.radius), 2)}

    @classmethod
    def from_dict(cls, data: Any) -> Optional["RetouchStroke"]:
        if isinstance(data, RetouchStroke):
            return data
        if not isinstance(data, dict):
            return None
        try:
            points = [(float(p[0]), float(p[1])) for p in data.get("points") or []]
            radius = float(data.get("radius", 0.0))
        except (TypeError, ValueError, IndexError):
            return None
        if not points or not (radius > 0 and math.isfinite(radius)):
            return None
        if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
            return None
        return cls(points=points, radius=radius)


def count_strokes(n: int) -> str:
    """'1 tah', '3 tahy', '5 tahů' — for status messages."""
    if n == 1:
        return "1 tah"
    return f"{n} tahy" if 2 <= n <= 4 else f"{n} tahů"


def strokes_to_data(strokes: Iterable[RetouchStroke]) -> List[dict]:
    return [s.to_dict() for s in strokes]


def strokes_from_data(data: Any) -> List[RetouchStroke]:
    """Strokes from a project file; anything malformed is skipped, not fatal."""
    strokes = []
    for raw in data if isinstance(data, (list, tuple)) else []:
        stroke = RetouchStroke.from_dict(raw)
        if stroke is not None:
            strokes.append(stroke)
    return strokes


# ------------------------------------------------------------------- Mask

def stroke_mask(stroke: RetouchStroke, shape: Tuple[int, ...], scale: float = 1.0,
                offset: Tuple[float, float] = (0.0, 0.0)
                ) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    The painted area in an image of `shape` whose pixel (0, 0) sits at
    `offset` in stroke coordinates and which is `scale` times their size
    (a proxy, a crop or an ROI patch of the full-resolution frame).

    Returns (bool mask, (x0, y0, x1, y1)) with the mask cropped to that box,
    or None when the stroke misses the image.
    """
    h, w = shape[:2]
    ox, oy = offset
    # Pixel centres sit at whole coordinates at every scale.
    pts = np.array([((x - ox + 0.5) * scale - 0.5, (y - oy + 0.5) * scale - 0.5)
                    for x, y in stroke.points], np.float64)
    radius = max(0.75, float(stroke.radius) * scale)
    pad = int(math.ceil(radius)) + 2
    x0 = max(0, int(math.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(math.floor(pts[:, 1].min())) - pad)
    x1 = min(w, int(math.ceil(pts[:, 0].max())) + pad + 1)
    y1 = min(h, int(math.ceil(pts[:, 1].max())) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
    # Sub-pixel drawing: OpenCV takes coordinates in 1/8 px with shift=3.
    fixed = np.round((pts - [x0, y0]) * 8.0).astype(np.int64)
    r8 = int(round(radius * 8.0))
    thickness = max(1, int(round(2.0 * radius)))
    for i, (px, py) in enumerate(fixed):
        cv2.circle(mask, (int(px), int(py)), r8, 1, -1, cv2.LINE_8, 3)
        if i:
            qx, qy = fixed[i - 1]
            cv2.line(mask, (int(qx), int(qy)), (int(px), int(py)), 1, thickness, cv2.LINE_8, 3)
    if not mask.any():
        return None
    return mask.astype(bool), (x0, y0, x1, y1)


# ------------------------------------------------------------------- Fill

def _membrane_fill(img: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """
    Harmonic fill: every unknown pixel ends up as the average of its four
    neighbours, the known pixels staying fixed — a membrane stretched over
    the hole's rim. Solved coarse to fine, so a hole of any width converges.
    """
    h, w = unknown.shape
    if min(h, w) > 12:
        known = (~unknown).astype(np.float32)
        weight = cv2.pyrDown(known)
        coarse = cv2.pyrDown(img * known[:, :, None]) / np.maximum(weight, 1e-6)[:, :, None]
        coarse = _membrane_fill(coarse, weight < 0.5)
        start = cv2.pyrUp(coarse, dstsize=(w, h))
    else:
        rim = img[~unknown]
        start = np.broadcast_to(rim.mean(axis=0) if rim.size else np.zeros(3, np.float32),
                                img.shape)

    out = img.copy()
    out[unknown] = start[unknown]
    for _ in range(MEMBRANE_SWEEPS):
        smooth = cv2.filter2D(out, -1, _NEIGHBOURS, borderType=cv2.BORDER_REPLICATE)
        out[unknown] = smooth[unknown]
    return out


def _grain_offsets(unknown: np.ndarray, low: np.ndarray, thickness: float
                   ) -> List[Tuple[int, int]]:
    """
    Offsets to nearby clean areas the grain can be copied from, best first:
    the whole hole shifted by them should land on known pixels of a similar
    brightness.
    """
    h, w = unknown.shape
    ys, xs = np.nonzero(unknown)
    if ys.size > 4000:
        pick = np.linspace(0, ys.size - 1, 4000).astype(np.int64)
        ys, xs = ys[pick], xs[pick]
    target = low[ys, xs].mean(axis=0)
    scored = []
    for distance in (1.3, 2.0, 3.0):
        step = max(3.0, distance * thickness)
        for k in range(16):
            angle = 2.0 * math.pi * k / 16.0
            dx, dy = int(round(step * math.cos(angle))), int(round(step * math.sin(angle)))
            sy, sx = ys + dy, xs + dx
            inside = (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
            usable = np.zeros_like(inside)
            usable[inside] = ~unknown[sy[inside], sx[inside]]
            coverage = float(usable.mean())
            if coverage < 0.5:
                continue
            tone = float(np.abs(low[sy[usable], sx[usable]].mean(axis=0) - target).sum())
            scored.append((-coverage, tone + 0.001 * step, (dx, dy)))
    scored.sort()
    return [offset for _c, _t, offset in scored[:4]]


def _heal_region(region: np.ndarray, unknown: np.ndarray) -> np.ndarray:
    """The healed pixel values of `region` (float32 BGR) inside `unknown`."""
    filled = _membrane_fill(region, unknown)

    # Grain: the image minus its local mean, taken from beside the hole. The
    # blur runs on the membrane-filled image, so the removed object cannot
    # leak into the grain along the hole's rim.
    thickness = 2.0 * float(cv2.distanceTransform(unknown.astype(np.uint8), cv2.DIST_L2, 3).max())
    low = cv2.GaussianBlur(filled, (0, 0), 2.5)
    grain = filled - low
    ys, xs = np.nonzero(unknown)
    copied = np.zeros((ys.size, 3), np.float32)
    pending = np.ones(ys.size, bool)
    h, w = unknown.shape
    for dx, dy in _grain_offsets(unknown, low, max(thickness, 2.0)):
        if not pending.any():
            break
        sy, sx = ys + dy, xs + dx
        ok = pending & (sy >= 0) & (sy < h) & (sx >= 0) & (sx < w)
        ok[ok] = ~unknown[sy[ok], sx[ok]]
        copied[ok] = grain[sy[ok], sx[ok]]
        pending &= ~ok
    return np.clip(filled[ys, xs] + copied, 0.0, 1.0)


# The healed area reaches this far beyond the brush (in brush radii) and fades
# out there: an out-of-focus object's soft edge extends past where the brush
# was aimed, and a fill anchored on that darkened rim would leave a ghost.
FEATHER_RADII = 1.0


def heal_stroke(image: np.ndarray, stroke: RetouchStroke, scale: float = 1.0,
                offset: Tuple[float, float] = (0.0, 0.0), feather: float = FEATHER_RADII
                ) -> Optional[Tuple[Tuple[int, int, int, int], np.ndarray]]:
    """
    Heals one stroke in `image` (float32 BGR in [0, 1], modified in place).

    Returns the box it touched and that box's pixels as they were before, so
    a caller can undo it; None when the stroke misses the image.
    """
    painted = stroke_mask(stroke, image.shape, scale, offset)
    if painted is None:
        return None
    mask, (x0, y0, x1, y1) = painted
    h, w = image.shape[:2]
    radius = max(0.75, stroke.radius * scale)
    soft = max(1.5, feather * radius)
    # Context around the hole: the membrane needs its rim, the grain a clean
    # patch a few brush widths away.
    margin = int(math.ceil(7.0 * radius + soft)) + 8
    bx0, by0 = max(0, x0 - margin), max(0, y0 - margin)
    bx1, by1 = min(w, x1 + margin), min(h, y1 + margin)
    core = np.zeros((by1 - by0, bx1 - bx0), bool)
    core[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0] = mask
    # Distance of every pixel from the painted area; the fill covers a soft
    # ring around it too, and fades out across that ring.
    distance = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 3)
    unknown = distance <= soft
    if unknown.all():
        return None   # nothing left to heal from

    region = image[by0:by1, bx0:bx1]
    before = region.copy()
    healed = _heal_region(np.ascontiguousarray(region, np.float32), unknown)
    t = np.clip(1.0 - distance[unknown] / soft, 0.0, 1.0)
    alpha = (t * t * (3.0 - 2.0 * t))[:, None]          # smoothstep
    region[unknown] = alpha * healed + (1.0 - alpha) * region[unknown]
    return (bx0, by0, bx1, by1), before


def apply_retouch(image: np.ndarray, strokes: Sequence[RetouchStroke], scale: float = 1.0,
                  offset: Tuple[float, float] = (0.0, 0.0),
                  should_cancel: Optional[Callable[[], bool]] = None,
                  copy: bool = True) -> np.ndarray:
    """
    `image` as float32 with every stroke healed, in order. With copy=False a
    float32 image is healed in place — a full-resolution export has no memory
    to spare for a second copy.
    """
    if copy or image.dtype != np.float32:
        out = np.array(image, dtype=np.float32, copy=True)
    else:
        out = image
    for stroke in strokes:
        if should_cancel is not None and should_cancel():
            break
        heal_stroke(out, stroke, scale, offset)
    return out


class RetouchCache:
    """
    A preview image with its strokes applied, redoing only what changed: a new
    stroke heals just its own neighbourhood, and undo puts back the pixels the
    stroke replaced instead of starting over.
    """

    def __init__(self):
        self._source: Optional[np.ndarray] = None
        self._placement: Optional[Tuple[float, float, float]] = None
        self._image: Optional[np.ndarray] = None
        self._applied: List[Tuple[RetouchStroke, Optional[Tuple[Tuple[int, int, int, int],
                                                              np.ndarray]]]] = []
        self.version = 0     # changes whenever the result does

    def result(self, source: np.ndarray, strokes: Sequence[RetouchStroke], scale: float = 1.0,
               offset: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        placement = (float(scale), float(offset[0]), float(offset[1]))
        if source is not self._source or placement != self._placement or self._image is None:
            self._source, self._placement = source, placement
            self._image = np.array(source, dtype=np.float32, copy=True)
            self._applied = []
            self.version += 1

        common = 0
        while (common < min(len(self._applied), len(strokes))
               and self._applied[common][0] == strokes[common]):
            common += 1
        while len(self._applied) > common:
            _stroke, undo = self._applied.pop()
            if undo is not None:
                (x0, y0, x1, y1), before = undo
                self._image[y0:y1, x0:x1] = before
            self.version += 1
        for stroke in strokes[common:]:
            self._applied.append((stroke, heal_stroke(self._image, stroke, scale, offset)))
            self.version += 1
        return self._image

    def clear(self):
        self._source = self._placement = self._image = None
        self._applied = []
        self.version += 1
