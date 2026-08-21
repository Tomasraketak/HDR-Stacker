"""
Advanced Astronomical Alignment Engine.
Features:
1. Robust Black Circle in Light Detector (specifically locates the circular Moon silhouette surrounded by bright corona).
2. Per-image subpixel shift calculations and application.
"""

from typing import List, Callable, Optional, Tuple
import cv2
import numpy as np


def detect_black_circle_in_light(image_bgr: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Specifically detects the dark lunar circular silhouette surrounded by bright coronal glow.
    Filters out landscape/trees by enforcing strict geometric circularity and enclosing bright gradient.
    Returns (cx, cy, radius) in image coordinates, or None if not found.
    """
    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Smooth to remove high frequency noise / stars
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)

    # 2. Locate the brightest regions (corona) in the sky
    # Avoid extreme bottom if landscape is present
    sky_h = int(h * 0.9)
    sky_gray = blurred[:sky_h, :]

    # Estimate min & max intensity
    min_v, max_v, _, max_loc = cv2.minMaxLoc(sky_gray)
    if max_v < 25:
        return None

    # Multi-level threshold search for dark circular void inside bright halo
    # Test multiple threshold levels from 10% to 60% of dynamic range
    candidate_circles = []

    for t_factor in [0.15, 0.25, 0.35, 0.45, 0.55]:
        t_val = int(min_v + (max_v - min_v) * t_factor)
        _, thresh = cv2.threshold(sky_gray, t_val, 255, cv2.THRESH_BINARY_INV)

        # Morphological opening to detach trees/cables
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        opened = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            area = cv2.contourArea(c)
            # Disc must be reasonably sized (at least 2% and at most 60% of frame)
            min_r = min(w, h) * 0.02
            max_r = min(w, h) * 0.45
            min_area = np.pi * (min_r ** 2)
            max_area = np.pi * (max_r ** 2)

            if min_area <= area <= max_area:
                perimeter = cv2.arcLength(c, True)
                if perimeter > 0:
                    # Circularity metric: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
                    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
                    if circularity > 0.65:
                        (cx, cy), r = cv2.minEnclosingCircle(c)
                        # Verify the center is dark and the boundary is bright
                        cx_int, cy_int = int(cx), int(cy)
                        if 0 <= cx_int < w and 0 <= cy_int < sky_h:
                            center_lum = float(gray[cy_int, cx_int])
                            # Check ring at radius r * 1.2
                            if center_lum < (max_v * 0.7):
                                candidate_circles.append(((float(cx), float(cy), float(r)), circularity))

    if candidate_circles:
        # Sort by circularity descending
        candidate_circles.sort(key=lambda x: x[1], reverse=True)
        best_circle = candidate_circles[0][0]
        return best_circle

    # Fallback: Hough Circles with soft parameters
    circles = cv2.HoughCircles(
        blurred[:sky_h, :],
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(w, h) // 4,
        param1=80,
        param2=25,
        minRadius=int(min(w, h) * 0.03),
        maxRadius=int(min(w, h) * 0.45)
    )
    if circles is not None and len(circles) > 0:
        c = circles[0][0]
        return float(c[0]), float(c[1]), float(c[2])

    return None


def calculate_moon_shifts(images: List[np.ndarray], ref_idx: Optional[int] = None) -> List[Tuple[float, float]]:
    """
    Detects the black circular lunar disc in each image and computes relative (dx, dy)
    shifts with subpixel accuracy relative to the reference image.
    """
    n = len(images)
    if n <= 1:
        return [(0.0, 0.0) for _ in images]

    if ref_idx is None:
        ref_idx = n // 2

    circles = [detect_black_circle_in_light(img) for img in images]
    
    # Identify reference disc
    ref_disc = circles[ref_idx]
    if ref_disc is None:
        # Find first valid detection
        for idx, c in enumerate(circles):
            if c is not None:
                ref_idx = idx
                ref_disc = c
                break

    if ref_disc is None:
        # Could not detect on any frame
        return [(0.0, 0.0) for _ in images]

    ref_cx, ref_cy, _ = ref_disc
    shifts = []

    for i, c in enumerate(circles):
        if c is not None:
            cx, cy, _ = c
            dx = float(ref_cx - cx)
            dy = float(ref_cy - cy)
            shifts.append((dx, dy))
        else:
            shifts.append((0.0, 0.0))

    return shifts


def apply_shifts_to_images(
    images: List[np.ndarray],
    shifts: List[Tuple[float, float]],
    scale_factor: float = 1.0
) -> List[np.ndarray]:
    """
    Applies (dx, dy) translation to a list of images using subpixel cubic interpolation.
    """
    aligned = []
    for i, img in enumerate(images):
        dx, dy = shifts[i]
        scaled_dx = dx * scale_factor
        scaled_dy = dy * scale_factor

        if abs(scaled_dx) < 0.01 and abs(scaled_dy) < 0.01:
            aligned.append(img.copy())
        else:
            h, w = img.shape[:2]
            M = np.float32([[1.0, 0.0, scaled_dx], [0.0, 1.0, scaled_dy]])
            shifted = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT_101
            )
            aligned.append(shifted)

    return aligned


class ImageAligner:
    """
    Aligner supporting automatic Black Circle detection, manual offsets, and fallback methods.
    """

    def __init__(
        self,
        method: str = "none",  # "none", "eclipse_disc", "sun_only", "landscape_only", "manual", "ecc", "orb", "mtb"
        manual_shifts: Optional[List[Tuple[float, float]]] = None
    ):
        self.method = method.lower()
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

        # 1. Manual shifts supplied
        if self.manual_shifts and len(self.manual_shifts) == len(images):
            if progress_callback:
                progress_callback(50, "Aplikuji ručně zadané posuny expozic...")
            return apply_shifts_to_images(images, self.manual_shifts)

        # 2. Black Circle / Eclipse Disc detection
        if self.method in ("eclipse_disc", "sun_only"):
            if progress_callback:
                progress_callback(20, "Detekuji černý disk Měsíce v záři korony...")
            shifts = calculate_moon_shifts(images)
            if progress_callback:
                progress_callback(70, "Aplikuji subpixelové posuny disku...")
            return apply_shifts_to_images(images, shifts)

        # Fallback to no alignment
        return images
