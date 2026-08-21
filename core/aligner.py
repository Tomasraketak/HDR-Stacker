"""
Advanced Image Alignment Module for Astronomical and HDR Photography.
Handles composite scenes (Sun in sky + static landscape foreground),
dedicated Sun ROI tracking, landscape tracking, subpixel eclipse disc centering, ECC, ORB, and MTB.
"""

from typing import List, Callable, Optional, Tuple
import cv2
import numpy as np


class ImageAligner:
    """
    Multi-algorithm image aligner tailored for astronomical sequences and HDR bracketing.
    """

    def __init__(
        self,
        method: str = "none",  # "none", "sun_only", "landscape_only", "eclipse_disc", "ecc", "orb", "mtb"
        max_bits: int = 5,
        exclude_range: int = 4,
        cut: bool = False
    ):
        self.method = method.lower()
        self.max_bits = max_bits
        self.exclude_range = exclude_range
        self.cut = cut

    def align(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Aligns a sequence of BGR uint8 images according to the selected strategy.
        """
        if len(images) <= 1 or self.method in ("none", "disabled", "off"):
            if progress_callback:
                progress_callback(100, "Zarovnání vypnuto (zachována původní geometrie scény).")
            return images

        if self.method == "sun_only":
            return self._align_sun_roi(images, progress_callback)
        elif self.method == "landscape_only":
            return self._align_landscape_roi(images, progress_callback)
        elif self.method == "eclipse_disc":
            return self._align_eclipse_disc(images, progress_callback)
        elif self.method == "ecc":
            return self._align_ecc(images, progress_callback)
        elif self.method == "orb":
            return self._align_orb(images, progress_callback)
        else:  # MTB
            return self._align_mtb(images, progress_callback)

    def _find_sun_center(self, gray: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Finds the center of the Sun / eclipse disc in the image.
        Uses Gaussian smoothing and weighted brightest/darkest centroid.
        """
        h, w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (25, 25), 0)
        
        # Check top 75% of the frame (sky area)
        sky_patch = blurred[:int(h * 0.85), :]
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(sky_patch)

        # For partial/solar: brightest area
        # For total eclipse: dark moon surrounded by bright corona ring
        # Find threshold around max_val
        if max_val > 50:
            thresh_val = max(30, int(max_val * 0.6))
            _, binary = cv2.threshold(sky_patch, thresh_val, 255, cv2.THRESH_BINARY)
            M = cv2.moments(binary)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                return float(cx), float(cy)

        # Fallback to max_loc
        return float(max_loc[0]), float(max_loc[1])

    def _align_sun_roi(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Aligns only the Sun / Solar Eclipse area by computing Sun displacement.
        """
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2

        if progress_callback:
            progress_callback(20, "Hledání polohy Slunce na snímcích...")

        sun_positions = []
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            pos = self._find_sun_center(gray)
            sun_positions.append(pos)

        ref_pos = sun_positions[ref_idx]
        if ref_pos is None:
            return images

        ref_cx, ref_cy = ref_pos
        aligned = []

        for i, img in enumerate(images):
            pos = sun_positions[i]
            if pos is not None and i != ref_idx:
                cx, cy = pos
                dx = ref_cx - cx
                dy = ref_cy - cy
                # Limit maximum reasonable drift (e.g. max 10% of image width)
                if np.hypot(dx, dy) < (w * 0.15):
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
                    aligned.append(shifted)
                    continue
            aligned.append(img.copy())

        if progress_callback:
            progress_callback(100, "Zarovnání Slunce dokončeno.")

        return aligned

    def _align_landscape_roi(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Aligns based on the lower portion of the frame (landscape / horizon / foreground).
        """
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2

        if progress_callback:
            progress_callback(20, "Zarovnávání popředí a krajiny...")

        # Crop bottom 50% for landscape alignment
        y_start = int(h * 0.45)
        ref_crop = images[ref_idx][y_start:, :]
        ref_gray = cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=1500)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        aligned = []
        for i, img in enumerate(images):
            if i == ref_idx or des_ref is None:
                aligned.append(img.copy())
                continue

            crop = img[y_start:, :]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            kp_img, des_img = orb.detectAndCompute(gray, None)

            if des_img is not None and len(kp_img) >= 6:
                matches = matcher.knnMatch(des_img, des_ref, k=2)
                good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
                if len(good) >= 6:
                    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                    if M is not None:
                        shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
                        aligned.append(shifted)
                        continue

            aligned.append(img.copy())

        if progress_callback:
            progress_callback(100, "Zarovnání krajiny dokončeno.")

        return aligned

    def _find_disc_center(self, gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        h, w = gray.shape[:2]
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        cy_approx, cx_approx = h // 2, w // 2
        center_patch = gray[max(0, cy_approx - 20):min(h, cy_approx + 20), max(0, cx_approx - 20):min(w, cx_approx + 20)]
        is_moon = float(np.mean(center_patch)) < float(np.mean(gray))
        mask = cv2.bitwise_not(thresh) if is_moon else thresh

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        best_c = None
        min_dist = float('inf')
        min_area = (min(w, h) * 0.08) ** 2 * np.pi

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                dist = np.hypot(cx - cx_approx, cy - cy_approx)
                if dist < min_dist:
                    min_dist = dist
                    best_c = c

        if best_c is not None and len(best_c) >= 5:
            (cx, cy), radius = cv2.minEnclosingCircle(best_c)
            return float(cx), float(cy), float(radius)

        return None

    def _align_eclipse_disc(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2

        if progress_callback:
            progress_callback(15, f"Detekce disku zatmění...")

        centers = [self._find_disc_center(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in images]
        ref_disc = centers[ref_idx]
        if ref_disc is None:
            for d in centers:
                if d is not None:
                    ref_disc = d
                    break

        if ref_disc is None:
            return self._align_mtb(images, progress_callback)

        ref_cx, ref_cy, _ = ref_disc
        aligned = []

        for i, img in enumerate(images):
            disc = centers[i]
            if disc is not None:
                cx, cy, _ = disc
                dx = ref_cx - cx
                dy = ref_cy - cy
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
                aligned.append(shifted)
            else:
                aligned.append(img.copy())

        if progress_callback:
            progress_callback(100, "Zarovnání disku dokončeno.")

        return aligned

    def _align_ecc(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2
        ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
        scale = min(1.0, 1000.0 / max(w, h))
        ref_small = cv2.resize(ref_gray, (int(w * scale), int(h * scale))) if scale < 1.0 else ref_gray

        aligned = []
        warp_mode = cv2.MOTION_EUCLIDEAN
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 35, 1e-3)

        for i, img in enumerate(images):
            if i == ref_idx:
                aligned.append(img.copy())
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(ref_small, gray_small, warp_matrix, warp_mode, criteria, None, 5)
                warp_matrix[0, 2] /= scale
                warp_matrix[1, 2] /= scale
                shifted = cv2.warpAffine(img, warp_matrix, (w, h), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT_101)
                aligned.append(shifted)
            except Exception:
                aligned.append(img.copy())

        if progress_callback:
            progress_callback(100, "ECC zarovnání dokončeno.")
        return aligned

    def _align_orb(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2
        ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        aligned = []

        for i, img in enumerate(images):
            if i == ref_idx or des_ref is None:
                aligned.append(img.copy())
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp_img, des_img = orb.detectAndCompute(gray, None)
            if des_img is None or len(kp_img) < 8:
                aligned.append(img.copy())
                continue

            matches = matcher.knnMatch(des_img, des_ref, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) >= 6:
                src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
                if M is not None:
                    shifted = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
                    aligned.append(shifted)
                    continue

            aligned.append(img.copy())

        if progress_callback:
            progress_callback(100, "ORB zarovnání dokončeno.")
        return aligned

    def _align_mtb(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        try:
            align_mtb = cv2.createAlignMTB(max_bits=self.max_bits, exclude_range=self.exclude_range, cut=self.cut)
            dst = [np.empty_like(img) for img in images]
            align_mtb.process(images, dst)
            if progress_callback:
                progress_callback(100, "MTB zarovnání dokončeno.")
            return dst
        except Exception:
            return images
