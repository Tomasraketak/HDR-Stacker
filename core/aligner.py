"""
Advanced Image Alignment Module for Astronomical and HDR Photography.
Supports:
1. Eclipse Disc Center Alignment (Subpixel Lunar/Solar silhouette alignment - Optimal for total/partial solar eclipses)
2. ECC Alignment (Enhanced Correlation Coefficient - Subpixel Euclidean/Affine)
3. ORB / SIFT Feature Alignment with RANSAC
4. MTB (Median Threshold Bitmap)
5. No Alignment (for precise tracking mounts)
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
        method: str = "eclipse_disc",  # "eclipse_disc", "ecc", "orb", "mtb", "none"
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
        Aligns a sequence of BGR uint8 images to a reference frame.
        """
        if len(images) <= 1 or self.method == "none":
            if progress_callback:
                progress_callback(100, "Zarovnání přeskočeno (režim bez zarovnání).")
            return images

        if self.method == "eclipse_disc":
            return self._align_eclipse_disc(images, progress_callback)
        elif self.method == "ecc":
            return self._align_ecc(images, progress_callback)
        elif self.method == "orb":
            return self._align_orb(images, progress_callback)
        else:  # default MTB
            return self._align_mtb(images, progress_callback)

    def _find_disc_center(self, gray: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Finds the center (cx, cy) and radius r of the lunar or solar disc in an image.
        Uses adaptive thresholding, morphological filtering, and contour/ellipse fitting.
        """
        h, w = gray.shape[:2]
        
        # Blur to eliminate high frequency noise
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        
        # 1. Try Otsu thresholding or adaptive threshold
        # For total eclipse: center is dark (Moon), corona is bright.
        # For partial eclipse: sun is bright, moon is dark bite.
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if the center area is darker than surrounding (lunar silhouette)
        cy_approx, cx_approx = h // 2, w // 2
        center_patch = gray[max(0, cy_approx - 20):min(h, cy_approx + 20), max(0, cx_approx - 20):min(w, cx_approx + 20)]
        is_moon_silhouette = float(np.mean(center_patch)) < float(np.mean(gray))
        
        if is_moon_silhouette:
            # Moon silhouette is dark surrounded by bright corona
            mask = cv2.bitwise_not(thresh)
        else:
            mask = thresh

        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        # Pick contour closest to image center with reasonable area
        best_c = None
        min_dist_to_center = float('inf')
        min_area = (min(w, h) * 0.08) ** 2 * np.pi  # At least 8% diameter

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                dist = np.hypot(cx - cx_approx, cy - cy_approx)
                if dist < min_dist_to_center:
                    min_dist_to_center = dist
                    best_c = c

        if best_c is not None and len(best_c) >= 5:
            # Fit enclosing circle / ellipse for subpixel accuracy
            (cx, cy), radius = cv2.minEnclosingCircle(best_c)
            return float(cx), float(cy), float(radius)

        # Fallback: Hough Circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(w, h) // 4,
            param1=100,
            param2=30,
            minRadius=int(min(w, h) * 0.05),
            maxRadius=int(min(w, h) * 0.48)
        )
        if circles is not None and len(circles) > 0:
            c = circles[0][0]
            return float(c[0]), float(c[1]), float(c[2])

        return None

    def _align_eclipse_disc(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Aligns images by detecting the subpixel center of the Moon/Sun silhouette.
        Extremely robust across different exposure levels!
        """
        n = len(images)
        h, w = images[0].shape[:2]
        
        # Reference is median exposure frame
        ref_idx = n // 2
        
        if progress_callback:
            progress_callback(15, f"Detekce astronomického středu zatmění (referenční snímek {ref_idx+1})...")

        centers = []
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            disc = self._find_disc_center(gray)
            centers.append(disc)

        ref_disc = centers[ref_idx]
        if ref_disc is None:
            # Try to find first valid reference disc
            for i, disc in enumerate(centers):
                if disc is not None:
                    ref_idx = i
                    ref_disc = disc
                    break

        if ref_disc is None:
            # If disc center detection fails completely, fallback to MTB
            if progress_callback:
                progress_callback(30, "Detekce disku nenalezena, přecházím na MTB zarovnání...")
            return self._align_mtb(images, progress_callback)

        ref_cx, ref_cy, _ = ref_disc
        aligned_images = []

        for i, img in enumerate(images):
            pct = int(30 + (i / n) * 65)
            if progress_callback:
                progress_callback(pct, f"Subpixelové zarovnávání snímku {i+1}/{n}...")

            disc = centers[i]
            if disc is not None:
                cx, cy, _ = disc
                dx = ref_cx - cx
                dy = ref_cy - cy
                
                # Subpixel translation matrix
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(
                    img, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                aligned_images.append(shifted)
            else:
                # If single frame failed disc detection, keep original
                aligned_images.append(img.copy())

        if progress_callback:
            progress_callback(100, "Astronomické zarovnání disku zatmění dokončeno.")

        return aligned_images

    def _align_ecc(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Enhanced Correlation Coefficient (ECC) alignment with subpixel accuracy.
        """
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2
        ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)
        
        # Downscale for fast estimation
        scale = min(1.0, 1200.0 / max(w, h))
        ref_small = cv2.resize(ref_gray, (int(w * scale), int(h * scale))) if scale < 1.0 else ref_gray

        aligned = []
        warp_mode = cv2.MOTION_EUCLIDEAN
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)

        for i, img in enumerate(images):
            pct = int(15 + (i / n) * 80)
            if progress_callback:
                progress_callback(pct, f"Výpočet ECC transformace pro snímek {i+1}/{n}...")

            if i == ref_idx:
                aligned.append(img.copy())
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (int(w * scale), int(h * scale))) if scale < 1.0 else gray
            
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(ref_small, gray_small, warp_matrix, warp_mode, criteria, None, 5)
                
                # Scale warp matrix back to original resolution
                warp_matrix[0, 2] /= scale
                warp_matrix[1, 2] /= scale
                
                shifted = cv2.warpAffine(
                    img, warp_matrix, (w, h),
                    flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REFLECT_101
                )
                aligned.append(shifted)
            except Exception as e:
                # If ECC fails on extreme exposure, fallback to MTB for this frame
                try:
                    align_mtb = cv2.createAlignMTB(max_bits=self.max_bits, exclude_range=self.exclude_range, cut=False)
                    shift = align_mtb.calculateShift(ref_gray, gray)
                    shifted = align_mtb.shiftMat(img, shift)
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
        """
        ORB Feature Matching alignment with RANSAC.
        """
        n = len(images)
        h, w = images[0].shape[:2]
        ref_idx = n // 2
        ref_gray = cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=2000)
        kp_ref, des_ref = orb.detectAndCompute(ref_gray, None)

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        aligned = []

        for i, img in enumerate(images):
            pct = int(15 + (i / n) * 80)
            if progress_callback:
                progress_callback(pct, f"Detekce a párování ORB bodů {i+1}/{n}...")

            if i == ref_idx or des_ref is None:
                aligned.append(img.copy())
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp_img, des_img = orb.detectAndCompute(gray, None)

            if des_img is None or len(kp_img) < 8:
                aligned.append(img.copy())
                continue

            matches = matcher.knnMatch(des_img, des_ref, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2 and m_n[0].distance < 0.75 * m_n[1].distance:
                    good_matches.append(m_n[0])

            if len(good_matches) >= 6:
                src_pts = np.float32([kp_img[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC)
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
        """
        Median Threshold Bitmap alignment.
        """
        if progress_callback:
            progress_callback(15, "Inicializace MTB zarovnání...")

        try:
            align_mtb = cv2.createAlignMTB(
                max_bits=self.max_bits,
                exclude_range=self.exclude_range,
                cut=self.cut
            )
            dst = [np.empty_like(img) for img in images]
            align_mtb.process(images, dst)

            if progress_callback:
                progress_callback(100, "MTB zarovnání dokončeno.")

            return dst
        except Exception as e:
            try:
                align_mtb = cv2.createAlignMTB(max_bits=self.max_bits, exclude_range=self.exclude_range, cut=False)
                ref_idx = len(images) // 2
                ref_img = images[ref_idx]
                aligned = []
                for idx, img in enumerate(images):
                    if idx == ref_idx:
                        aligned.append(img.copy())
                    else:
                        shift = align_mtb.calculateShift(ref_img, img)
                        shifted = align_mtb.shiftMat(img, shift)
                        aligned.append(shifted)
                if progress_callback:
                    progress_callback(100, "MTB zarovnání dokončeno.")
                return aligned
            except Exception:
                return images
