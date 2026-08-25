"""
HDR Merging and Exposure Fusion module.
Supports:
1. Mertens Exposure Fusion (Gold standard for solar eclipses and natural HDR)
2. Debevec HDR with Camera Response Calibration & Radiance Map
3. Robertson HDR
4. Tonemapping operators (Reinhard, Drago, Mantiuk)
"""

from typing import List, Callable, Optional, Tuple
import cv2
import numpy as np


class HDRMerger:
    """
    Handles fusion and HDR construction.
    """

    @staticmethod
    def merge_mertens(
        images: List[np.ndarray],
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 0.0,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> np.ndarray:
        """
        Merges images using Mertens multi-scale Exposure Fusion.
        Returns float32 BGR image in range [0.0, 1.0].
        """
        if progress_callback:
            progress_callback(20, "Zahájení Laplaceovy pyramidové fúze (Mertens)...")

        mertens = cv2.createMergeMertens(
            contrast_weight=float(contrast_weight),
            saturation_weight=float(saturation_weight),
            exposure_weight=float(exposure_weight)
        )

        if progress_callback:
            progress_callback(50, "Slučování vrstev expozic...")

        fusion = mertens.process(images)
        
        # Clip to [0, 1] range
        fusion = np.clip(fusion, 0.0, 1.0).astype(np.float32)

        if progress_callback:
            progress_callback(100, "Mertens fúze dokončena.")

        return fusion

    @staticmethod
    def merge_debevec(
        images: List[np.ndarray],
        times: List[float],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Debevec HDR Radiance Map (32-bit linear floating point) and CRF.
        :param images: List of uint8 BGR images
        :param times: List of exposure times in seconds
        :return: (hdr_radiance_map, response_curve)
        """
        times_arr = np.array(times, dtype=np.float32)

        if progress_callback:
            progress_callback(20, "Kalibrace odezvy snímače (Debevec CRF)...")

        # Optimization: Calculate CRF on small images to prevent OOM and speed up
        sample_size = 512
        small_images = []
        for img in images:
            h, w = img.shape[:2]
            if w > sample_size or h > sample_size:
                scale = sample_size / max(w, h)
                small_images.append(cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
            else:
                small_images.append(img)

        calibrate = cv2.createCalibrateDebevec()
        crf = calibrate.process(small_images, times_arr)

        if progress_callback:
            progress_callback(60, "Výpočet 32-bit lineární mapy jasu (Radiance Map)...")

        merge_deb = cv2.createMergeDebevec()
        hdr_map = merge_deb.process(images, times_arr, crf)

        if progress_callback:
            progress_callback(100, "Debevec HDR dokončeno.")

        return hdr_map, crf

    @staticmethod
    def merge_robertson(
        images: List[np.ndarray],
        times: List[float],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Robertson HDR Radiance Map and CRF.
        """
        times_arr = np.array(times, dtype=np.float32)

        if progress_callback:
            progress_callback(20, "Kalibrace odezvy snímače (Robertson CRF)...")

        sample_size = 512
        small_images = []
        for img in images:
            h, w = img.shape[:2]
            if w > sample_size or h > sample_size:
                scale = sample_size / max(w, h)
                small_images.append(cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
            else:
                small_images.append(img)

        calibrate = cv2.createCalibrateRobertson()
        crf = calibrate.process(small_images, times_arr)

        if progress_callback:
            progress_callback(60, "Výpočet Robertson HDR mapy...")

        merge_rob = cv2.createMergeRobertson()
        hdr_map = merge_rob.process(images, times_arr, crf)

        if progress_callback:
            progress_callback(100, "Robertson HDR dokončeno.")

        return hdr_map, crf

    @staticmethod
    def tonemap(
        hdr_map: np.ndarray,
        method: str = "reinhard",
        gamma: float = 1.0,
        intensity: float = 0.0,
        light_adapt: float = 0.8,
        color_adapt: float = 0.0,
        drago_bias: float = 0.85,
        mantiuk_scale: float = 0.7,
        mantiuk_saturation: float = 1.0
    ) -> np.ndarray:
        """
        Tonemaps a 32-bit floating point HDR image to standard dynamic range [0, 1].
        """
        method = method.lower()
        if method == "reinhard":
            tonemapper = cv2.createTonemapReinhard(
                gamma=float(gamma),
                intensity=float(intensity),
                light_adapt=float(light_adapt),
                color_adapt=float(color_adapt)
            )
        elif method == "drago":
            tonemapper = cv2.createTonemapDrago(
                gamma=float(gamma),
                saturation=float(mantiuk_saturation),
                bias=float(drago_bias)
            )
        elif method == "mantiuk":
            tonemapper = cv2.createTonemapMantiuk(
                gamma=float(gamma),
                scale=float(mantiuk_scale),
                saturation=float(mantiuk_saturation)
            )
        else:
            tonemapper = cv2.createTonemap(gamma=float(gamma))

        ldr = tonemapper.process(hdr_map)
        return np.clip(ldr, 0.0, 1.0).astype(np.float32)
