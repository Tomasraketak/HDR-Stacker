"""
HDR merging and exposure fusion.

Supports:
1. Mertens exposure fusion (the gold standard for solar eclipses and natural HDR),
   with an optional band-tiled path that keeps peak memory bounded on
   full-resolution 24-45 MP brackets.
2. Debevec HDR with camera response calibration and a linear radiance map.
3. Robertson HDR.
4. Tonemapping operators (Reinhard, Drago, Mantiuk).

Every entry point validates its inputs and scrubs non-finite values, so a bad
bracket produces a clear error message instead of a NaN image or a hard crash.
"""

from typing import List, Callable, Optional, Tuple
import cv2
import numpy as np

# CRF calibration is done on downscaled copies: the response curve is a global
# property of the sensor, so 512 px samples are plenty and avoid multi-GB peaks.
CRF_SAMPLE_SIZE = 512

# Full-resolution fusion above this many megapixels is processed in horizontal
# bands to keep peak RAM bounded (important on 8-16 GB laptops).
TILED_FUSION_MEGAPIXELS = 12.0

# Overlap between bands, in pixels. Must comfortably exceed the support of the
# Laplacian pyramid used by Mertens so the seams are invisible.
BAND_OVERLAP = 256


class HDRMergeError(ValueError):
    """Raised when a bracket cannot be merged (bad sizes, unusable exposure times)."""


def _validate_stack(images: List[np.ndarray]) -> List[np.ndarray]:
    """Ensures a non-empty list of equally sized 8-bit BGR images."""
    if not images:
        raise HDRMergeError("Nebyly předány žádné snímky ke složení.")

    out: List[np.ndarray] = []
    h0, w0 = images[0].shape[:2]
    for img in images:
        if img is None or img.size == 0:
            raise HDRMergeError("Jeden ze snímků je prázdný nebo se nepodařilo načíst.")
        if img.shape[:2] != (h0, w0):
            raise HDRMergeError(
                f"Snímky mají různé rozměry ({img.shape[1]}x{img.shape[0]} vs {w0}x{h0}). "
                "Použijte snímky ze stejného fotoaparátu a se stejným rozlišením."
            )
        if img.dtype != np.uint8:
            img = np.clip(np.nan_to_num(img.astype(np.float32)), 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out.append(np.ascontiguousarray(img))
    return out


def sanitize_exposure_times(times: List[float], count: int) -> np.ndarray:
    """
    Turns a user/EXIF-supplied list of exposure times into something the
    Debevec/Robertson solvers can actually use.

    Zero, negative and non-finite values are the classic cause of an all-NaN
    radiance map, and identical times make the response curve singular — both
    are repaired here (or rejected with an explanatory message).
    """
    arr = np.asarray(list(times) + [0.0] * max(0, count - len(times)), dtype=np.float64)[:count]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    valid = arr[arr > 0.0]
    if valid.size == 0:
        raise HDRMergeError(
            "Žádný snímek nemá použitelný expoziční čas. Debevec a Robertson HDR "
            "potřebují znát časy závěrky — načtěte snímky s EXIF, nebo použijte "
            "metodu Mertens Exposure Fusion, která časy nepotřebuje."
        )

    # Fill in gaps with a plausible geometric progression around the known times.
    if valid.size < arr.size:
        median = float(np.median(valid))
        for i in range(arr.size):
            if arr[i] <= 0.0:
                arr[i] = median * (2.0 ** (i - arr.size / 2.0))

    if float(np.max(arr)) / float(np.min(arr)) < 1.05:
        raise HDRMergeError(
            "Všechny snímky mají prakticky stejný expoziční čas, takže z nich nelze "
            "sestavit HDR. Zkontrolujte EV řadu, nebo použijte Mertens Exposure Fusion."
        )

    # Break exact ties, which make the least-squares system singular.
    order = np.argsort(arr)
    for pos in range(1, order.size):
        i, prev = order[pos], order[pos - 1]
        if arr[i] <= arr[prev]:
            arr[i] = arr[prev] * 1.001

    return arr.astype(np.float32)


def _downscale_for_crf(images: List[np.ndarray], sample_size: int = CRF_SAMPLE_SIZE) -> List[np.ndarray]:
    small = []
    for img in images:
        h, w = img.shape[:2]
        if max(w, h) > sample_size:
            scale = sample_size / float(max(w, h))
            small.append(cv2.resize(img, (max(8, int(w * scale)), max(8, int(h * scale))),
                                    interpolation=cv2.INTER_AREA))
        else:
            small.append(img)
    return small


def _finite(arr: np.ndarray, hi: Optional[float] = None) -> np.ndarray:
    """Replaces NaN/Inf with finite values; optionally clips to [0, hi]."""
    out = np.nan_to_num(arr, nan=0.0, posinf=(hi if hi is not None else 1e6), neginf=0.0)
    if hi is not None:
        out = np.clip(out, 0.0, hi)
    return out.astype(np.float32)


class HDRMerger:
    """Handles exposure fusion and HDR radiance-map construction."""

    # ------------------------------------------------------------------ Mertens

    @staticmethod
    def merge_mertens(
        images: List[np.ndarray],
        contrast_weight: float = 1.0,
        saturation_weight: float = 1.0,
        exposure_weight: float = 1.0,
        progress_callback: Optional[Callable[[int, str], None]] = None,
        allow_tiling: bool = True,
    ) -> np.ndarray:
        """
        Merges a bracket with Mertens multi-scale exposure fusion.

        Returns a float32 BGR image in [0, 1]. Large stacks are fused in
        horizontal bands with a generous overlap so peak memory stays bounded
        while the result remains seamless.
        """
        imgs = _validate_stack(images)

        # All three weights at zero makes every pixel weight zero -> division by
        # zero inside OpenCV and a black frame. Keep at least contrast alive.
        cw, sw, ew = float(contrast_weight), float(saturation_weight), float(exposure_weight)
        if cw <= 0.0 and sw <= 0.0 and ew <= 0.0:
            cw = 1.0

        h, w = imgs[0].shape[:2]
        megapixels = (h * w) / 1e6

        if allow_tiling and megapixels > TILED_FUSION_MEGAPIXELS and h > 4 * BAND_OVERLAP:
            return HDRMerger._merge_mertens_banded(imgs, cw, sw, ew, progress_callback)

        if progress_callback:
            progress_callback(20, "Zahájení Laplaceovy pyramidové fúze (Mertens)...")

        fusion = HDRMerger._mertens_once(imgs, cw, sw, ew)

        if progress_callback:
            progress_callback(100, "Mertens fúze dokončena.")
        return fusion

    @staticmethod
    def _mertens_once(images: List[np.ndarray], cw: float, sw: float, ew: float) -> np.ndarray:
        try:
            mertens = cv2.createMergeMertens(
                contrast_weight=cw, saturation_weight=sw, exposure_weight=ew
            )
            fusion = mertens.process(images)
        except cv2.error as e:
            raise HDRMergeError(f"OpenCV nedokázal provést Mertens fúzi: {e}") from e
        return _finite(fusion, hi=1.0)

    @staticmethod
    def _merge_mertens_banded(
        images: List[np.ndarray],
        cw: float, sw: float, ew: float,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> np.ndarray:
        """
        Fuses the stack in overlapping horizontal bands and cross-fades the
        overlaps, so a 45 MP bracket never needs the whole stack in float32 at once.
        """
        h, w = images[0].shape[:2]
        n = len(images)

        # Choose a band height that keeps one band's float32 working set modest.
        # Roughly: n * band_h * w * 3 channels * 4 bytes * ~6 pyramid copies.
        budget_bytes = 700 * 1024 * 1024
        band_h = int(budget_bytes / max(1, n * w * 3 * 4 * 6))
        band_h = int(np.clip(band_h, 4 * BAND_OVERLAP, h))

        result = np.zeros((h, w, 3), dtype=np.float32)
        weight = np.zeros((h, 1), dtype=np.float32)

        y = 0
        band_index = 0
        total_bands = max(1, int(np.ceil(h / max(1, band_h - BAND_OVERLAP))))
        while y < h:
            y0 = max(0, y - (BAND_OVERLAP if y > 0 else 0))
            y1 = min(h, y0 + band_h)
            if h - y1 < BAND_OVERLAP:  # absorb a tiny trailing band
                y1 = h

            if progress_callback:
                pct = int(10 + 85 * band_index / total_bands)
                progress_callback(pct, f"Fúze pásu {band_index + 1}/{total_bands} (šetrné k paměti)...")

            band = [img[y0:y1] for img in images]
            fused = HDRMerger._mertens_once(band, cw, sw, ew)

            # Linear ramp over the overlap region on each side of the band.
            bh = y1 - y0
            ramp = np.ones((bh, 1), dtype=np.float32)
            fade = min(BAND_OVERLAP, bh // 2)
            if y0 > 0 and fade > 0:
                ramp[:fade, 0] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            if y1 < h and fade > 0:
                ramp[-fade:, 0] = np.linspace(1.0, 0.0, fade, dtype=np.float32)

            result[y0:y1] += fused * ramp[:, :, None]
            weight[y0:y1] += ramp
            del band, fused

            if y1 >= h:
                break
            y = y1
            band_index += 1

        np.maximum(weight, 1e-6, out=weight)
        result /= weight[:, :, None]

        if progress_callback:
            progress_callback(100, "Mertens fúze dokončena (pásový režim).")
        return _finite(result, hi=1.0)

    # ----------------------------------------------------------------- Debevec

    @staticmethod
    def merge_debevec(
        images: List[np.ndarray],
        times: List[float],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the Debevec 32-bit linear radiance map and the camera response.
        Returns (hdr_radiance_map, response_curve).
        """
        imgs = _validate_stack(images)
        times_arr = sanitize_exposure_times(times, len(imgs))

        if progress_callback:
            progress_callback(20, "Kalibrace odezvy snímače (Debevec CRF)...")

        try:
            crf = cv2.createCalibrateDebevec().process(_downscale_for_crf(imgs), times_arr)
            crf = _finite(crf)

            if progress_callback:
                progress_callback(60, "Výpočet 32-bit lineární mapy jasu (Radiance Map)...")

            hdr_map = cv2.createMergeDebevec().process(imgs, times_arr, crf)
        except cv2.error as e:
            raise HDRMergeError(f"OpenCV nedokázal spočítat Debevec HDR: {e}") from e

        hdr_map = _finite(hdr_map)

        if progress_callback:
            progress_callback(100, "Debevec HDR dokončeno.")
        return hdr_map, crf

    # --------------------------------------------------------------- Robertson

    @staticmethod
    def merge_robertson(
        images: List[np.ndarray],
        times: List[float],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the Robertson radiance map and camera response."""
        imgs = _validate_stack(images)
        times_arr = sanitize_exposure_times(times, len(imgs))

        if progress_callback:
            progress_callback(20, "Kalibrace odezvy snímače (Robertson CRF)...")

        try:
            crf = cv2.createCalibrateRobertson().process(_downscale_for_crf(imgs), times_arr)
            crf = _finite(crf)

            if progress_callback:
                progress_callback(60, "Výpočet Robertson HDR mapy...")

            hdr_map = cv2.createMergeRobertson().process(imgs, times_arr, crf)
        except cv2.error as e:
            raise HDRMergeError(f"OpenCV nedokázal spočítat Robertson HDR: {e}") from e

        hdr_map = _finite(hdr_map)

        if progress_callback:
            progress_callback(100, "Robertson HDR dokončeno.")
        return hdr_map, crf

    # -------------------------------------------------------------- Tonemapping

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
        """Tonemaps a 32-bit HDR radiance map down to [0, 1] float32 BGR."""
        if hdr_map is None or hdr_map.size == 0:
            raise HDRMergeError("Prázdná HDR mapa — tonemapping nelze provést.")

        # OpenCV's tonemappers divide by the image maximum; a NaN or a zero image
        # there produces an all-NaN result, so scrub before and after.
        src = _finite(hdr_map)
        peak = float(src.max())
        if peak <= 0.0:
            return np.zeros(src.shape, dtype=np.float32)

        method = (method or "reinhard").lower()
        gamma = float(max(0.05, gamma))

        try:
            if method == "reinhard":
                tm = cv2.createTonemapReinhard(
                    gamma=gamma,
                    intensity=float(intensity),
                    light_adapt=float(np.clip(light_adapt, 0.0, 1.0)),
                    color_adapt=float(np.clip(color_adapt, 0.0, 1.0)),
                )
            elif method == "drago":
                tm = cv2.createTonemapDrago(
                    gamma=gamma,
                    saturation=float(max(0.0, mantiuk_saturation)),
                    bias=float(np.clip(drago_bias, 0.7, 0.9)),
                )
            elif method == "mantiuk":
                tm = cv2.createTonemapMantiuk(
                    gamma=gamma,
                    scale=float(np.clip(mantiuk_scale, 0.6, 0.9)),
                    saturation=float(max(0.0, mantiuk_saturation)),
                )
            else:
                tm = cv2.createTonemap(gamma=gamma)

            ldr = tm.process(src)
        except cv2.error as e:
            raise HDRMergeError(f"Tonemapping selhal: {e}") from e

        return _finite(ldr, hi=1.0)
