"""
High-performance post-processing and export.

Designed for instant live preview (LUT-accelerated, vectorised operations),
astronomical coronal enhancement with dark-sky noise isolation, edge-preserving
denoising, and Unicode-safe file writing on Windows.
"""

from typing import Optional
import os
import cv2
import numpy as np

# Bilateral filtering is O(d^2) per pixel; above this size we filter a proxy and
# upsample the correction, which is visually identical and orders of magnitude faster.
DENOISE_FULL_MAX_PIXELS = 6_000_000


def _sanitize(img: np.ndarray) -> np.ndarray:
    """float32 BGR in [0, 1] with no NaN/Inf — the invariant every stage assumes."""
    arr = np.asarray(img)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def build_tone_curve_lut(
    brightness: float = 0.0,      # -0.5 .. 0.5
    contrast: float = 1.0,        # 0.2 .. 3.0
    gamma: float = 1.0,           # 0.2 .. 3.0
    shadow_lift: float = 0.0,     # 0.0 .. 1.0
    highlight_drop: float = 0.0,  # 0.0 .. 1.0
    lut_size: int = 1024
) -> np.ndarray:
    """Builds a 1-D look-up table for instantaneous tone mapping."""
    lut_size = max(2, int(lut_size))
    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    y = x.copy()

    # 1. Shadow lift and highlight roll-off, applied in linear-ish space first.
    if shadow_lift > 0.001:
        shadow_weight = np.clip(1.0 - x * 2.5, 0.0, 1.0)
        y += shadow_lift * 0.25 * (shadow_weight ** 1.5)

    if highlight_drop > 0.001:
        hl_weight = np.clip((x - 0.4) * 2.0, 0.0, 1.0)
        y -= highlight_drop * 0.25 * (hl_weight ** 1.5)

    # 2. Brightness and contrast.
    if brightness != 0.0:
        y = y + float(brightness)
    if contrast != 1.0:
        y = (y - 0.5) * float(contrast) + 0.5

    # 3. Gamma. Clamp first: a negative base with a fractional exponent is NaN.
    if abs(gamma - 1.0) > 0.001 and gamma > 0.05:
        y = np.power(np.clip(y, 0.0, None), 1.0 / float(gamma))

    y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def apply_denoise(image_f32_bgr: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Edge-preserving bilateral denoising: removes sensor grain from the sky
    without blurring fine coronal rays.
    """
    if strength <= 0.01:
        return image_f32_bgr

    img = _sanitize(image_f32_bgr)
    h, w = img.shape[:2]
    u8 = (img * 255.0).astype(np.uint8)

    d = 7
    sigma_color = float(strength * 75.0)
    sigma_space = float(strength * 9.0)

    try:
        if h * w > DENOISE_FULL_MAX_PIXELS:
            # Denoise a proxy, then transfer only the low-frequency correction back.
            scale = (DENOISE_FULL_MAX_PIXELS / float(h * w)) ** 0.5
            sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
            small = cv2.resize(u8, (sw, sh), interpolation=cv2.INTER_AREA)
            small_dn = cv2.bilateralFilter(small, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            correction = small_dn.astype(np.float32) - small.astype(np.float32)
            correction = cv2.resize(correction, (w, h), interpolation=cv2.INTER_LINEAR)
            denoised = np.clip(u8.astype(np.float32) + correction, 0.0, 255.0)
        else:
            denoised = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space).astype(np.float32)
    except cv2.error:
        return img

    blend = min(1.0, float(strength) * 1.5)
    res = (1.0 - blend) * img + blend * (denoised / 255.0)
    return np.clip(res, 0.0, 1.0).astype(np.float32)


def apply_coronal_detail_enhancement(
    img_bgr: np.ndarray,
    boost: float = 0.0,
    radius: float = 6.0,
    sky_threshold: float = 0.06
) -> np.ndarray:
    """
    Astronomical coronal detail enhancement.

    Multi-scale frequency separation with strict dark-sky gating, so the
    background sky grain is never sharpened and the overexposed inner corona
    never gains ringing halos.
    """
    if boost <= 0.01:
        return img_bgr

    img = _sanitize(img_bgr)

    # Perceptual luma: Y = 0.114*B + 0.587*G + 0.299*R
    luma = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]

    sigma1 = float(max(0.5, radius))
    sigma2 = float(max(1.0, radius * 2.5))

    blur1 = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    blur2 = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

    detail1 = luma - blur1
    detail2 = blur1 - blur2

    # Sky suppression: a smooth gate that is 0 in the dark sky and 1 in the corona.
    # The exponent argument is bounded because luma is already clipped to [0, 1].
    sky_gate = 1.0 / (1.0 + np.exp(-35.0 * (luma - float(sky_threshold))))
    # Overexposure roll-off, so the blown inner corona does not ring.
    core_gate = np.clip((1.0 - luma) * 1.8, 0.0, 1.0)
    mask = sky_gate * core_gate

    combined = (detail1 * 1.3 + detail2 * 0.7) * (float(boost) * 1.5) * mask
    return np.clip(img + combined[:, :, None], 0.0, 1.0).astype(np.float32)


def apply_postprocessing(
    image_f32: np.ndarray,
    brightness: float = 0.0,      # -0.5 .. 0.5
    contrast: float = 1.0,        # 0.2 .. 3.0
    gamma: float = 1.0,           # 0.2 .. 3.0
    saturation: float = 1.0,      # 0.0 .. 3.0
    coronal_boost: float = 0.0,   # 0.0 .. 2.0
    coronal_radius: float = 6.0,  # 1.0 .. 30.0 px
    shadow_lift: float = 0.0,     # 0.0 .. 1.0
    highlight_drop: float = 0.0,  # 0.0 .. 1.0
    denoise_strength: float = 0.0 # 0.0 .. 1.0
) -> np.ndarray:
    """Vectorised post-processing pipeline. Always returns float32 BGR in [0, 1]."""
    img = _sanitize(image_f32)
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    # 1. Denoise first, so later sharpening does not amplify grain.
    if denoise_strength > 0.01:
        img = apply_denoise(img, denoise_strength)

    # 2. Coronal detail boost, isolated from the sky background.
    if coronal_boost > 0.01:
        img = apply_coronal_detail_enhancement(img, boost=coronal_boost, radius=coronal_radius)

    # 3. Tone curve via LUT (brightness, contrast, gamma, shadows, highlights).
    needs_tone = (
        brightness != 0.0 or contrast != 1.0 or abs(gamma - 1.0) > 0.001
        or shadow_lift > 0.001 or highlight_drop > 0.001
    )
    if needs_tone:
        lut_size = 1024
        lut = build_tone_curve_lut(brightness, contrast, gamma, shadow_lift, highlight_drop, lut_size)
        # img is guaranteed finite and in [0, 1], so the index cast is always safe.
        indices = np.clip(img * (lut_size - 1), 0, lut_size - 1).astype(np.int32)
        img = lut[indices]

    # 4. Saturation around perceptual luma.
    if abs(saturation - 1.0) > 0.005:
        luma = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        img = luma[:, :, None] + float(saturation) * (img - luma[:, :, None])

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- Export

def _write_bytes(filepath: str, data: np.ndarray) -> bool:
    """
    Writes encoded image bytes through Python's own file layer.

    cv2.imwrite goes through the C locale and silently fails on paths containing
    non-ASCII characters — which on a Czech Windows install is most of them
    (C:\\Users\\Tomáš\\Obrázky\\...). Encoding in memory and writing here avoids that
    entirely.
    """
    directory = os.path.dirname(os.path.abspath(filepath))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(data.tobytes())
    return True


def save_image(
    filepath: str,
    image_f32_bgr: np.ndarray,
    hdr_radiance_map: Optional[np.ndarray] = None,
    jpeg_quality: int = 100
) -> bool:
    """
    Saves the result in the format implied by the extension:
      .tif/.tiff -> 16-bit TIFF      .png -> 16-bit PNG
      .jpg/.jpeg -> quality JPEG     .hdr/.exr -> 32-bit float radiance

    Returns True on success. Unicode paths are handled correctly on Windows.
    """
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in ('.hdr', '.exr'):
            if hdr_radiance_map is not None and hdr_radiance_map.size > 0:
                radiance = np.nan_to_num(hdr_radiance_map.astype(np.float32),
                                         nan=0.0, posinf=0.0, neginf=0.0)
            else:
                # Mertens has no radiance map; export the fused result as linear float.
                radiance = _sanitize(image_f32_bgr)
            ok, encoded = cv2.imencode(ext, radiance)
            return _write_bytes(filepath, encoded) if ok else False

        clipped = _sanitize(image_f32_bgr)

        if ext in ('.tif', '.tiff'):
            ok, encoded = cv2.imencode('.tif', (clipped * 65535.0).astype(np.uint16))
        elif ext == '.png':
            ok, encoded = cv2.imencode('.png', (clipped * 65535.0).astype(np.uint16))
        elif ext in ('.jpg', '.jpeg'):
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 1, 100))]
            ok, encoded = cv2.imencode('.jpg', (clipped * 255.0).astype(np.uint8), params)
        else:
            ok, encoded = cv2.imencode('.png', (clipped * 65535.0).astype(np.uint16))
            if ok and not ext:
                filepath = filepath + '.png'

        return _write_bytes(filepath, encoded) if ok else False

    except (cv2.error, OSError, ValueError) as e:
        print(f"Error saving image to {filepath}: {e}")
        return False


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """
    Unicode-safe replacement for cv2.imread.

    Reads the file with Python and decodes from memory, so accented paths and
    network shares work on Windows exactly like ASCII ones.
    """
    try:
        with open(filepath, 'rb') as f:
            buf = np.frombuffer(f.read(), dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, flags)
        return img
    except (OSError, ValueError, cv2.error):
        return None
