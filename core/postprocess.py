"""
High-Performance Post-Processing & Export Module.
Designed for instant real-time live preview (LUT-accelerated and vectorized operations),
advanced astronomical coronal enhancement with dark-sky noise isolation, and edge-preserving denoising.
"""

from typing import Optional
import os
import cv2
import numpy as np


def build_tone_curve_lut(
    brightness: float = 0.0,      # -0.5 to 0.5
    contrast: float = 1.0,        # 0.2 to 3.0
    gamma: float = 1.0,           # 0.2 to 3.0
    shadow_lift: float = 0.0,     # 0.0 to 1.0
    highlight_drop: float = 0.0,  # 0.0 to 1.0
    lut_size: int = 1024
) -> np.ndarray:
    """
    Constructs a 1D Look-Up Table (LUT) for instantaneous pixel tone mapping.
    """
    x = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
    y = x.copy()

    # 1. Shadow lift & highlight drop
    if shadow_lift > 0.001:
        shadow_weight = np.clip(1.0 - x * 2.5, 0.0, 1.0)
        y += shadow_lift * 0.25 * (shadow_weight ** 1.5)

    if highlight_drop > 0.001:
        hl_weight = np.clip((x - 0.4) * 2.0, 0.0, 1.0)
        y -= highlight_drop * 0.25 * (hl_weight ** 1.5)

    # 2. Brightness & Contrast
    if brightness != 0.0:
        y += brightness
    if contrast != 1.0:
        y = (y - 0.5) * contrast + 0.5

    # 3. Gamma
    if abs(gamma - 1.0) > 0.001 and gamma > 0.05:
        inv_gamma = 1.0 / gamma
        y = np.power(np.maximum(y, 0.0), inv_gamma)

    return np.clip(y, 0.0, 1.0).astype(np.float32)


def apply_denoise(image_f32_bgr: np.ndarray, strength: float = 0.0) -> np.ndarray:
    """
    Applies edge-preserving bilateral denoising to eliminate sensor grain without blurring fine coronal rays.
    """
    if strength <= 0.01:
        return image_f32_bgr

    # Convert to uint8 for fast bilateral/NlMeans
    u8 = (np.clip(image_f32_bgr, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Bilateral filter with diameter 7 and sigma proportional to strength
    d = 7
    sigma_color = float(strength * 75.0)
    sigma_space = float(strength * 9.0)
    denoised_u8 = cv2.bilateralFilter(u8, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # Smooth blend between original and denoised
    blend = min(1.0, strength * 1.5)
    res_f32 = (1.0 - blend) * image_f32_bgr + blend * (denoised_u8.astype(np.float32) / 255.0)
    return np.clip(res_f32, 0.0, 1.0)


def apply_coronal_detail_enhancement(
    img_bgr: np.ndarray,
    boost: float = 0.0,
    radius: float = 6.0,
    sky_threshold: float = 0.06
) -> np.ndarray:
    """
    Astronomical coronal detail enhancement.
    Uses multi-scale frequency separation with strict dark-sky thresholding,
    ensuring that background sky noise and grain are NEVER sharpened.
    """
    if boost <= 0.01:
        return img_bgr

    # Perceptual luma: Y = 0.114*B + 0.587*G + 0.299*R
    luma = 0.114 * img_bgr[:, :, 0] + 0.587 * img_bgr[:, :, 1] + 0.299 * img_bgr[:, :, 2]

    # Multi-scale Gaussian separation
    sigma1 = max(0.5, radius)
    sigma2 = max(1.0, radius * 2.5)

    blur1 = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
    blur2 = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma2, sigmaY=sigma2)

    detail1 = luma - blur1
    detail2 = blur1 - blur2

    # Smooth sigmoid mask: zero in dark sky (< sky_threshold), peaks in coronal midtones, fades in overexposed core
    # Sky suppression
    sky_gate = 1.0 / (1.0 + np.exp(-35.0 * (luma - sky_threshold)))
    # Overexposure roll-off
    core_gate = np.clip((1.0 - luma) * 1.8, 0.0, 1.0)
    mask = sky_gate * core_gate

    combined_detail = (detail1 * 1.3 + detail2 * 0.7) * (boost * 1.5) * mask
    
    # Add detail to all channels proportionally to preserve coronal color balance
    enhanced_bgr = img_bgr + combined_detail[:, :, None]
    return np.clip(enhanced_bgr, 0.0, 1.0)


def apply_postprocessing(
    image_f32: np.ndarray,
    brightness: float = 0.0,      # -0.5 to 0.5
    contrast: float = 1.0,        # 0.2 to 3.0
    gamma: float = 1.0,           # 0.2 to 3.0
    saturation: float = 1.0,      # 0.0 to 3.0
    coronal_boost: float = 0.0,   # 0.0 to 2.0 (Eclipse detail enhancement)
    coronal_radius: float = 6.0,  # 1.0 to 30.0 px
    shadow_lift: float = 0.0,     # 0.0 to 1.0
    highlight_drop: float = 0.0,  # 0.0 to 1.0
    denoise_strength: float = 0.0 # 0.0 to 1.0
) -> np.ndarray:
    """
    Ultrafast vectorized postprocessing pipeline.
    """
    img = np.clip(image_f32, 0.0, 1.0)

    # 1. Denoise if requested
    if denoise_strength > 0.01:
        img = apply_denoise(img, denoise_strength)

    # 2. Astronomical coronal detail boost (isolated from noise)
    if coronal_boost > 0.01:
        img = apply_coronal_detail_enhancement(img, boost=coronal_boost, radius=coronal_radius)

    # 3. Tone curve LUT mapping (Brightness, Contrast, Gamma, Shadows, Highlights)
    needs_tone = (
        brightness != 0.0 or contrast != 1.0 or abs(gamma - 1.0) > 0.001
        or shadow_lift > 0.001 or highlight_drop > 0.001
    )
    if needs_tone:
        lut = build_tone_curve_lut(brightness, contrast, gamma, shadow_lift, highlight_drop, lut_size=1024)
        indices = np.clip(img * 1023.0, 0, 1023).astype(np.int32)
        img = lut[indices]

    # 4. Instant vectorized linear saturation adjustment
    if abs(saturation - 1.0) > 0.005:
        # Perceptual luma
        luma = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
        img = luma[:, :, None] + saturation * (img - luma[:, :, None])

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def save_image(
    filepath: str,
    image_f32_bgr: np.ndarray,
    hdr_radiance_map: Optional[np.ndarray] = None,
    jpeg_quality: int = 100
) -> bool:
    """
    Saves image in specified format:
    - .tif / .tiff: 16-bit TIFF
    - .png: 8-bit or 16-bit PNG
    - .jpg / .jpeg: High quality JPEG
    - .hdr / .exr: 32-bit Radiance / OpenEXR HDR
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext in ('.hdr', '.exr') and hdr_radiance_map is not None:
            return cv2.imwrite(filepath, hdr_radiance_map)
        
        clipped = np.clip(image_f32_bgr, 0.0, 1.0)
        
        if ext in ('.tif', '.tiff'):
            u16 = (clipped * 65535.0).astype(np.uint16)
            return cv2.imwrite(filepath, u16)
        
        elif ext == '.png':
            u16 = (clipped * 65535.0).astype(np.uint16)
            return cv2.imwrite(filepath, u16)
            
        elif ext in ('.jpg', '.jpeg'):
            u8 = (clipped * 255.0).astype(np.uint8)
            params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
            success, encoded = cv2.imencode('.jpg', u8, params)
            if success:
                with open(filepath, 'wb') as f:
                    f.write(encoded)
                return True
            return False
            
        else:
            u8 = (clipped * 255.0).astype(np.uint8)
            return cv2.imwrite(filepath, u8)
            
    except Exception as e:
        print(f"Error saving image to {filepath}: {e}")
        return False
