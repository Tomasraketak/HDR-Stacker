"""
Post-processing filters and export utilities.
Includes custom astronomical coronal detail enhancement algorithms (High-Pass / Multiscale Unsharp Mask)
and adjustments for exposure, contrast, gamma, saturation, and export formats.
"""

from typing import Optional
import os
import cv2
import numpy as np


def apply_postprocessing(
    image_f32: np.ndarray,
    brightness: float = 0.0,      # -1.0 to 1.0
    contrast: float = 1.0,        # 0.2 to 3.0
    gamma: float = 1.0,           # 0.2 to 3.0
    saturation: float = 1.0,      # 0.0 to 3.0
    coronal_boost: float = 0.0,   # 0.0 to 2.0 (Eclipse detail enhancement)
    coronal_radius: float = 5.0,  # 1.0 to 30.0 px
    shadow_lift: float = 0.0,     # 0.0 to 1.0
    highlight_drop: float = 0.0   # 0.0 to 1.0
) -> np.ndarray:
    """
    Applies image processing adjustments to an image in float32 format [0.0, 1.0] (BGR).
    Returns float32 BGR image in [0.0, 1.0].
    """
    img = np.clip(image_f32, 0.0, 1.0).copy()

    # 1. Coronal Detail Enhancer (Astronomical Unsharp Masking / Multi-scale filter)
    if coronal_boost > 0.001:
        # Convert to LAB for luminance-only detail extraction
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]  # range 0 to 100 in OpenCV float LAB
        
        # Multi-scale Gaussian blur
        sigma1 = max(0.5, coronal_radius)
        sigma2 = max(1.0, coronal_radius * 2.5)
        
        blur1 = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma1, sigmaY=sigma1)
        blur2 = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=sigma2, sigmaY=sigma2)
        
        # Bandpass details
        detail1 = l_channel - blur1
        detail2 = blur1 - blur2
        
        # Protect deep blacks and pure white saturated center using a smooth bell-shaped weight
        norm_l = np.clip(l_channel / 100.0, 0.0, 1.0)
        sin_val = np.maximum(0.0, np.sin(np.pi * norm_l))
        mask = np.power(sin_val, 0.8)
        
        combined_detail = (detail1 * 1.2 + detail2 * 0.8) * coronal_boost * mask
        enhanced_l = np.clip(l_channel + combined_detail * 15.0, 0.0, 100.0)
        
        lab[:, :, 0] = enhanced_l
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        img = np.clip(img, 0.0, 1.0)

    # 2. Shadow Lift / Highlight Drop
    if shadow_lift > 0.001 or highlight_drop > 0.001:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if shadow_lift > 0.001:
            shadow_mask = np.clip(1.0 - gray * 2.0, 0.0, 1.0)
            img += shadow_lift * 0.3 * shadow_mask[:, :, None]
        if highlight_drop > 0.001:
            highlight_mask = np.clip((gray - 0.5) * 2.0, 0.0, 1.0)
            img -= highlight_drop * 0.3 * highlight_mask[:, :, None]
        img = np.clip(img, 0.0, 1.0)

    # 3. Brightness & Contrast
    if brightness != 0.0:
        img = img + brightness
    if contrast != 1.0:
        img = (img - 0.5) * contrast + 0.5
    img = np.clip(img, 0.0, 1.0)

    # 4. Gamma correction
    if abs(gamma - 1.0) > 0.001 and gamma > 0.05:
        inv_gamma = 1.0 / gamma
        img = np.power(np.maximum(img, 0.0), inv_gamma)

    # 5. Saturation
    if abs(saturation - 1.0) > 0.001:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0.0, 1.0)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


def save_image(
    filepath: str,
    image_f32_bgr: np.ndarray,
    hdr_radiance_map: Optional[np.ndarray] = None,
    jpeg_quality: int = 100
) -> bool:
    """
    Saves image in specified format:
    - .tif / .tiff: 16-bit TIFF (optimal for PixInsight / Photoshop / Astro processing)
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
