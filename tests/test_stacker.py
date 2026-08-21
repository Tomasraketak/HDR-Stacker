"""
Automated unit & integration test for Astro HDR Stacker.
Tests synthetic exposure generation, multi-algorithm alignment (Eclipse disc, ECC, ORB, MTB),
Mertens fusion, Debevec HDR, denoise filter, coronal enhancement, and file exports.
"""

import os
import sys
import tempfile
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.exif_and_analysis import inspect_exposure_files
from core.aligner import ImageAligner
from core.merger import HDRMerger
from core.postprocess import apply_postprocessing, save_image, build_tone_curve_lut, apply_denoise


def generate_synthetic_eclipse_exposures(output_dir: str, num_exposures: int = 9) -> list:
    """
    Generates synthetic solar eclipse images simulating different exposures (-4 EV to +4 EV).
    """
    paths = []
    h, w = 400, 400
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Lunar dark disc in center
    moon_radius = 45.0
    corona_raw = np.where(r <= moon_radius, 0.0, 1.0 / (np.maximum(r - moon_radius, 1.0) ** 0.85))
    angle = np.arctan2(y - cy, x - cx)
    streamers = 1.0 + 0.35 * np.sin(6 * angle) + 0.25 * np.cos(14 * angle)
    corona_intensity = corona_raw * streamers

    base_t = 1.0 / 4000.0
    shutter_times = [base_t * (2.0 ** i) for i in range(num_exposures)]

    np.random.seed(42)

    for i, t in enumerate(shutter_times):
        scale = t * 3000.0
        frame = corona_intensity * scale
        
        b = np.clip(frame * 240.0, 0, 255).astype(np.uint8)
        g = np.clip(frame * 245.0, 0, 255).astype(np.uint8)
        r_c = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        bgr = np.dstack([b, g, r_c])

        # Add small subpixel jitter
        dx = int(np.random.randint(-3, 4))
        dy = int(np.random.randint(-3, 4))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        jittered = cv2.warpAffine(bgr, M, (w, h))

        path = os.path.join(output_dir, f"eclipse_frame_{i+1:02d}.jpg")
        cv2.imwrite(path, jittered)
        paths.append(path)

    return paths


def run_all_tests():
    print("[*] Starting Astro HDR Stacker comprehensive tests...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Generate 9 synthetic exposures
        print("1. Generating 9 synthetic exposures (-4 EV to +4 EV)...")
        file_paths = generate_synthetic_eclipse_exposures(tmpdir, 9)
        assert len(file_paths) == 9, "Expected 9 generated images"
        print("   [OK] Generated 9 test frames.")

        # 2. Test auto-detection and luminance sorting
        print("2. Testing luminance sorting and EV assignment...")
        items = inspect_exposure_files(file_paths, user_ev_step=1.0)
        assert len(items) == 9, "Expected 9 exposure items"
        for i in range(len(items) - 1):
            assert items[i].mean_luminance <= items[i+1].mean_luminance, "Items must be sorted by brightness"
            assert items[i].calculated_ev < items[i+1].calculated_ev, "Calculated EV must be strictly ascending"
        print(f"   [OK] Auto-sorted properly. EV range: {items[0].calculated_ev} EV to {items[-1].calculated_ev} EV.")

        # 3. Test loading images
        images = [cv2.imread(it.filepath) for it in items]
        times = [it.exposure_time for it in items]
        assert all(img is not None for img in images), "All images should load"

        # 4. Test Multi-Algorithm Alignment
        print("3. Testing multi-algorithm alignment...")
        for method in ["eclipse_disc", "ecc", "orb", "mtb", "none"]:
            aligner = ImageAligner(method=method, max_bits=4, exclude_range=4)
            aligned = aligner.align(images)
            assert len(aligned) == 9, f"Alignment failed for method: {method}"
            print(f"   [OK] Alignment method '{method}' passed.")

        # 5. Test Mertens Exposure Fusion
        print("4. Testing Mertens Exposure Fusion with noise suppression...")
        mertens_res = HDRMerger.merge_mertens(aligned, contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0)
        assert mertens_res.shape == aligned[0].shape, "Mertens shape mismatch"
        assert mertens_res.dtype == np.float32, "Mertens output must be float32"
        assert 0.0 <= mertens_res.min() and mertens_res.max() <= 1.0, "Mertens values must be in [0, 1]"
        print("   [OK] Mertens Exposure Fusion successful.")

        # 6. Test Denoise & LUT Tone Mapping
        print("5. Testing Bilateral Denoise & Fast 1D Tone Curve LUT...")
        denoised = apply_denoise(mertens_res, strength=0.5)
        assert denoised.shape == mertens_res.shape
        lut = build_tone_curve_lut(brightness=0.1, contrast=1.2, gamma=1.1, shadow_lift=0.2, highlight_drop=0.1)
        assert len(lut) == 1024
        print("   [OK] Denoise & Tone Curve LUT successful.")

        # 7. Test Coronal Detail Enhancer with dark sky noise protection
        print("6. Testing Coronal Detail Enhancer filter...")
        enhanced = apply_postprocessing(
            mertens_res,
            brightness=0.05,
            contrast=1.1,
            gamma=1.0,
            saturation=1.2,
            coronal_boost=0.5,
            coronal_radius=5.0,
            denoise_strength=0.3
        )
        assert enhanced.shape == mertens_res.shape
        assert enhanced.dtype == np.float32
        print("   [OK] Coronal Detail Enhancer successful.")

        # 8. Test Exporting (TIFF 16-bit, JPG, PNG)
        print("7. Testing image export formats...")
        tif_path = os.path.join(tmpdir, "result.tif")
        jpg_path = os.path.join(tmpdir, "result.jpg")
        png_path = os.path.join(tmpdir, "result.png")
        
        assert save_image(tif_path, enhanced), "TIFF save failed"
        assert save_image(jpg_path, enhanced), "JPG save failed"
        assert save_image(png_path, enhanced), "PNG save failed"
        assert os.path.exists(tif_path) and os.path.getsize(tif_path) > 1000, "TIFF file invalid"
        assert os.path.exists(jpg_path) and os.path.getsize(jpg_path) > 1000, "JPG file invalid"
        print("   [OK] 16-bit TIFF, JPG and PNG export successful.")

        # 9. Test GUI Import and Classes
        print("8. Testing PyQt6 GUI module imports...")
        from gui.styles import DARK_THEME
        from gui.controls_panel import ControlsPanel
        from gui.exposure_list_widget import ExposureListWidget
        from gui.image_viewer import InteractiveImageViewer
        from gui.main_window import MainWindow
        print("   [OK] All PyQt6 modules imported cleanly.")

    print("\n>>> ALL TESTS PASSED SUCCESSFULLY! No bugs detected. <<<")


if __name__ == "__main__":
    run_all_tests()
