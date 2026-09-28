"""
Comprehensive test suite for Astro HDR Stacker.

Covers the numerical core (detection, alignment, fusion, tonemapping,
post-processing, export) and the stability scenarios that used to crash the
GUI: rapid ROI dragging, worker cancellation, and closing during work.

Run with:  python tests/test_stacker.py
"""

import os
import sys
import json
import math
import shutil
import tempfile
import time
import traceback

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from core.exif_and_analysis import (
    inspect_exposure_files, format_shutter_speed, read_image_size,
    estimate_stack_megapixels,
)
from core.aligner import (
    detect_black_circle_in_light, calculate_moon_shifts,
    apply_shifts_to_images, find_sun_or_moon_center,
    detect_point_lights, estimate_translation_from_points,
    calculate_light_pattern_shifts,
)
from core.merger import HDRMerger, HDRMergeError, sanitize_exposure_times
from core.postprocess import (
    apply_postprocessing, save_image, build_tone_curve_lut,
    apply_denoise, imread_unicode,
)
from core.image_cache import ImageCache, available_memory_bytes
from core.project import (build_project, save_project, load_project, resolved_paths,
                          apply_frame_records, ProjectError, PROJECT_FORMAT_VERSION)
from core.solar_position import (sun_position, sun_angular_radius_deg, julian_day,
                                 greenwich_sidereal_time_deg, refraction_deg,
                                 ecliptic_horizontal)
from core.exif_and_analysis import extract_capture_time, extract_gps_position
from core.eclipse_composite import (
    SkyCamera, Calibration, CompositeSettings, CompositeError, PartialFrame, solve_camera,
    calibrate, compute_placements, detect_sun_disc, detect_totality_disc, cut_out_sun,
    load_image_float, render_composite, frame_gains, path_polyline, format_time,
    ColorGrade, apply_grade,
)
from core.solar_position import _solar_coordinates
from datetime import datetime, timedelta

_FAILURES = []
_PASSES = 0


def check(condition: bool, message: str):
    global _PASSES
    if condition:
        _PASSES += 1
    else:
        _FAILURES.append(message)
        print(f"   [FAIL] {message}")


def section(title: str):
    print(f"\n=== {title} ===")


# --------------------------------------------------------------- Test fixtures

def generate_synthetic_eclipse_exposures(output_dir: str, num_exposures: int = 9,
                                         size: int = 400, jitter: int = 3) -> list:
    """Synthesises a totality bracket: dark lunar disc, streamered corona, jitter."""
    paths = []
    h = w = size
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    moon_radius = size * 0.1125
    corona_raw = np.where(r <= moon_radius, 0.0,
                          1.0 / (np.maximum(r - moon_radius, 1.0) ** 0.85))
    angle = np.arctan2(y - cy, x - cx)
    streamers = 1.0 + 0.35 * np.sin(6 * angle) + 0.25 * np.cos(14 * angle)
    corona = corona_raw * streamers

    base_t = 1.0 / 4000.0
    rng = np.random.default_rng(42)

    for i in range(num_exposures):
        t = base_t * (2.0 ** i)
        frame = corona * (t * 3000.0)
        bgr = np.dstack([
            np.clip(frame * 240.0, 0, 255).astype(np.uint8),
            np.clip(frame * 245.0, 0, 255).astype(np.uint8),
            np.clip(frame * 255.0, 0, 255).astype(np.uint8),
        ])
        if jitter:
            dx, dy = int(rng.integers(-jitter, jitter + 1)), int(rng.integers(-jitter, jitter + 1))
            bgr = cv2.warpAffine(bgr, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h))

        path = os.path.join(output_dir, f"eclipse_frame_{i + 1:02d}.jpg")
        cv2.imwrite(path, bgr)
        paths.append(path)
    return paths


def generate_lamp_scene_bracket(num_exposures: int = 9, shake: float = 6.0,
                                ground_gain: float = 0.35, n_lamps: int = 24,
                                seed: int = 7):
    """
    A totality frame over a landscape: corona at the top, a horizon of static
    street lamps at the bottom, and random camera shake between exposures.

    Returns (images, true_shifts_to_reference, ref_idx).
    """
    h, w = 900, 1400
    rng = np.random.default_rng(seed)
    y, x = np.ogrid[:h, :w]
    cy, cx = int(h * 0.32), w // 2
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    moon = 60.0
    corona = np.where(r <= moon, 0.0, 1.0 / np.maximum(r - moon, 1.0) ** 0.85)
    angle = np.arctan2(y - cy, x - cx)
    corona = corona * (1 + 0.35 * np.sin(6 * angle) + 0.25 * np.cos(13 * angle))

    lamps = np.zeros((h, w), np.float32)
    for _ in range(n_lamps):
        lx = rng.uniform(w * 0.02, w * 0.98)
        ly = rng.uniform(h * 0.62, h * 0.95)
        lamps[int(ly), int(lx)] = rng.uniform(3.0, 9.0)
    lamps = cv2.GaussianBlur(lamps, (0, 0), 1.6)

    ground = np.zeros((h, w), np.float32)
    ground[int(h * 0.60):, :] = 0.012

    images, offsets = [], []
    ref_idx = num_exposures // 2
    for i in range(num_exposures):
        t = (1 / 4000.0) * (2.0 ** i)
        s = t * 3000.0
        frame = corona * s + lamps * (s * 0.9) + ground * (s * ground_gain)
        frame = frame + rng.normal(0, 0.0025, frame.shape)
        bgr = np.dstack([np.clip(frame * 240, 0, 255),
                         np.clip(frame * 245, 0, 255),
                         np.clip(frame * 255, 0, 255)]).astype(np.uint8)

        dx = 0.0 if i == ref_idx else rng.uniform(-shake, shake)
        dy = 0.0 if i == ref_idx else rng.uniform(-shake, shake)
        offsets.append((dx, dy))
        images.append(cv2.warpAffine(bgr, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE))

    truth = [(offsets[ref_idx][0] - dx, offsets[ref_idx][1] - dy) for dx, dy in offsets]
    return images, truth, ref_idx


def test_static_light_alignment():
    section("2b. Static street-light pattern alignment")

    images, truth, ref_idx = generate_lamp_scene_bracket()

    points = detect_point_lights(images[ref_idx])
    check(len(points) >= 10, f"lamps must be detected in the reference frame (got {len(points)})")
    check(points.shape[1] == 3, "detections must be (x, y, brightness)")
    h, w = images[ref_idx].shape[:2]
    check(bool(np.all(points[:, 1] > h * 0.4)),
          "the default search band must stay below the Sun")
    check(bool(np.all(np.diff(points[:, 2]) <= 1e-3)), "detections must be brightest-first")

    # A pure translation must be recovered exactly.
    shifted = np.column_stack([points[:, 0] - 7.25, points[:, 1] + 3.5, points[:, 2]])
    recovered = estimate_translation_from_points(points, shifted)
    check(recovered is not None, "a synthetic translation must be recovered")
    if recovered:
        check(abs(recovered[0] - 7.25) < 0.2 and abs(recovered[1] + 3.5) < 0.2,
              f"recovered translation must be accurate (got {recovered[:2]})")

    # Half the lamps missing, plus decoys: the constellation vote must still win.
    partial = shifted[::2]
    decoys = np.column_stack([
        np.random.default_rng(3).uniform(0, w, 8),
        np.random.default_rng(4).uniform(h * 0.5, h, 8),
        np.full(8, 5.0)])
    noisy = np.vstack([partial, decoys]).astype(np.float32)
    robust = estimate_translation_from_points(points, noisy)
    check(robust is not None and abs(robust[0] - 7.25) < 0.5 and abs(robust[1] + 3.5) < 0.5,
          "matching must survive missing lamps and spurious detections")

    # Full-bracket accuracy against ground truth.
    shifts, matches = calculate_light_pattern_shifts(images)
    check(len(shifts) == len(images), "one shift per frame")
    errors = [np.hypot(sx - tx, sy - ty) for (sx, sy), (tx, ty) in zip(shifts, truth)]
    confident = [e for e, m in zip(errors, matches) if m > 0]
    check(len(confident) >= len(images) - 2,
          f"most frames must align confidently ({len(confident)}/{len(images)})")
    check(float(np.median(confident)) < 0.5,
          f"confident alignments must be subpixel (median {np.median(confident):.3f} px)")
    print(f"   {len(confident)}/{len(images)} frames confident, "
          f"median error {np.median(confident):.3f} px")

    # It must beat disc alignment on this scene, which is the whole point.
    moon_shifts = calculate_moon_shifts(images)
    moon_errors = [np.hypot(sx - tx, sy - ty) for (sx, sy), (tx, ty) in zip(moon_shifts, truth)]
    print(f"   lunar-disc alignment on the same scene: "
          f"median {np.median(moon_errors):.3f} px")
    check(float(np.median(confident)) < float(np.median(moon_errors)),
          "light-pattern alignment must beat disc alignment when a fixed foreground exists")

    # A tripod-steady bracket must produce essentially zero shift.
    steady, steady_truth, _ = generate_lamp_scene_bracket(shake=0.0)
    steady_shifts, steady_matches = calculate_light_pattern_shifts(steady)
    steady_err = [np.hypot(sx - tx, sy - ty)
                  for (sx, sy), (tx, ty), m in zip(steady_shifts, steady_truth, steady_matches)
                  if m > 0]
    check(float(np.max(steady_err)) < 0.5,
          f"a steady bracket must not be moved (max {np.max(steady_err):.3f} px)")

    # A frame with no usable pattern must be left alone, never guessed at.
    blank = [np.zeros((300, 400, 3), np.uint8) for _ in range(3)]
    blank_shifts, blank_matches = calculate_light_pattern_shifts(blank)
    check(all(s == (0.0, 0.0) for s in blank_shifts),
          "a featureless bracket must produce no shifts")
    check(all(m <= 0 for m in blank_matches), "a featureless bracket must report no matches")

    check(len(detect_point_lights(None)) == 0, "None input must return no detections")
    check(estimate_translation_from_points(points[:2], points[:2]) is None,
          "too few points must return None rather than a bogus translation")
    check(calculate_light_pattern_shifts([]) == ([], []), "an empty stack is handled")


# ------------------------------------------------------------------ Core tests

def test_exposure_analysis(tmpdir, paths):
    section("1. EXIF analysis, EV mapping and user-state preservation")

    items = inspect_exposure_files(paths, user_ev_step=1.0)
    check(len(items) == 9, "expected 9 exposure items")
    for i in range(len(items) - 1):
        check(items[i].mean_luminance <= items[i + 1].mean_luminance,
              f"frames must be sorted by brightness at index {i}")
        check(items[i].calculated_ev < items[i + 1].calculated_ev,
              f"EV must be strictly ascending at index {i}")
    print(f"   EV range: {items[0].calculated_ev} .. {items[-1].calculated_ev}")

    check(all(it.width == 400 and it.height == 400 for it in items),
          "dimensions must be read from the file header")
    check(abs(estimate_stack_megapixels(items) - 0.16) < 0.01,
          "stack megapixels must be estimated from the headers")

    # User state must survive a re-scan; this used to silently reset alignment.
    items[0].shift_x, items[0].shift_y, items[0].is_valid = 7.5, -3.5, False
    preserve = {it.filepath: it for it in items}
    rescanned = inspect_exposure_files(paths, user_ev_step=1.0, preserve=preserve)
    target = next(it for it in rescanned if it.filepath == items[0].filepath)
    check(target.shift_x == 7.5 and target.shift_y == -3.5 and not target.is_valid,
          "manual shifts and exclusions must survive a re-scan")

    check(format_shutter_speed(1 / 1000) == "1/1000s", "shutter formatting: fractions")
    check(format_shutter_speed(2.0) == "2s", "shutter formatting: whole seconds")
    check(format_shutter_speed(0) == "N/A", "shutter formatting: invalid input")
    check(format_shutter_speed(float('nan')) == "N/A", "shutter formatting: NaN input")
    check(read_image_size(paths[0]) == (400, 400), "header size read")
    return items


def test_detection_and_alignment(items):
    section("2. Lunar disc detection and subpixel alignment")

    images = [imread_unicode(it.filepath) for it in items]
    check(all(img is not None for img in images), "all frames must decode")

    circle = detect_black_circle_in_light(images[4])
    check(circle is not None, "moon silhouette must be detected")
    if circle:
        cx, cy, rad = circle
        print(f"   disc at ({cx:.1f}, {cy:.1f}) r={rad:.1f}px")
        check(150 < cx < 250 and 150 < cy < 250, f"disc centre out of range: {circle}")
        check(30 < rad < 65, f"disc radius out of range: {rad}")

    cx, cy = find_sun_or_moon_center(images[4])
    check(150 < cx < 250 and 150 < cy < 250, f"universal finder out of range: {cx},{cy}")

    # A pure landscape must not be mistaken for a disc.
    landscape = np.zeros((300, 400, 3), np.uint8)
    landscape[200:, :] = 40
    fx, fy = find_sun_or_moon_center(landscape)
    check(0 <= fx < 400 and 0 <= fy < 300, "finder must stay in bounds on a featureless frame")

    shifts = calculate_moon_shifts(images)
    check(len(shifts) == 9, "one shift per frame")
    check(all(np.isfinite(dx) and np.isfinite(dy) for dx, dy in shifts),
          "all shifts must be finite")

    aligned = apply_shifts_to_images(images, shifts)
    check(len(aligned) == 9, "alignment must return every frame")
    check(all(a.shape == images[0].shape for a in aligned), "alignment must preserve shape")

    # Alignment must actually reduce residual disc scatter.
    before = [detect_black_circle_in_light(i) for i in images]
    after = [detect_black_circle_in_light(a) for a in aligned]
    spread_before = np.std([c[0] for c in before if c]) + np.std([c[1] for c in before if c])
    spread_after = np.std([c[0] for c in after if c]) + np.std([c[1] for c in after if c])
    print(f"   disc scatter: {spread_before:.2f}px -> {spread_after:.2f}px")
    check(spread_after <= spread_before + 0.1, "alignment must not increase disc scatter")

    # Degenerate inputs must not raise.
    check(calculate_moon_shifts([]) == [], "empty stack returns empty shifts")
    check(calculate_moon_shifts([images[0]]) == [(0.0, 0.0)], "single frame needs no shift")
    check(len(calculate_moon_shifts(images, ref_idx=999)) == 9, "out-of-range ref index is clamped")
    check(apply_shifts_to_images(images[:2], []) is not None, "missing shifts are tolerated")
    check(detect_black_circle_in_light(None) is None, "None input returns None")
    check(detect_black_circle_in_light(np.zeros((4, 4, 3), np.uint8)) is None,
          "a tiny frame returns None")
    return images, aligned


def test_merging(aligned, items):
    section("3. Fusion engines, exposure-time validation and tonemapping")

    times = [it.exposure_time for it in items]

    fusion = HDRMerger.merge_mertens(aligned, 1.0, 1.0, 1.0)
    check(fusion.shape == aligned[0].shape, "Mertens shape mismatch")
    check(fusion.dtype == np.float32, "Mertens must return float32")
    check(0.0 <= fusion.min() and fusion.max() <= 1.0, "Mertens must stay in [0, 1]")
    check(not np.isnan(fusion).any(), "Mertens output must be finite")

    # All-zero weights used to divide by zero and produce a black frame.
    degenerate = HDRMerger.merge_mertens(aligned, 0.0, 0.0, 0.0)
    check(degenerate.max() > 0.01, "zero weights must not yield a black image")

    banded = HDRMerger._merge_mertens_banded(aligned, 1.0, 1.0, 1.0)
    diff = float(np.abs(fusion - banded).max())
    print(f"   banded vs single-pass fusion: max diff {diff:.2e}")
    check(diff < 1e-4, f"memory-bounded banded fusion must match single-pass (got {diff})")

    hdr, crf = HDRMerger.merge_debevec(aligned, times)
    check(np.isfinite(hdr).all(), "Debevec radiance map must be finite")
    check(hdr.max() > 0, "Debevec radiance map must be non-trivial")

    ldr = HDRMerger.tonemap(hdr, "reinhard")
    check(np.isfinite(ldr).all() and ldr.max() <= 1.0, "Reinhard tonemap must be finite and bounded")
    for method in ("drago", "mantiuk", "linear"):
        out = HDRMerger.tonemap(hdr, method)
        check(np.isfinite(out).all(), f"{method} tonemap must be finite")

    hdr_r, _ = HDRMerger.merge_robertson(aligned, times)
    check(np.isfinite(hdr_r).all(), "Robertson radiance map must be finite")

    # Exposure-time validation: a zero time used to poison the whole map with NaN.
    repaired = sanitize_exposure_times([0.0, 0.2, 0.4, 0.8], 4)
    check(bool((repaired > 0).all()), "zero exposure times must be repaired")
    check(bool(np.all(np.diff(np.sort(repaired)) > 0)), "repaired times must be distinct")

    for bad, label in (([0.0] * 4, "all-zero times"), ([0.1] * 4, "identical times")):
        try:
            sanitize_exposure_times(bad, 4)
            check(False, f"{label} must be rejected with a clear message")
        except HDRMergeError:
            check(True, "")

    hdr_zero, _ = HDRMerger.merge_debevec(aligned[:4], [0.0, 0.2, 0.4, 0.8])
    check(not np.isnan(hdr_zero).any(), "a zero exposure time must not produce NaN radiance")

    try:
        HDRMerger.merge_mertens([aligned[0], cv2.resize(aligned[1], (100, 100))])
        check(False, "mismatched frame sizes must be rejected")
    except HDRMergeError:
        check(True, "")

    try:
        HDRMerger.merge_mertens([])
        check(False, "an empty stack must be rejected")
    except HDRMergeError:
        check(True, "")

    check(HDRMerger.tonemap(np.zeros((8, 8, 3), np.float32)).max() == 0.0,
          "an all-black radiance map must tonemap to black, not NaN")
    return fusion, hdr


def test_postprocessing(fusion):
    section("4. Post-processing pipeline")

    lut = build_tone_curve_lut(brightness=0.1, contrast=1.2, gamma=1.1,
                               shadow_lift=0.2, highlight_drop=0.1)
    check(len(lut) == 1024, "LUT size")
    check(bool(np.isfinite(lut).all()) and lut.min() >= 0 and lut.max() <= 1,
          "LUT must be finite and bounded")
    check(bool(np.isfinite(build_tone_curve_lut(brightness=-0.5, gamma=0.4)).all()),
          "extreme LUT parameters must stay finite")

    denoised = apply_denoise(fusion, strength=0.5)
    check(denoised.shape == fusion.shape, "denoise must preserve shape")
    check(float(np.std(cv2.Laplacian(denoised, cv2.CV_32F))) <=
          float(np.std(cv2.Laplacian(fusion, cv2.CV_32F))) + 1e-6,
          "denoise must not add high-frequency energy")

    enhanced = apply_postprocessing(
        fusion, brightness=0.05, contrast=1.1, gamma=1.0, saturation=1.2,
        coronal_boost=0.5, coronal_radius=5.0, denoise_strength=0.3)
    check(enhanced.shape == fusion.shape and enhanced.dtype == np.float32,
          "post-processing must preserve shape and dtype")
    check(bool(np.isfinite(enhanced).all()), "post-processing output must be finite")

    # Non-finite input used to produce a garbage LUT index and could crash.
    poisoned = np.full((32, 32, 3), np.nan, np.float32)
    out = apply_postprocessing(poisoned, brightness=0.2, gamma=0.5, contrast=2.0,
                               saturation=1.5, coronal_boost=0.5, denoise_strength=0.5)
    check(bool(np.isfinite(out).all()), "NaN input must be scrubbed, not propagated")

    inf_in = np.full((16, 16, 3), np.inf, np.float32)
    check(bool(np.isfinite(apply_postprocessing(inf_in)).all()), "Inf input must be scrubbed")

    identity = apply_postprocessing(fusion)
    check(float(np.abs(identity - fusion).max()) < 1e-5,
          "default parameters must be a no-op")

    # The dark sky must be protected from the coronal sharpener.
    sky = np.zeros((64, 64, 3), np.float32)
    sky += np.random.default_rng(0).normal(0.02, 0.005, sky.shape).astype(np.float32)
    sky = np.clip(sky, 0, 1)
    boosted = apply_postprocessing(sky, coronal_boost=2.0, coronal_radius=4.0)
    check(float(np.std(boosted)) <= float(np.std(sky)) * 1.5,
          "the coronal filter must not amplify dark-sky grain")
    return enhanced


def test_export(tmpdir, enhanced, hdr):
    section("5. Export formats and Unicode paths")

    # Accented directory: the exact shape of a Czech Windows user folder.
    unicode_dir = os.path.join(tmpdir, "Tomáš Příliš žluťoučký kůň")
    os.makedirs(unicode_dir, exist_ok=True)

    for ext, radiance in ((".tif", None), (".png", None), (".jpg", None), (".hdr", hdr)):
        path = os.path.join(unicode_dir, "výsledek" + ext)
        check(save_image(path, enhanced, hdr_radiance_map=radiance),
              f"{ext} export must succeed on a Unicode path")
        check(os.path.exists(path) and os.path.getsize(path) > 500,
              f"{ext} file must be written and non-trivial")
        check(imread_unicode(path) is not None, f"{ext} file must read back")

    tif = imread_unicode(os.path.join(unicode_dir, "výsledek.tif"), cv2.IMREAD_UNCHANGED)
    check(tif is not None and tif.dtype == np.uint16, "TIFF must be 16-bit")

    # A Mertens result has no radiance map; .hdr must still export as float.
    hdr_no_map = os.path.join(unicode_dir, "mertens.hdr")
    check(save_image(hdr_no_map, enhanced, hdr_radiance_map=None),
          ".hdr export must work without a radiance map")
    readback = imread_unicode(hdr_no_map, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    check(readback is not None and readback.dtype == np.float32,
          ".hdr must be written as 32-bit float, not 8-bit")

    # A path whose parent is a regular file can never be created, on any OS
    # and for any user — so this exercises the failure branch reliably.
    blocker = os.path.join(tmpdir, "blocker.txt")
    with open(blocker, "w") as f:
        f.write("not a directory")
    check(save_image(os.path.join(blocker, "sub", "f.tif"), enhanced) is False,
          "an unwritable path must return False, not raise")


def test_image_cache(paths):
    section("6. Bounded image cache")

    cache = ImageCache(budget_bytes=2 * 1024 * 1024)
    first = cache.get(paths[0], 1.0)
    check(first is not None, "cache must decode a valid file")
    check(cache.get(paths[0], 1.0) is first, "a repeat request must hit the cache")
    check(cache.get(paths[0], 0.25).shape[0] == first.shape[0] // 4,
          "scaled requests must be honoured")

    for p in paths:
        cache.get(p, 1.0)
    _entries, used, budget = cache.stats()
    check(used <= budget, f"cache must respect its budget ({used} > {budget})")

    check(cache.get(os.path.join(os.path.dirname(paths[0]), "missing.jpg")) is None,
          "a missing file must return None, not raise")
    cache.invalidate()
    check(cache.stats()[1] == 0, "invalidate must free everything")

    mem = available_memory_bytes()
    check(mem is None or mem > 0, "memory probe must return None or a positive value")


def test_projects(tmpdir, paths):
    section("6b. Project files — saving and reopening a session")

    items = inspect_exposure_files(paths, user_ev_step=1.0)
    items[0].shift_x, items[0].shift_y = 7.5, -3.5
    items[2].is_valid = False
    settings = {"gamma": 1.75, "algo": "debevec", "crop_rect": (40, 30, 280, 220)}

    # A Unicode project name in a Unicode folder — the normal Czech case.
    project_dir = os.path.join(tmpdir, "můj projekt")
    os.makedirs(project_dir, exist_ok=True)
    project_path = os.path.join(project_dir, "zatmění 2026.ahdrproj")

    save_project(build_project(items, settings, project_path=project_path,
                               crop_rect=(40, 30, 280, 220), ev_step=1.5,
                               roi_active=True, roi_rect=(10, 20, 300, 300)),
                 project_path)
    check(os.path.isfile(project_path), "project must be written to a Unicode path")

    loaded, missing = load_project(project_path)
    check(not missing, f"no frames should be missing right after saving: {missing}")
    check(len(loaded.frames) == len(items), "every frame must be stored")
    check(loaded.settings == settings, "settings must round-trip exactly")
    check(loaded.crop_rect == (40, 30, 280, 220), "the crop must round-trip")
    check(loaded.roi_active is True and loaded.roi_rect == (10, 20, 300, 300),
          "ROI state must round-trip")
    check(abs(loaded.ev_step - 1.5) < 1e-6, "the EV step must round-trip")
    check(loaded.format_version == PROJECT_FORMAT_VERSION, "format version is recorded")

    # Per-frame state must land on the right frames even though re-inspection
    # re-sorts the list by exposure.
    fresh = inspect_exposure_files(resolved_paths(loaded, project_path), user_ev_step=1.0)
    matched = apply_frame_records(loaded, fresh, project_path)
    check(matched == len(items), f"every frame must be matched by path ({matched})")
    restored = {os.path.basename(it.filepath): (round(it.shift_x, 1), round(it.shift_y, 1),
                                                it.is_valid) for it in fresh}
    original = {os.path.basename(it.filepath): (round(it.shift_x, 1), round(it.shift_y, 1),
                                                it.is_valid) for it in items}
    check(restored == original, "shifts and exclusions must be restored per frame")

    # Moving the whole folder must keep the project working, via relative paths.
    # The project and its photos have to live in one self-contained folder for
    # that to be a meaningful test — that is how a card gets copied to a laptop.
    bundle = os.path.join(tmpdir, "výprava")
    os.makedirs(bundle, exist_ok=True)
    bundle_photos = []
    for src in paths:
        dst = os.path.join(bundle, os.path.basename(src))
        shutil.copy2(src, dst)
        bundle_photos.append(dst)
    bundle_items = inspect_exposure_files(bundle_photos, user_ev_step=1.0)
    bundle_items[0].shift_x = 4.5
    bundle_project = os.path.join(bundle, "výprava.ahdrproj")
    save_project(build_project(bundle_items, settings, project_path=bundle_project),
                 bundle_project)

    moved_root = os.path.join(tmpdir, "přesunuto")
    shutil.copytree(bundle, moved_root)
    moved_project = os.path.join(moved_root, "výprava.ahdrproj")
    moved, moved_missing = load_project(moved_project)
    moved_found = resolved_paths(moved, moved_project)
    check(not moved_missing and len(moved_found) == len(bundle_photos),
          f"a moved project must still find its photos ({len(moved_found)})")
    check(all(moved_root in f for f in moved_found),
          "a moved project must use the photos next to it, not the originals")

    # A deleted photo is reported but must not abort the load.
    # Delete from the copy AND make the original unreachable for that frame,
    # otherwise the absolute-path fallback legitimately finds the original.
    os.remove(os.path.join(moved_root, os.path.basename(bundle_photos[1])))
    os.remove(bundle_photos[1])
    partial, partial_missing = load_project(moved_project)
    partial_found = resolved_paths(partial, moved_project)
    check(len(partial_missing) == 1, f"the missing photo must be reported ({partial_missing})")
    check(len(partial_found) == len(bundle_photos) - 1,
          "the remaining photos must still load")

    # Malformed input must raise a clear error, never a traceback.
    for name, content in (("broken.ahdrproj", "{not json at all"),
                          ("alien.ahdrproj", '{"something": 1}')):
        bad = os.path.join(tmpdir, name)
        with open(bad, "w", encoding="utf-8") as f:
            f.write(content)
        try:
            load_project(bad)
            check(False, f"{name} must be rejected")
        except ProjectError:
            check(True, "")

    try:
        load_project(os.path.join(tmpdir, "does-not-exist.ahdrproj"))
        check(False, "a missing project file must be rejected")
    except ProjectError:
        check(True, "")

    # A project from a newer build must be refused rather than half-read.
    future = os.path.join(tmpdir, "future.ahdrproj")
    with open(project_path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["format_version"] = PROJECT_FORMAT_VERSION + 5
    with open(future, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    try:
        load_project(future)
        check(False, "a newer format version must be refused")
    except ProjectError as e:
        check("novější verzí" in str(e), "the version error must explain itself")

    # Unknown keys from a future build must be ignored, not fatal.
    forward = os.path.join(tmpdir, "forward.ahdrproj")
    with open(project_path, encoding="utf-8") as f:
        payload = json.load(f)
    payload["future_top_level"] = {"x": 1}
    payload["frames"][0]["future_frame_field"] = 42
    with open(forward, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    forward_project, _ = load_project(forward)
    check(len(forward_project.frames) == len(items),
          "unknown keys from a newer build must be ignored, not fatal")
    print(f"   project round-trip verified, {os.path.getsize(project_path)} B on disk")


# ------------------------------------------------- Eclipse sequence composite

COMPOSITE_SITE = (42.5987, -5.5671, 2.0)          # León, CEST
COMPOSITE_TOTALITY = datetime(2026, 8, 12, 20, 28, 0)


def _write_exif_jpeg(path: str, bgr_u8: np.ndarray, moment: datetime,
                     gps=None, offset: str = "+02:00"):
    """Saves a JPEG carrying DateTimeOriginal (+ sub-seconds, offset) and GPS."""
    from PIL import Image, ExifTags
    img = Image.fromarray(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB))
    exif = img.getexif()
    sub = exif.get_ifd(ExifTags.IFD.Exif)
    sub[0x9003] = moment.strftime("%Y:%m:%d %H:%M:%S")
    sub[0x9291] = f"{moment.microsecond // 10000:02d}"
    sub[0x9011] = offset
    if gps is not None:
        lat, lon = gps
        g = exif.get_ifd(ExifTags.IFD.GPSInfo)
        g[1] = "N" if lat >= 0 else "S"
        g[2] = (float(int(abs(lat))), float(int(abs(lat) * 60) % 60), (abs(lat) * 3600) % 60)
        g[3] = "E" if lon >= 0 else "W"
        g[4] = (float(int(abs(lon))), float(int(abs(lon) * 60) % 60), (abs(lon) * 3600) % 60)
    img.save(path, quality=97, exif=exif)


def _crescent_frame(size=(900, 640), centre=(430.3, 310.6), radius=31.0,
                    moon_offset=(22.0, -9.0), level=0.9, tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    """A filtered partial-phase frame: a limb-darkened disc minus the Moon."""
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xx - centre[0], yy - centre[1])
    mu = np.sqrt(np.clip(1.0 - (d / radius) ** 2, 0.0, 1.0))
    disc = np.clip(radius + 0.5 - d, 0.0, 1.0) * (0.6 + 0.4 * mu)
    moon = np.clip(np.hypot(xx - centre[0] - moon_offset[0],
                            yy - centre[1] - moon_offset[1]) - radius * 1.03 + 0.5, 0.0, 1.0)
    value = disc * moon * level
    bgr = np.dstack([value * tint[0], value * tint[1], value * tint[2]])
    return np.clip(bgr * 255.0 + 1.5, 0, 255).astype(np.uint8)


def _composite_scene(tmpdir: str):
    """
    A synthetic totality background photographed by a known camera, plus
    filtered partial-phase frames with EXIF times — everything the composite
    needs, with ground truth to check against.
    """
    lat, lon, off = COMPOSITE_SITE
    az0, alt0 = sun_position(COMPOSITE_TOTALITY, lat, lon, off)
    rs = sun_angular_radius_deg(COMPOSITE_TOTALITY, off)
    cam = SkyCamera(1600, 1000, 2200.0, az0 + 4.0, alt0 + 3.0, 0.8)

    w, h = cam.width, cam.height
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    h1 = cam.project(az0 - 15.0, 0.0)
    h2 = cam.project(az0 + 15.0, 0.0)
    # Ground below the true horizon line, a blue twilight sky above it.
    side = (h2[0] - h1[0]) * (yy - h1[1]) - (h2[1] - h1[1]) * (xx - h1[0])
    sky = side < 0
    bg = np.zeros((h, w, 3), np.float32)
    bg[..., 0] = np.where(sky, 0.30 - 0.12 * yy / h, 0.03)
    bg[..., 1] = np.where(sky, 0.16 - 0.06 * yy / h, 0.03)
    bg[..., 2] = np.where(sky, 0.08, 0.03)
    sx, sy = cam.project(az0, alt0)
    r_moon = cam.focal * math.tan(math.radians(rs)) * 1.03
    d = np.hypot(xx - sx, yy - sy)
    corona = np.where(d <= r_moon, 0.0, 1.0 / np.maximum(d - r_moon + 1.0, 1.0) ** 0.9)
    bg = np.clip(bg * (d > r_moon)[..., None] + corona[..., None] * np.array([0.9, 0.95, 1.0]),
                 0.0, 1.0)
    bg_path = os.path.join(tmpdir, "totalita_stack.tif")
    save_image(bg_path, bg)

    frames = []
    for minutes, level, moon_dx in ((-40, 0.45, 12.0), (-12, 0.95, 40.0), (30, 0.7, -30.0)):
        moment = COMPOSITE_TOTALITY + timedelta(minutes=minutes, seconds=0.25)
        path = os.path.join(tmpdir, f"castecna_{minutes:+d}.jpg")
        _write_exif_jpeg(path, _crescent_frame(level=level, moon_offset=(moon_dx, -6.0)),
                         moment, gps=(lat, lon))
        frames.append((path, moment, level))
    return dict(camera=cam, bg_path=bg_path, horizon=(h1[0], h1[1], h2[0], h2[1]),
                sun=(sx, sy), sun_radius_deg=rs, moon_radius=r_moon, frames=frames)


def test_solar_position():
    section("8a. Solar ephemeris")
    # Meeus, Astronomical Algorithms, example 25.a (1992 Oct 13.0).
    ra, dec, dist, _eps = _solar_coordinates(julian_day(datetime(1992, 10, 13)))
    check(abs(ra - 198.38083) < 0.001 and abs(dec + 7.78507) < 0.001,
          f"Sun RA/Dec must match Meeus 25.a ({ra:.5f}, {dec:.5f})")
    check(abs(dist - 0.99766) < 0.0001, f"Sun distance must match Meeus ({dist:.5f})")
    # Meeus example 12.a: GMST on 1987 April 10, 0h UT.
    gmst = greenwich_sidereal_time_deg(julian_day(datetime(1987, 4, 10)))
    check(abs(gmst - 197.693195) < 1e-4, f"sidereal time must match Meeus 12.a ({gmst:.6f})")

    # At local solar noon the true altitude is 90 - |lat - dec|, due south.
    moment = datetime(2026, 6, 21, 12, 0)
    (az, alt), _m = max(((sun_position(moment + timedelta(minutes=m), 50.0, 0.0, 0.0,
                                       refraction=False), m) for m in range(-30, 31)),
                        key=lambda t: t[0][1])
    check(abs(alt - (90.0 - 50.0 + 23.43)) < 0.05, f"noon altitude at the solstice ({alt:.3f})")
    check(abs(az - 180.0) < 1.5, f"noon azimuth must be south ({az:.2f})")

    r0 = refraction_deg(0.0)
    check(0.45 < r0 < 0.65 and refraction_deg(45.0) < 0.02,
          f"refraction must be ~0.5 deg at the horizon, tiny high up ({r0:.3f})")
    rs = sun_angular_radius_deg(datetime(2026, 8, 12))
    check(0.260 < rs < 0.266, f"August solar radius ({rs:.4f})")
    # The Sun lies on the ecliptic: the closest ecliptic sample is within a step.
    lat, lon, off = COMPOSITE_SITE
    az_s, alt_s = sun_position(COMPOSITE_TOTALITY, lat, lon, off)
    nearest = min(math.hypot((a - az_s) * math.cos(math.radians(alt_s)), h - alt_s)
                  for _l, a, h in ecliptic_horizontal(COMPOSITE_TOTALITY, lat, lon, off, 0.25))
    check(nearest < 0.25, f"the Sun must lie on the ecliptic ({nearest:.3f} deg)")
    print(f"   Meeus examples reproduced; León totality Sun at az {az_s:.2f}, alt {alt_s:.2f}")


def _check_grades(bg, settings, cutouts, camera, placements, reference):
    """Colour grading: the pixel maths, and the master / per-frame / background layers."""
    ramp = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    grey = np.repeat(ramp[None, :, None], 3, axis=2)
    colour = np.dstack([ramp * 0.3, ramp * 0.6, ramp]).astype(np.float32)

    check(np.array_equal(apply_grade(colour, ColorGrade()), colour),
          "a neutral grade must leave the image untouched")
    warm = apply_grade(grey, ColorGrade(temperature=60.0))
    luma = warm @ np.array([0.114, 0.587, 0.299], np.float32)
    check(float(warm[0, 40, 2]) > float(warm[0, 40, 0]) * 1.4
          and float(np.abs(luma[0, :80] - ramp[:80]).max()) < 0.01,
          "a warmer temperature must shift grey towards red at the same brightness")
    magenta = apply_grade(grey, ColorGrade(tint=60.0))
    check(float(magenta[0, 50, 1]) < float(magenta[0, 50, 2]) * 0.9,
          "a magenta tint must lower green")
    mono = apply_grade(colour, ColorGrade(saturation=-100.0))
    check(float(np.ptp(mono, axis=2).max()) < 1e-5, "saturation -100 must be greyscale")
    vivid = apply_grade(colour, ColorGrade(saturation=50.0))
    check(float(np.ptp(vivid[0, 50])) > float(np.ptp(colour[0, 50])) * 1.3,
          "more saturation must spread the channels")
    punchy = apply_grade(grey, ColorGrade(contrast=60.0))
    check(punchy[0, 0, 0] == 0.0 and abs(float(punchy[0, 100, 0]) - 1.0) < 1e-6
          and abs(float(punchy[0, 50, 0]) - 0.5) < 1e-6
          and punchy[0, 25, 0] < 0.25 and punchy[0, 75, 0] > 0.75
          and bool(np.all(np.diff(punchy[0, :, 0]) >= -1e-7)),
          "contrast must keep black, white and the pivot, and stay monotonic")
    lifted = apply_grade(grey, ColorGrade(midtones=60.0))
    check(lifted[0, 0, 0] == 0.0 and abs(float(lifted[0, 100, 0]) - 1.0) < 1e-6
          and abs(float(lifted[0, 50, 0]) - 0.65) < 1e-5,
          "mid-tones must move the middle only")
    brighter = apply_grade(grey, ColorGrade(exposure=1.0))
    check(abs(float(brighter[0, 20, 0]) - 0.4) < 1e-5 and float(brighter.max()) <= 1.0,
          "+1 EV must double the brightness, clipped at white")
    # The black sky around a cut-out Sun, and the Moon, must never turn grey:
    # black stays exactly black, and the curves' slope at black is bounded
    # (a plain gamma of 2 would lift 0.01 to 0.1 on its own).
    extreme = ColorGrade(contrast=-100.0, midtones=100.0, saturation=100.0, temperature=100.0)
    check(float(apply_grade(np.zeros((4, 4, 3), np.float32), extreme).max()) == 0.0
          and float(apply_grade(np.full((4, 4, 3), 0.01, np.float32), extreme).max()) < 0.08,
          "black must stay black under any grade")
    big = np.random.default_rng(3).random((37, 23, 3), dtype=np.float32)
    mixed = ColorGrade(exposure=0.4, contrast=35.0, midtones=-20.0, saturation=30.0,
                       temperature=-25.0, tint=15.0)
    check(np.allclose(apply_grade(big, mixed, band_rows=5), apply_grade(big, mixed, band_rows=4096),
                      atol=1e-6),
          "grading in bands must equal grading in one piece")

    # Master exposure and a frame's own exposure add up.
    base = [g for g, _c in frame_gains(settings, cutouts)]
    settings.sun_grade.exposure = 1.0
    settings.frames[0].grade.exposure = -1.0
    both = [g for g, _c in frame_gains(settings, cutouts)]
    check(abs(both[0] - base[0]) < 1e-6 and abs(both[1] - 2.0 * base[1]) < 1e-6,
          "the master and per-frame exposure must add up")
    settings.sun_grade.exposure = 0.0
    settings.frames[0].grade.exposure = 0.0

    # Master temperature warms every Sun; one frame can cancel it for itself.
    settings.sun_grade.temperature = 80.0
    settings.frames[1].grade.temperature = -80.0
    toned = render_composite(bg, settings, cutouts, camera)
    ratios = []
    for place in placements:
        r = place.radius
        box = toned[int(place.y - r):int(place.y + r) + 1, int(place.x - r):int(place.x + r) + 1]
        lit = box.max(axis=2) > 0.3
        ratios.append(float(box[..., 2][lit].mean() / max(1e-6, box[..., 0][lit].mean())))
    check(ratios[0] > 1.5 and ratios[2] > 1.5 and abs(ratios[1] - 1.0) < 0.1,
          f"the master grade must warm every Sun except the one that cancels it ({ratios})")
    settings.sun_grade = ColorGrade()
    settings.frames[1].grade = ColorGrade()

    # The background has its own grade, which leaves the Suns alone.
    settings.background_grade.exposure = -1.0
    darker = render_composite(bg, settings, cutouts, camera)
    corner = (slice(0, 40), slice(0, 40))
    check(np.allclose(darker[corner], reference[corner] * 0.5, atol=2e-3),
          "the background grade must darken the background")
    p1 = placements[1]
    sun_box = (slice(int(p1.y - 3), int(p1.y + 4)), slice(int(p1.x - 3), int(p1.x + 4)))
    check(abs(float(darker[sun_box].max()) - float(reference[sun_box].max())) < 0.02,
          "the background grade must not touch the Suns")
    pre = apply_grade(bg, settings.background_grade)
    check(np.allclose(render_composite(pre, settings, cutouts, camera, background_graded=True),
                      darker, atol=1e-6),
          "a pre-graded background must render the same picture")
    settings.background_grade = ColorGrade()
    print(f"   grades: master/per-frame R:B ratios {', '.join(f'{v:.2f}' for v in ratios)}")


def test_eclipse_composite(tmpdir):
    section("8b. Eclipse sequence composite — calibration, placement, brightness")
    scene = _composite_scene(tmpdir)
    cam_true = scene["camera"]
    lat, lon, off = COMPOSITE_SITE

    # EXIF: capture time with sub-seconds and offset, and GPS.
    path0, moment0, _lvl = scene["frames"][0]
    moment, offset = extract_capture_time(path0)
    check(moment == moment0 and offset == 2.0,
          f"EXIF capture time must round-trip ({moment}, {offset})")
    gps = extract_gps_position(path0)
    check(gps is not None and abs(gps[0] - lat) < 1e-4 and abs(gps[1] - lon) < 1e-4,
          f"EXIF GPS must round-trip ({gps})")

    # Sun detection in a filtered frame, including a thin crescent and a red Sun.
    for moon_dx, tint, label in ((12.0, (1, 1, 1), "fat"), (46.0, (1, 1, 1), "thin"),
                                 (-20.0, (0.02, 0.3, 1.0), "red")):
        frame = _crescent_frame(moon_offset=(moon_dx, -4.0), tint=tint)
        disc = detect_sun_disc(frame)
        ok = disc is not None and math.hypot(disc[0] - 430.3, disc[1] - 310.6) < 1.0 \
            and abs(disc[2] - 31.0) < 1.0
        check(ok, f"{label} crescent: solar disc must be recovered ({disc})")
    check(detect_sun_disc(np.full((300, 400, 3), 2, np.uint8)) is None,
          "a black frame has no Sun")

    bg = load_image_float(scene["bg_path"])
    check(bg is not None and bg.dtype == np.float32 and bg.shape == (1000, 1600, 3),
          "a 16-bit TIFF background must load as float")
    disc = detect_totality_disc(bg)
    sx, sy = scene["sun"]
    check(disc is not None and math.hypot(disc[0] - sx, disc[1] - sy) < 0.7,
          f"the lunar disc must be found in the corona ({disc} vs {sx:.1f},{sy:.1f})")
    check(disc is not None and abs(disc[2] - scene["moon_radius"]) < 1.0,
          f"the lunar radius must be measured ({disc[2] if disc else None} vs "
          f"{scene['moon_radius']:.2f})")

    # Camera solve: exact inputs give the exact camera back.
    az0, alt0 = sun_position(COMPOSITE_TOTALITY, lat, lon, off)
    rs = scene["sun_radius_deg"]
    exact = Calibration(1600, 1000, sx, sy, 2 * cam_true.focal * math.tan(math.radians(rs)),
                        scene["horizon"], 0.0)
    for mode in ("auto", "diameter", "horizon"):
        cam = solve_camera(exact, az0, alt0, rs, mode).camera
        check(abs(cam.focal - cam_true.focal) < 0.5 and abs(cam.roll - cam_true.roll) < 0.01
              and abs(((cam.yaw - cam_true.yaw + 180) % 360) - 180) < 0.01
              and abs(cam.pitch - cam_true.pitch) < 0.01,
              f"camera solve ({mode}) must recover the true camera ({cam})")
    try:
        solve_camera(Calibration(1600, 1000, sx, sy), az0, alt0, rs, "auto")
        check(False, "no scale information must be refused")
    except CompositeError:
        check(True, "")

    # Full pipeline with the DETECTED Moon: the Moon is ~3 % bigger than the
    # Sun, so the automatic mode must lean on the horizon for the scale.
    settings = CompositeSettings(
        background_path=scene["bg_path"], background_time=format_time(COMPOSITE_TOTALITY),
        utc_offset_hours=off, latitude=lat, longitude=lon, location_set=True,
        sun_x=disc[0], sun_y=disc[1], sun_diameter=2 * disc[2], horizon=scene["horizon"])
    cutouts = []
    for path, moment, _level in scene["frames"]:
        img = load_image_float(path)
        fdisc = detect_sun_disc(img)
        settings.frames.append(PartialFrame(path=path, filename=os.path.basename(path),
                                            time=format_time(extract_capture_time(path)[0]),
                                            disc=fdisc))
        cutouts.append(cut_out_sun(img, fdisc))
    report = calibrate(settings, 1600, 1000)
    placements = compute_placements(settings, report.camera)
    worst = 0.0
    for place, (_p, moment, _l) in zip(placements, scene["frames"]):
        a, e = sun_position(moment, lat, lon, off)
        tx, ty = cam_true.project(a, e)
        worst = max(worst, math.hypot(place.x - tx, place.y - ty))
    true_r = cam_true.focal * math.tan(math.radians(rs))
    check(worst < 1.5, f"partial Suns must land on the true path (worst {worst:.2f} px)")
    check(all(abs(p.radius - true_r) < 0.03 * true_r for p in placements),
          "each Sun must be drawn at the true solar size")
    print(f"   placement error {worst:.2f} px over a 70-minute sequence")

    # Rendering: every Sun equalised to the same surface brightness.
    settings.target_level = 0.8
    out = render_composite(bg, settings, cutouts, report.camera)
    check(out.shape == bg.shape and np.isfinite(out).all(), "the composite must be finite")
    levels = []
    for place in placements:
        x, y, r = int(round(place.x)), int(round(place.y)), place.radius
        patch = out[int(y - r):int(y + r) + 1, int(x - r):int(x + r) + 1].max(axis=2)
        levels.append(float(np.percentile(patch, 97)))
    check(max(levels) - min(levels) < 0.08 and all(abs(v - 0.8) < 0.12 for v in levels),
          f"surface brightness must be equalised to the target ({levels})")
    gains = frame_gains(settings, cutouts)
    check(gains[0][0] > gains[1][0] * 1.5,
          "the dimmest frame must receive the largest automatic gain")

    # A manual EV correction brightens only that Sun.
    settings.frames[2].grade.exposure = -1.0
    dimmer = render_composite(bg, settings, cutouts, report.camera)
    p2 = placements[2]
    region = (slice(int(p2.y - p2.radius), int(p2.y + p2.radius) + 1),
              slice(int(p2.x - p2.radius), int(p2.x + p2.radius) + 1))
    check(float(dimmer[region].max()) < float(out[region].max()) * 0.7,
          "a -1 EV correction must darken that Sun")
    settings.frames[2].grade.exposure = 0.0

    _check_grades(bg, settings, cutouts, report.camera, placements, out)

    # A Sun below the horizon must not be painted over the landscape.
    below = CompositeSettings.from_dict(settings.to_dict())
    below.frames[2].time = format_time(COMPOSITE_TOTALITY + timedelta(minutes=75))
    place_below = compute_placements(below, report.camera)[2]
    a, e = sun_position(COMPOSITE_TOTALITY + timedelta(minutes=75), lat, lon, off)
    check(e < -0.5, f"the test Sun must really be below the horizon ({e:.2f})")
    clipped = render_composite(bg, below, cutouts, report.camera)
    if place_below is not None and 0 <= place_below.x < 1600 and 0 <= place_below.y < 1000:
        x, y = int(place_below.x), int(place_below.y)
        check(np.allclose(clipped[y - 3:y + 4, x - 3:x + 4], bg[y - 3:y + 4, x - 3:x + 4],
                          atol=1e-3),
              "a Sun below the horizon must be hidden by the landscape")
    below.clip_below_horizon = False
    unclipped = render_composite(bg, below, cutouts, report.camera)
    if place_below is not None and 0 <= place_below.x < 1600 and 0 <= place_below.y < 1000:
        check(float(unclipped[y - 3:y + 4, x - 3:x + 4].max()) > 0.5,
              "with clipping off the Sun must be painted")

    # The path overlay passes through the totality Sun.
    path = path_polyline(settings, report.camera, 1.0)
    near = min(math.hypot(px - sx, py - sy) for t, px, py in path if np.isfinite(px))
    check(near < 8.0, f"the drawn path must pass the totality Sun ({near:.2f} px)")

    # Proxy rendering is the same picture at a smaller size.
    proxy = cv2.resize(bg, (800, 500), interpolation=cv2.INTER_AREA)
    small = render_composite(proxy, settings, cutouts, report.camera, scale=0.5)
    p0 = placements[0]
    # Sample the whole disc: its centre may well be covered by the Moon.
    r_half = int(math.ceil(p0.radius * 0.5))
    x_half, y_half = int(round(p0.x * 0.5)), int(round(p0.y * 0.5))
    small_patch = small[y_half - r_half:y_half + r_half + 1, x_half - r_half:x_half + r_half + 1]
    check(float(small_patch.max()) > 0.6,
          "the proxy preview must show the Sun at the scaled position")

    # Settings survive a dict round-trip, including tuples and grades.
    settings.sun_grade.saturation = 25.0
    settings.background_grade.contrast = -10.0
    settings.frames[1].grade.temperature = -30.0
    again = CompositeSettings.from_dict(json.loads(json.dumps(settings.to_dict())))
    check(again.horizon == tuple(settings.horizon) and len(again.frames) == 3
          and again.frames[0].disc == tuple(settings.frames[0].disc),
          "composite settings must round-trip through JSON")
    check(again.sun_grade == settings.sun_grade and again.background_grade == settings.background_grade
          and again.frames[1].grade == settings.frames[1].grade,
          "master, background and per-frame grades must round-trip through JSON")
    settings.sun_grade = ColorGrade()
    settings.background_grade = ColorGrade()
    settings.frames[1].grade = ColorGrade()
    # Projects from before the grades: target level and per-frame EV carry over.
    old = CompositeSettings.from_dict({"target_level": 0.425,
                                       "frames": [{"path": "x.jpg", "ev_adjust": 0.5}]})
    check(abs(old.sun_grade.exposure + 1.0) < 1e-6 and abs(old.target_level - 0.85) < 1e-9
          and abs(old.frames[0].grade.exposure - 0.5) < 1e-6,
          f"an older project's brightness settings must migrate ({old.sun_grade}, "
          f"{old.frames[0].grade})")
    check(ColorGrade.from_dict({"contrast": 900, "tint": "x"}) == ColorGrade(contrast=100.0),
          "grade values must be clamped and bad ones ignored")
    check(CompositeSettings.from_dict({"blend_mode": "bogus", "frames": [{"path": "x",
                                                                          "future": 1}]}
                                      ).blend_mode == "lighten",
          "unknown values and keys must be tolerated")

    # Saved inside a project, the composite follows the folder when it moves.
    bundle = os.path.join(tmpdir, "kompozit")
    os.makedirs(bundle, exist_ok=True)
    moved_settings = CompositeSettings.from_dict(settings.to_dict())
    moved_settings.background_path = shutil.copy2(scene["bg_path"], bundle)
    for frame in moved_settings.frames:
        frame.path = shutil.copy2(frame.path, bundle)
    project_path = os.path.join(bundle, "časosběr.ahdrproj")
    save_project(build_project([], {}, project_path=project_path,
                               eclipse_composite=moved_settings.to_dict()), project_path)
    moved_root = os.path.join(tmpdir, "kompozit_přesunut")
    shutil.copytree(bundle, moved_root)
    shutil.rmtree(bundle)
    loaded, missing = load_project(os.path.join(moved_root, "časosběr.ahdrproj"))
    restored = CompositeSettings.from_dict(loaded.eclipse_composite)
    check(not missing and restored.background_path.startswith(moved_root)
          and all(f.path.startswith(moved_root) for f in restored.frames),
          f"a moved composite project must find its photos ({missing})")
    check(restored.horizon == tuple(settings.horizon) and restored.sun_diameter == settings.sun_diameter,
          "the composite calibration must survive the project file")
    return scene


def test_composite_gui(tmpdir, scene):
    section("8c. Eclipse composite editor — GUI")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication, QEventLoop
    # Kept referenced: a collected QApplication aborts every later widget.
    app = QApplication.instance() or QApplication([])
    check(app is not None, "a QApplication must exist")
    import gui.eclipse_composite_window as ecw
    from gui.main_window import MainWindow

    class _SilentBox:
        """Modal message boxes would block an unattended test."""
        StandardButton = ecw.QMessageBox.StandardButton
        messages = []

        @classmethod
        def information(cls, *args, **kwargs):
            cls.messages.append(args[1:])

        warning = information

        @classmethod
        def question(cls, *args, **kwargs):
            return ecw.QMessageBox.StandardButton.Yes

    original_box = ecw.QMessageBox
    ecw.QMessageBox = _SilentBox

    def pump(timeout_ms: int, until=None) -> bool:
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            if until is not None and until():
                return True
            time.sleep(0.01)
        return until() if until is not None else True

    try:
        window = MainWindow(session_persistence=False)
        window.show()
        window.open_eclipse_composite()
        editor = window._composite_window
        check(editor is not None and editor.isVisible(), "the composite editor must open")

        editor.set_background(scene["bg_path"])
        pump(10000, lambda: editor._bg_proxy is not None and not editor.busy())
        s = editor.settings
        check(editor._bg_size == (1600, 1000), f"background size ({editor._bg_size})")
        check(s.sun_diameter > 0 and math.hypot(s.sun_x - scene["sun"][0],
                                                s.sun_y - scene["sun"][1]) < 1.0,
              "loading the background must find the lunar disc")

        # The stack has no EXIF: take the time and place from an original frame.
        editor.bg_time.set_value(COMPOSITE_TOTALITY)
        editor._on_bg_time_changed()
        editor.spin_utc.setValue(2.0)
        editor.combo_place.setCurrentIndex(1)            # León preset
        editor._on_horizon_drawn(*scene["horizon"])

        editor.add_partials([p for p, _m, _l in scene["frames"]])
        pump(15000, lambda: not editor.busy() and len(s.frames) == 3)
        check(len(s.frames) == 3 and all(f.time_from_exif for f in s.frames),
              "partial frames must be added with their EXIF times")
        check(all(editor._cutouts.get(f.path) is not None for f in s.frames),
              "every partial frame must yield a Sun cut-out")
        check([f.moment for f in s.frames] == sorted(f.moment for f in s.frames),
              "frames must be ordered by time")

        pump(500, lambda: all(p is not None for p in editor._placements) and editor._placements)
        check(editor._report is not None, f"the editor must calibrate ({editor._calib_error})")
        check(len(editor._placements) == 3 and all(p is not None for p in editor._placements),
              "every frame must be placed")
        check(editor.canvas._base_pixmap is not None, "the preview must be rendered")
        overlay = editor.canvas._overlay
        check(bool(overlay.get("path")) and bool(overlay.get("ticks"))
              and len(overlay.get("markers", [])) == 3,
              "the path, its time ticks and the frame markers must be drawn")

        # Drag a Sun on the canvas, then nudge it with the keyboard.
        place = editor._placements[1]
        # Two moves before the debounced preview re-renders — a fast drag.
        editor._on_frame_dragged(1, place.x + 4.0, place.y - 2.0)
        editor._on_frame_dragged(1, place.x + 12.0, place.y - 5.0)
        check(abs(s.frames[1].offset_x - 12.0) < 0.05 and abs(s.frames[1].offset_y + 5.0) < 0.05,
              f"dragging a Sun must store the offset from its computed place "
              f"({s.frames[1].offset_x}, {s.frames[1].offset_y})")
        editor._select_frame(1)
        editor.nudge_selected(0.2, 0.0)
        check(abs(s.frames[1].offset_x - 12.2) < 0.01, "arrow nudges must add to the offset")
        editor._reset_frame_manual()
        check(s.frames[1].offset_x == 0.0 and s.frames[1].offset_y == 0.0,
              "the reset must return the Sun to its computed place")

        # Per-frame, master and background grades, and the look controls.
        editor.grade_frame.rows["exposure"].setValue(0.5)
        editor.grade_frame.rows["temperature"].setValue(30.0)
        check(abs(s.frames[1].grade.exposure - 0.5) < 1e-6
              and s.frames[1].grade.temperature == 30.0 and s.frames[0].grade.temperature == 0.0,
              "the per-frame sliders must edit the selected frame only")
        editor._select_frame(0)
        check(editor.grade_frame.rows["temperature"].value() == 0.0,
              "selecting another frame must show that frame's own grade")
        editor._select_frame(1)
        check(editor.grade_frame.rows["temperature"].value() == 30.0,
              "selecting a frame again must show its grade")
        editor.grade_master.rows["saturation"].setValue(-40.0)
        editor.grade_master.rows["contrast"].setValue(20.0)
        check(s.sun_grade.saturation == -40.0 and s.sun_grade.contrast == 20.0,
              "the master sliders must edit the grade of all Suns")
        proxy_mean = float(editor._bg_proxy.mean())
        editor.grade_background.rows["exposure"].setValue(-1.0)
        pump(300)
        check(s.background_grade.exposure == -1.0 and editor._bg_graded is not None
              and abs(float(editor._bg_graded.mean()) - proxy_mean * 0.5) < 0.01,
              "the background grade must be applied to the preview")
        cached = editor._bg_graded
        editor.grade_master.rows["tint"].setValue(10.0)
        pump(300)
        check(editor._bg_graded is cached, "a Sun-only change must not re-grade the background")
        editor.grade_background.reset()
        check(s.background_grade == ColorGrade(), "the reset must clear the background grade")
        editor.combo_blend.setCurrentIndex(editor.combo_blend.findData("normal"))
        check(s.blend_mode == "normal", "the blend mode must be applied")
        editor.chk_ecliptic.setChecked(True)
        pump(300)
        check(bool(editor.canvas._overlay.get("ecliptic")), "the ecliptic overlay must be drawn")

        # Full-resolution export.
        out_path = os.path.join(tmpdir, "časosběr_kompozit.tif")
        task = editor.start_export(out_path)
        check(task is not None, "the export must start")
        pump(30000, lambda: not editor.busy())
        written = imread_unicode(out_path, cv2.IMREAD_UNCHANGED)
        check(written is not None and written.dtype == np.uint16 and written.shape[:2] == (1000, 1600),
              "the composite must be exported as a full-size 16-bit TIFF")

        # The composite is part of the project and comes back on reopening.
        project_path = os.path.join(tmpdir, "kompozit_gui.ahdrproj")
        check(window._write_project(project_path), "a composite-only project must save")
        window.new_project()
        check(not window._composite_settings.frames, "a new project must clear the composite")
        check(window._load_project_file(project_path, remember_path=True),
              "a composite-only project must open")
        check(len(window._composite_settings.frames) == 3
              and window._composite_settings.horizon is not None
              and abs(window._composite_settings.frames[1].grade.exposure - 0.5) < 1e-6
              and window._composite_settings.sun_grade.saturation == -40.0,
              "the composite must be restored from the project, grades included")

        window.open_eclipse_composite()
        reopened = window._composite_window
        pump(15000, lambda: reopened._bg_proxy is not None and not reopened.busy()
             and len(reopened._cutouts) == 3)
        check(reopened._bg_proxy is not None and len(reopened._cutouts) == 3,
              "a reopened project must reload its background and cut-outs")

        # Closing with a load in flight must not leave a thread running.
        reopened._cutouts.clear()
        reopened._reload_inputs()
        window.close()
        check(not any(t.isRunning() for t in reopened._tasks), "closing must stop every task")
        print(f"   editor calibrated, placed 3 Suns, exported {written.shape if written is not None else None}")
    finally:
        ecw.QMessageBox = original_box


# ----------------------------------------------------------------- GUI tests

def test_gui(paths):
    section("7. GUI stability — the scenarios that used to crash")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QCoreApplication, QEventLoop

    # The QApplication must stay referenced: if it is garbage-collected, every
    # subsequent widget construction aborts with "Must construct a QApplication".
    app = QApplication.instance() or QApplication([])

    from gui.main_window import MainWindow, StackingWorker
    from gui.controls_panel import ControlsPanel
    from gui.image_viewer import ImageViewerContainer
    from gui.exposure_list_widget import ExposureListWidget
    from gui.manual_align_dialog import ManualAlignDialog
    check(app is not None and all(cls is not None for cls in
              (MainWindow, StackingWorker, ControlsPanel, ImageViewerContainer,
               ExposureListWidget, ManualAlignDialog)),
          "every GUI module must import")
    print("   all GUI modules import cleanly")

    # session_persistence off: a test must not read or overwrite the real
    # user's auto-saved session.
    window = MainWindow(session_persistence=False)
    window.show()
    window.exposure_list.load_files(paths)
    check(len(window.exposure_list.items) == 9, "GUI must load all 9 frames")
    window._setup_initial_preview()

    def pump(timeout_ms: int = 400, until=None) -> bool:
        """
        Processes Qt events for real wall-clock time.

        processEvents returns immediately when the queue is empty, so a plain
        loop count waits for nothing — the deadline has to be measured in time.
        """
        deadline = time.monotonic() + timeout_ms / 1000.0
        while time.monotonic() < deadline:
            QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            if until is not None and until():
                return True
            time.sleep(0.01)
        return until() if until is not None else True

    # (a) Rapid ROI dragging: previously one QThread per mouse-move event, each
    #     decoding the whole bracket. The debounce must collapse them into one.
    window.viewer_container.btn_roi_toggle.setChecked(True)
    for i in range(60):
        window.viewer_container.viewer.set_roi_center(150 + i, 150 + i, emit_signal=True)
        QCoreApplication.processEvents()
    live = 1 if (window._worker and window._worker.isRunning()) else 0
    check(live + len(window._retired_workers) <= 3,
          f"debounce must collapse 60 ROI drags (workers alive: {live + len(window._retired_workers)})")
    print(f"   60 rapid ROI drags -> {live + len(window._retired_workers)} worker(s)")

    pump(3000)
    check(window._base_merged_bgr is not None, "ROI stacking must produce a result")
    if window._base_merged_bgr is not None:
        check(bool(np.isfinite(window._base_merged_bgr).all()), "ROI result must be finite")

    # (b) Live slider adjustments must never raise on the GUI thread.
    for value in (0.0, 0.5, 1.0, 2.0):
        window.controls.slider_coronal_boost.setValue(value)
        window.controls.slider_gamma.setValue(max(0.4, value))
        QCoreApplication.processEvents()
    check(True, "")
    print("   live slider sweep completed without exceptions")

    # (c) Full-scene stacking after leaving ROI mode.
    window.viewer_container.btn_roi_toggle.setChecked(False)
    pump(4000)
    check(window._base_merged_bgr is not None, "full-scene stacking must produce a result")

    # (d) Repeatedly cancelling a running worker must not terminate a thread
    #     mid-allocation (the old code called QThread.terminate()).
    for _ in range(8):
        window._run_stacking()
        QCoreApplication.processEvents()
    pump(4000)
    check(all(not w.isRunning() for w in window._retired_workers),
          "all retired workers must unwind cleanly")
    print(f"   8 back-to-back cancellations survived; {len(window._retired_workers)} parked")

    # (e) A worker whose files vanish must fail gracefully, not crash.
    ghost = list(window.exposure_list.items)
    for it in ghost:
        it.filepath = it.filepath + ".gone"
    worker = StackingWorker(ghost, window.controls.get_settings(), scale=0.25)
    errors = []
    worker.failed.connect(errors.append)
    worker.start()
    worker.wait(5000)
    # The failure signal is queued to this thread; it only arrives once the
    # event loop runs, so waiting on the thread alone is not enough.
    pump(1000, until=lambda: len(errors) > 0)
    check(len(errors) == 1 and "Nelze načíst" in errors[0],
          "a missing file must produce one clean error")

    # (f) The alignment dialog must load off-thread and restore on cancel.
    for it, original in zip(window.exposure_list.items, paths):
        it.filepath = original
    items = window.exposure_list.get_active_items()
    dialog = ManualAlignDialog(items, parent=window)
    dialog.show()
    dialog._start_loading()
    pump(20000, until=lambda: dialog._frames_loaded >= len(items))
    check(dialog._frames_loaded == len(items),
          f"the dialog must load every frame (got {dialog._frames_loaded}/{len(items)})")
    # Auto-alignment may already have written shifts onto these items, so
    # compare against what the shift actually was when the dialog opened.
    before = items[dialog.current_idx].shift_x
    dialog._nudge(5.0, -3.0)
    check(abs(items[dialog.current_idx].shift_x - (before + 5.0)) < 1e-6,
          "nudging must update the shift")
    dialog.reject()
    check(abs(items[dialog.current_idx].shift_x - before) < 1e-6,
          "cancel must restore the original shifts")
    check(dialog._loader is None or not dialog._loader.isRunning(),
          "closing the dialog must stop its loader thread")

    # (f2) Window geometry must fit real laptop desktops. A 15" 1080p panel at
    #      Windows' default 125 % scaling leaves about 1536x826 usable; anything
    #      taller opens with its action buttons below the screen edge.
    LAPTOP_DESKTOPS = [
        ("1920x1080 @100%", 1920, 1032),
        ("1920x1080 @125%", 1536, 826),
        ("1920x1080 @150%", 1280, 688),
        ("1600x900 @100%", 1600, 852),
        ("1366x768 @100%", 1366, 728),
    ]
    win_min = window.minimumSizeHint()
    dlg = ManualAlignDialog(window.exposure_list.get_active_items(), parent=window)
    dlg_min = dlg.minimumSizeHint()
    for label, dw, dh in LAPTOP_DESKTOPS:
        check(win_min.width() <= dw and win_min.height() <= dh,
              f"main window minimum {win_min.width()}x{win_min.height()} must fit {label}")
        check(dlg_min.width() <= dw and dlg_min.height() <= dh,
              f"align dialog minimum {dlg_min.width()}x{dlg_min.height()} must fit {label}")
    print(f"   minimums: window {win_min.width()}x{win_min.height()}, "
          f"dialog {dlg_min.width()}x{dlg_min.height()} — fit all 5 laptop desktops")

    # The Apply button must stay inside the dialog even when squeezed hard.
    dlg.resize(900, 480)
    QCoreApplication.processEvents()
    btn = dlg.btn_apply
    bottom_right = btn.mapTo(dlg, btn.rect().bottomRight())
    top_left = btn.mapTo(dlg, btn.rect().topLeft())
    check(0 <= top_left.x() and 0 <= top_left.y()
          and bottom_right.x() <= dlg.width() and bottom_right.y() <= dlg.height(),
          f"Apply button must stay on-screen when the dialog is squeezed "
          f"(button at {top_left.x()},{top_left.y()}-{bottom_right.x()},{bottom_right.y()} "
          f"in a {dlg.width()}x{dlg.height()} dialog)")
    dlg.reject()

    # (f3) Output crop: same rectangle on every exposure, preview and export.
    from gui.main_window import _apply_crop, StackingWorker as SW

    probe = np.arange(120 * 200 * 3, dtype=np.uint8).reshape(120, 200, 3)
    cropped = _apply_crop(probe, (40, 20, 60, 50), scale=1.0)
    check(cropped.shape[:2] == (50, 60), "crop must produce the requested size")
    check(np.array_equal(cropped, probe[20:70, 40:100]),
          "crop must take the requested region, not an offset one")
    half = _apply_crop(probe, (40, 20, 60, 50), scale=0.5)
    check(half.shape[:2] == (25, 30), "crop must scale with the working proxy")
    check(_apply_crop(probe, None).shape == probe.shape, "no crop rect is a pass-through")
    huge = _apply_crop(probe, (150, 100, 500, 500), scale=1.0)
    check(huge.shape[0] > 0 and huge.shape[1] > 0,
          "a crop reaching past the edge must clamp, not produce an empty image")

    window.controls.chk_crop.setChecked(True)
    pump(300)
    seeded = window.controls.get_crop_rect()
    check(seeded is not None and seeded[2] == 400 and seeded[3] == 400,
          f"enabling the crop must seed it from the real frame size (got {seeded})")

    for key, value in (("x", 80), ("y", 60), ("w", 240), ("h", 180)):
        window.controls.spin_crop[key].setValue(value)
    pump(200)
    check(window._crop_rect == (80, 60, 240, 180),
          f"numeric crop edits must reach the window (got {window._crop_rect})")

    window._run_stacking()
    pump(15000, until=lambda: window._base_merged_bgr is not None
         and window._base_merged_bgr.shape[:2] == (45, 60))
    merged_shape = window._base_merged_bgr.shape[:2] if window._base_merged_bgr is not None else None
    check(merged_shape == (45, 60),
          f"the preview must be the cropped region at proxy scale (got {merged_shape})")

    # Crop selection state machine: turning the mode on shows the full frame
    # with a marker; drawing exits the mode; the finished preview drops the
    # marker because the scene has become the crop itself.
    viewer = window.viewer_container.viewer
    window.controls.btn_crop_select.setChecked(True)
    pump(600)
    check(window.controls.btn_crop_select.isChecked(),
          "seeding defaults must not cancel the selection mode that triggered it")
    check(viewer.get_crop_rect() is not None,
          "the crop marker must be visible while composing")
    check(viewer._scene_size() == (400, 400),
          f"composing must show the full uncropped frame (got {viewer._scene_size()})")

    window._on_crop_drawn(60, 40, 260, 200)
    check(not window.controls.btn_crop_select.isChecked(),
          "drawing a crop must leave selection mode")
    check(viewer.get_crop_rect() == (60, 40, 260, 200),
          "the drawn rectangle must be shown")
    pump(15000, until=lambda: viewer._scene_size() == (260, 200))
    check(viewer._scene_size() == (260, 200),
          f"the finished preview scene must be the crop (got {viewer._scene_size()})")
    check(viewer.get_crop_rect() is None,
          "the marker must be dropped once the preview is itself the crop")

    # Drawing on the image must feed straight back into the controls.
    window._on_crop_drawn(10, 20, 300, 200)
    check(window.controls.get_crop_rect() == (10, 20, 300, 200),
          "a crop drawn on the image must update the numeric fields")
    check(window.viewer_container.viewer.get_crop_rect() == (10, 20, 300, 200),
          "the viewer must show the crop it was given")

    # A cropped full-resolution export must come out at exactly the crop size.
    from gui.main_window import FullResExportWorker as FEW
    crop_out = os.path.join(os.path.dirname(paths[0]), "crop_export.tif")
    crop_worker = FEW(window.exposure_list.get_active_items(),
                      window.controls.get_settings(), crop_out,
                      export_scale=1.0, crop_rect=(10, 20, 300, 200))
    crop_errors, crop_done = [], []
    crop_worker.failed.connect(crop_errors.append)
    crop_worker.finished_success.connect(crop_done.append)
    crop_worker.start()
    crop_worker.wait(120000)
    pump(2000, until=lambda: bool(crop_done or crop_errors))
    check(not crop_errors, f"cropped export must not fail: {crop_errors}")
    if os.path.exists(crop_out):
        written = imread_unicode(crop_out, cv2.IMREAD_UNCHANGED)
        check(written is not None and written.shape[:2] == (200, 300),
              f"cropped export must be exactly the crop size "
              f"(got {None if written is None else written.shape[:2]})")
        print(f"   cropped export: {written.shape[1]}x{written.shape[0]} px at full resolution")

    # Alignment must run before the crop, otherwise the warp drags replicated
    # border pixels in along the crop edge.
    shifted_items = list(window.exposure_list.get_active_items())
    for it in shifted_items:
        it.shift_x, it.shift_y = 12.0, 8.0
    order_worker = SW(shifted_items, window.controls.get_settings(),
                      scale=1.0, crop_rect=(120, 120, 160, 160))
    order_results = []
    order_worker.finished_success.connect(lambda *a: order_results.append(a[0]))
    order_worker.start()
    order_worker.wait(60000)
    pump(2000, until=lambda: bool(order_results))
    check(bool(order_results) and order_results[0].shape[:2] == (160, 160),
          "align-then-crop must still yield exactly the crop size")
    for it in shifted_items:
        it.shift_x, it.shift_y = 0.0, 0.0

    window.controls.chk_crop.setChecked(False)
    pump(300)
    check(window._crop_rect is None, "disabling the crop must clear it everywhere")

    # (f4) A whole session must survive save -> quit -> reopen.
    session_dir = os.path.join(os.path.dirname(paths[0]), "relace")
    os.makedirs(session_dir, exist_ok=True)
    session_project = os.path.join(session_dir, "relace.ahdrproj")

    window.exposure_list.items[0].shift_x = 6.5
    window.exposure_list.items[0].shift_y = -2.5
    window.exposure_list.items[1].is_valid = False
    window.controls.combo_align.setCurrentIndex(1)
    window.controls.slider_gamma.setValue(1.65)
    window.controls.slider_coronal_boost.setValue(0.85)
    window.controls.chk_crop.setChecked(True)
    pump(400)
    for key, value in (("x", 30), ("y", 25), ("w", 260), ("h", 210)):
        window.controls.spin_crop[key].setValue(value)
    pump(300)

    saved_settings = dict(window.controls.get_settings())
    saved_frames = {os.path.basename(it.filepath):
                    (round(it.shift_x, 1), round(it.shift_y, 1), it.is_valid)
                    for it in window.exposure_list.items}

    check(window._write_project(session_project), "the session must save")
    check(os.path.isfile(session_project), "the project file must exist")
    check(window._project_path == session_project, "the project path must be remembered")
    check(os.path.basename(session_project) in window.windowTitle(),
          "the title bar must name the open project")

    # A genuinely fresh window, as if the app had been restarted.
    reopened = MainWindow(session_persistence=False)
    reopened.show()
    check(reopened._load_project_file(session_project, remember_path=True),
          "the project must reopen")
    pump(1500)

    restored_settings = dict(reopened.controls.get_settings())
    drifted = [k for k, v in saved_settings.items() if restored_settings.get(k) != v]
    check(not drifted, f"every setting must survive the round trip (drifted: {drifted})")

    restored_frames = {os.path.basename(it.filepath):
                       (round(it.shift_x, 1), round(it.shift_y, 1), it.is_valid)
                       for it in reopened.exposure_list.items}
    check(restored_frames == saved_frames,
          "alignment shifts and exclusions must be restored onto the right frames")
    check(reopened._crop_rect == (30, 25, 260, 210),
          f"the crop must be restored (got {reopened._crop_rect})")
    print(f"   session round-trip: {len(restored_frames)} frames, "
          f"{len(saved_settings)} settings, crop and alignment all restored")

    # The automatic end-of-session snapshot must be reloadable too. It is
    # redirected into the temp folder so the real user's file is untouched.
    snapshot = os.path.join(session_dir, "auto_session.ahdrproj")
    reopened._session_file = lambda: snapshot
    reopened._session_persistence = True
    reopened._autosave_session()
    check(os.path.isfile(reopened._session_file()),
          "closing must leave an auto-restorable session snapshot")
    reopened.exposure_list.clear_all()
    reopened.restore_last_session()
    pump(1200)
    check(len(reopened.exposure_list.items) == len(saved_frames),
          "the auto-saved session must restore every frame")
    check(reopened._project_path is None,
          "restoring the auto-snapshot must not make it the current project, "
          "so Ctrl+S cannot overwrite it")

    # Opening a corrupt project must report, not crash, and leave state intact.
    broken_project = os.path.join(session_dir, "rozbity.ahdrproj")
    with open(broken_project, "w", encoding="utf-8") as f:
        f.write("{ tohle rozhodne neni projekt")
    before_count = len(reopened.exposure_list.items)
    check(reopened._load_project_file(broken_project, remember_path=True, quiet=True) is False,
          "a corrupt project must fail cleanly")
    check(len(reopened.exposure_list.items) == before_count,
          "a failed load must not destroy the open session")

    reopened.controls.chk_crop.setChecked(False)
    reopened.close()
    window.controls.chk_crop.setChecked(False)
    pump(300)

    # (g) End-to-end export through the real worker, including alignment,
    #     fusion, post-processing and the 16-bit write.
    from gui.main_window import FullResExportWorker
    out_path = os.path.join(os.path.dirname(paths[0]), "export test — výsledek.tif")
    settings = window.controls.get_settings()
    settings.update({'coronal_boost': 0.4, 'denoise': 0.2, 'gamma': 1.2})
    exporter = FullResExportWorker(window.exposure_list.get_active_items(),
                                   settings, out_path, export_scale=1.0)
    export_errors, export_done = [], []
    exporter.failed.connect(export_errors.append)
    exporter.finished_success.connect(export_done.append)
    exporter.start()
    exporter.wait(120000)
    pump(2000, until=lambda: bool(export_done or export_errors))
    check(not export_errors, f"full export must not fail: {export_errors}")
    check(len(export_done) == 1 and os.path.exists(out_path), "export must write the file")
    if os.path.exists(out_path):
        written = imread_unicode(out_path, cv2.IMREAD_UNCHANGED)
        check(written is not None and written.dtype == np.uint16,
              "exported TIFF must be 16-bit")
        check(written is not None and written.shape[:2] == (400, 400),
              "exported TIFF must be at full resolution")
        print(f"   exported {written.shape} {written.dtype} to a Unicode filename")

    # (h) A memory-constrained export must be offered at a reduced scale
    #     rather than being attempted and killed.
    scale = window._choose_export_scale(window.exposure_list.get_active_items())
    check(scale == 1.0, "a small stack must export at full resolution without prompting")

    # (i) Closing with work in flight must not destroy a running QThread.
    window._run_stacking()
    pump(150)
    window.close()
    check(all(not w.isRunning() for w in
              [window._worker, window._sun_worker] + window._retired_workers if w),
          "closing must wait for every worker")
    print("   window closed cleanly with work in flight")


# ------------------------------------------------------------------- Runner

def run_all_tests() -> int:
    print("[*] Astro HDR Stacker — comprehensive test suite")

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_synthetic_eclipse_exposures(tmpdir, 9)
        check(len(paths) == 9, "fixture generation")

        items = test_exposure_analysis(tmpdir, paths)
        _images, aligned = test_detection_and_alignment(items)
        test_static_light_alignment()
        fusion, hdr = test_merging(aligned, items)
        enhanced = test_postprocessing(fusion)
        test_export(tmpdir, enhanced, hdr)
        test_image_cache(paths)
        test_projects(tmpdir, paths)
        test_solar_position()
        scene = test_eclipse_composite(tmpdir)

        try:
            test_gui(paths)
        except Exception:
            traceback.print_exc()
            _FAILURES.append("GUI test suite raised an exception")

        try:
            test_composite_gui(tmpdir, scene)
        except Exception:
            traceback.print_exc()
            _FAILURES.append("composite GUI test raised an exception")

    print("\n" + "=" * 64)
    if _FAILURES:
        print(f">>> {len(_FAILURES)} FAILURE(S), {_PASSES} checks passed:")
        for f in _FAILURES:
            print(f"    - {f}")
        return 1

    print(f">>> ALL {_PASSES} CHECKS PASSED <<<")
    return 0


if __name__ == "__main__":
    sys.exit(run_all_tests())
