"""
Comprehensive test suite for Astro HDR Stacker.

Covers the numerical core (detection, alignment, fusion, tonemapping,
post-processing, export) and the stability scenarios that used to crash the
GUI: rapid ROI dragging, worker cancellation, and closing during work.

Run with:  python tests/test_stacker.py
"""

import os
import sys
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
)
from core.merger import HDRMerger, HDRMergeError, sanitize_exposure_times
from core.postprocess import (
    apply_postprocessing, save_image, build_tone_curve_lut,
    apply_denoise, imread_unicode,
)
from core.image_cache import ImageCache, available_memory_bytes

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

    window = MainWindow()
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
        fusion, hdr = test_merging(aligned, items)
        enhanced = test_postprocessing(fusion)
        test_export(tmpdir, enhanced, hdr)
        test_image_cache(paths)

        try:
            test_gui(paths)
        except Exception:
            traceback.print_exc()
            _FAILURES.append("GUI test suite raised an exception")

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
