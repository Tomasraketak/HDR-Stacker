"""
Bounded decoded-image cache and host memory introspection.

The interactive workflow re-stacks the bracket every time a ROI moves or a
setting changes. Re-decoding nine 24 MP JPEGs from disk each time costs seconds
and thrashes the allocator; caching the decoded frames turns that into a memcpy.
The cache is bounded in bytes so it can never be the thing that exhausts RAM.
"""

import os
import sys
import threading
from collections import OrderedDict
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from core.postprocess import imread_unicode
except ImportError:  # pragma: no cover
    from .postprocess import imread_unicode


def available_memory_bytes() -> Optional[int]:
    """
    Best-effort free physical memory, or None when it cannot be determined.

    Uses GlobalMemoryStatusEx on Windows and /proc/meminfo on Linux, so no
    third-party dependency is needed just to avoid an out-of-memory crash.
    """
    try:
        if sys.platform.startswith('win'):
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
            return None

        if hasattr(os, 'sysconf') and 'SC_AVPHYS_PAGES' in os.sysconf_names:
            return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
    except Exception:
        return None
    return None


def default_cache_budget_bytes() -> int:
    """A cache budget that leaves plenty of headroom for the fusion itself."""
    avail = available_memory_bytes()
    if avail is None:
        return 768 * 1024 * 1024
    # Never take more than a quarter of what is free, and stay within 256 MB..2 GB.
    return int(min(2048 * 1024 * 1024, max(256 * 1024 * 1024, avail * 0.25)))


class ImageCache:
    """
    Thread-safe LRU cache of decoded BGR frames, keyed by (path, mtime, scale).

    Returned arrays are treated as read-only by callers; anything that modifies
    a frame (alignment, cropping) must work on its own copy.
    """

    def __init__(self, budget_bytes: Optional[int] = None):
        self._lock = threading.RLock()
        self._store: "OrderedDict[Tuple[str, float, int], np.ndarray]" = OrderedDict()
        self._bytes = 0
        self._budget = budget_bytes if budget_bytes is not None else default_cache_budget_bytes()

    @property
    def budget_bytes(self) -> int:
        return self._budget

    def _key(self, path: str, scale: float) -> Tuple[str, float, int]:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        # Quantise the scale so 0.2500001 and 0.25 share an entry.
        return (path, mtime, int(round(scale * 10000)))

    def get(self, path: str, scale: float = 1.0) -> Optional[np.ndarray]:
        """
        Returns the decoded frame at the requested scale, decoding it if needed.

        Returns None when the file cannot be read or decoded — callers must
        handle that rather than assuming an array came back.
        """
        scale = float(np.clip(scale, 0.01, 1.0))
        key = self._key(path, scale)

        with self._lock:
            hit = self._store.get(key)
            if hit is not None:
                self._store.move_to_end(key)
                return hit

        # Decode outside the lock so parallel workers do not serialise on I/O.
        img = imread_unicode(path, cv2.IMREAD_COLOR)
        if img is None:
            return None

        if scale < 0.999:
            h, w = img.shape[:2]
            nw, nh = max(16, int(round(w * scale))), max(16, int(round(h * scale)))
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        img = np.ascontiguousarray(img)

        with self._lock:
            if key not in self._store:
                self._store[key] = img
                self._bytes += img.nbytes
                self._evict_locked()
        return img

    def _evict_locked(self):
        while self._bytes > self._budget and len(self._store) > 1:
            _, victim = self._store.popitem(last=False)
            self._bytes -= victim.nbytes

    def invalidate(self, path: Optional[str] = None):
        """Drops one file's entries, or the whole cache when path is None."""
        with self._lock:
            if path is None:
                self._store.clear()
                self._bytes = 0
                return
            for key in [k for k in self._store if k[0] == path]:
                self._bytes -= self._store.pop(key).nbytes

    def stats(self) -> Tuple[int, int, int]:
        """(entries, bytes_used, budget_bytes)"""
        with self._lock:
            return len(self._store), self._bytes, self._budget


# Process-wide cache shared by the preview worker, the export worker and the
# alignment dialog, so a frame is decoded at most once per scale.
GLOBAL_IMAGE_CACHE = ImageCache()
