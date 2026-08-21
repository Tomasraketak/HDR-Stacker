"""
Image Alignment Module using Median Threshold Bitmap (MTB) alignment.
Specially suited for exposures of vastly different brightness (such as solar eclipse sequences).
"""

from typing import List, Callable, Optional
import cv2
import numpy as np


class ImageAligner:
    """
    Aligns multiple exposures using OpenCV's AlignMTB (Median Threshold Bitmap).
    """

    def __init__(self, max_bits: int = 5, exclude_range: int = 4, cut: bool = True):
        """
        :param max_bits: Number of pyramid levels for coarse-to-fine shift discovery.
        :param exclude_range: Range around median to exclude from thresholding to avoid noise.
        :param cut: Whether to crop aligned images to valid common area or keep padded size.
        """
        self.max_bits = max_bits
        self.exclude_range = exclude_range
        self.cut = cut

    def align(
        self,
        images: List[np.ndarray],
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> List[np.ndarray]:
        """
        Aligns a sequence of BGR uint8 images to the reference exposure.
        """
        if len(images) <= 1:
            return images

        if progress_callback:
            progress_callback(10, "Inicializace zarovnání (MTB)...")

        try:
            align_mtb = cv2.createAlignMTB(
                max_bits=self.max_bits,
                exclude_range=self.exclude_range,
                cut=self.cut
            )

            if progress_callback:
                progress_callback(30, f"Výpočet posunů a zarovnání {len(images)} snímků...")

            dst = [np.empty_like(img) for img in images]
            align_mtb.process(images, dst)

            if progress_callback:
                progress_callback(100, "Zarovnání dokončeno.")

            return dst

        except Exception as e:
            # Fallback shift calculation per pair
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
                    progress_callback(100, "Zarovnání dokončeno.")
                return aligned
            except Exception as e2:
                print(f"Alignment error: {e2}, falling back to unaligned images.")
                if progress_callback:
                    progress_callback(100, f"Varování: Zarovnání selhalo ({e2}), použity původní snímky.")
                return images
