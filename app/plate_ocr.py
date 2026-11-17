"""
ClaimLens — Number Plate OCR Engine
Detects and reads vehicle registration plates from extracted frames.
Uses EasyOCR as primary (better for Devanagari/non-Latin scripts)
and Tesseract as fallback.

CHANGE: PLATE_REGION_RATIOS if your local plates appear in different
positions (bottom-third is standard for most countries).
"""
import cv2
import numpy as np
import re
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PlateResult:
    plate_text: str          # e.g. "BA1CHA2345"
    plate_formatted: str     # e.g. "BA 1 CHA 2345"
    confidence: float        # 0.0 – 1.0
    method: str              # "easyocr" | "tesseract" | "none"


class PlateOCR:
    """
    CHANGE: NEPAL_PLATE_PATTERN if you are in a different country.
    Nepal format: [Province][Zone][Alpha][Number]
    e.g. BA 1 CHA 2345, GA 1 JA 1234

    Other common formats to adapt:
    - India:  [State][District][Alpha2][4digits] → [A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}
    - UK:     [2letters][2digits][space][3letters] → [A-Z]{2}[0-9]{2} [A-Z]{3}
    """

    # ── CHANGE this regex to your country's plate format ──────────────────
    PLATE_PATTERN = re.compile(
        r'([A-Z]{1,2})\s*(\d{1,2})\s*([A-Z]{1,4})\s*(\d{1,6})',
        re.IGNORECASE
    )

    # Where to look for the plate in the frame (fraction of image height)
    # Bottom 25% of image usually contains the plate
    PLATE_REGION_LOWER = 0.55   # CHANGE if plates appear higher in frame
    PLATE_REGION_UPPER = 0.95

    def __init__(self):
        self._reader = None   # lazy-load EasyOCR (heavy import)

    def read_plate_from_frames(
        self, frames: List[np.ndarray]
    ) -> PlateResult:
        """
        Try to read a plate from multiple frames.
        Returns the highest-confidence result found.
        """
        best: Optional[PlateResult] = None

        for frame in frames:
            result = self._read_single_frame(frame)
            if result and (best is None or result.confidence > best.confidence):
                best = result
            if best and best.confidence > 0.85:
                break   # good enough — stop early

        return best or PlateResult("UNKNOWN", "UNKNOWN", 0.0, "none")

    def verify_plate_matches_policy(
        self, detected: str, policy_plate: str
    ) -> Tuple[bool, float]:
        """
        Check if detected plate matches the registered policy plate.
        Returns (match: bool, similarity: float 0–1).
        Uses fuzzy matching to handle OCR errors (O↔0, I↔1, B↔8).
        """
        clean_det    = self._normalise(detected)
        clean_policy = self._normalise(policy_plate)

        if clean_det == clean_policy:
            return True, 1.0

        # Character-level similarity (handle common OCR errors)
        common = sum(a == b for a, b in zip(clean_det, clean_policy))
        length = max(len(clean_det), len(clean_policy))
        sim    = common / length if length > 0 else 0.0

        return sim >= 0.85, sim

    # ── Private helpers ────────────────────────────────────────────────────

    def _read_single_frame(self, frame_bgr: np.ndarray) -> Optional[PlateResult]:
        # Crop to plate region
        h = frame_bgr.shape[0]
        roi = frame_bgr[int(h * self.PLATE_REGION_LOWER):int(h * self.PLATE_REGION_UPPER), :]

        # Preprocess for OCR
        processed = self._preprocess(roi)

        # Try EasyOCR first
        result = self._try_easyocr(processed)
        if result and result.confidence > 0.5:
            return result

        # Fallback to Tesseract
        return self._try_tesseract(processed)

    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Enhance plate region for better OCR accuracy."""
        gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Upscale — tesseract works better at higher resolution
        upscaled  = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        # Adaptive threshold to handle uneven lighting
        thresh    = cv2.adaptiveThreshold(
            upscaled, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        # Denoise
        denoised  = cv2.fastNlMeansDenoising(thresh, h=10)
        return denoised

    def _try_easyocr(self, img: np.ndarray) -> Optional[PlateResult]:
        try:
            if self._reader is None:
                import easyocr
                # CHANGE: add 'ne' (Nepali) if plates include Devanagari script
                self._reader = easyocr.Reader(['en'], gpu=False)

            results = self._reader.readtext(img, detail=1)
            for (_, text, conf) in results:
                parsed = self._parse_plate(text)
                if parsed:
                    return PlateResult(
                        plate_text      = parsed["raw"],
                        plate_formatted = parsed["formatted"],
                        confidence      = float(conf),
                        method          = "easyocr"
                    )
        except Exception as e:
            logger.debug(f"EasyOCR failed: {e}")
        return None

    def _try_tesseract(self, img: np.ndarray) -> Optional[PlateResult]:
        try:
            import pytesseract
            from app.config import get_settings
            pytesseract.pytesseract.tesseract_cmd = get_settings().tesseract_cmd

            # PSM 8 = single word mode, good for plates
            config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text   = pytesseract.image_to_string(img, config=config).strip()
            parsed = self._parse_plate(text)
            if parsed:
                return PlateResult(
                    plate_text      = parsed["raw"],
                    plate_formatted = parsed["formatted"],
                    confidence      = 0.65,   # Tesseract doesn't give per-result confidence
                    method          = "tesseract"
                )
        except Exception as e:
            logger.debug(f"Tesseract failed: {e}")
        return None

    def _parse_plate(self, text: str) -> Optional[dict]:
        """Match text against Nepal plate regex and format it."""
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        m     = self.PLATE_PATTERN.search(clean)
        if m:
            parts     = [g for g in m.groups() if g]
            raw       = "".join(parts)
            formatted = " ".join(parts)
            return {"raw": raw, "formatted": formatted}
        return None

    def _normalise(self, plate: str) -> str:
        """Remove spaces, uppercase, fix common OCR confusions."""
        s = re.sub(r'[^A-Z0-9]', '', plate.upper())
        # Common OCR errors
        s = s.replace('O', '0').replace('I', '1').replace('B', '8')
        return s
