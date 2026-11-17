"""
ClaimLens — TensorFlow Serving Client
Sends batched image arrays to TF Serving REST API and parses predictions.

TF Serving must be running separately — see docker-compose.yml.
CHANGE: model input/output names if you retrain with different signatures.
"""
import numpy as np
import httpx
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.config import get_settings

logger    = logging.getLogger(__name__)
settings  = get_settings()


# ── Damage categories matching CarDD + VeHIDE ────────────────────────────────
# CHANGE: update this list if you trained on different classes
DAMAGE_CLASSES = [
    "dent",
    "scratch",
    "broken_glass",
    "lost_parts",
    "punctured",
    "torn",
    "broken_lights",
    "paint_damage",
]

# Severity mapping for triage decision
DAMAGE_SEVERITY = {
    "broken_glass":  "severe",
    "lost_parts":    "severe",
    "dent":          "moderate",
    "broken_lights": "moderate",
    "torn":          "moderate",
    "scratch":       "minor",
    "paint_damage":  "minor",
    "punctured":     "moderate",
}

# Rough repair cost ranges (NPR) — CHANGE to match local garage prices
# Format: (min_cost, max_cost)
REPAIR_COSTS_NPR = {
    "dent":          (8_000,  25_000),
    "scratch":       (2_000,   8_500),
    "broken_glass":  (12_000, 35_000),
    "lost_parts":    (15_000, 80_000),
    "punctured":     (5_000,  18_000),
    "torn":          (10_000, 40_000),
    "broken_lights": (6_000,  22_000),
    "paint_damage":  (3_000,  12_000),
}

VEHICLE_CLASSES = [
    "sedan", "suv", "pickup_truck", "hatchback",
    "van", "bus", "motorcycle", "truck",
]


@dataclass
class DamageDetection:
    damage_type: str
    confidence: float
    severity: str
    location_hint: str   # e.g. "front", "rear", "side"
    repair_cost_min: int
    repair_cost_max: int


@dataclass
class ModelPredictions:
    damage_detections: List[DamageDetection]
    vehicle_type: str
    vehicle_confidence: float
    total_cost_estimate: float
    cost_breakdown: Dict[str, float]


class TFServingClient:
    """
    Calls TensorFlow Serving REST API endpoints.

    TF Serving runs models at:
      http://{host}:{port}/v1/models/{model_name}:predict

    CHANGE TF_SERVING_HOST and ports in .env if your TF Serving
    is running on a different machine or port.
    """

    def __init__(self):
        self.base_url = settings.tf_serving_rest_url
        self.timeout  = 30.0   # seconds — increase for slow GPU servers

    async def predict_damage(
        self, images: List[np.ndarray]
    ) -> List[DamageDetection]:
        """
        Run damage detection on a batch of frames.
        Returns the top damage detections across all frames.

        CHANGE: 'serving_default' is the default signature key.
        If you exported with a custom signature, update it here.
        """
        if not images:
            return []

        # Stack images into batch (N, 300, 300, 3)
        batch = np.stack(images, axis=0).tolist()

        payload = {
            "signature_name": "serving_default",
            "instances": batch
        }

        try:
            response = await self._post(
                f"/v1/models/{settings.damage_model_name}:predict",
                payload
            )
            raw_preds = response.get("predictions", [])
            return self._parse_damage_predictions(raw_preds)

        except Exception as e:
            logger.warning(f"TF Serving damage call failed: {e}. Using mock.")
            # ── DEVELOPMENT FALLBACK ─────────────────────────────────────
            # When TF Serving is not yet running, return mock predictions
            # so the rest of the pipeline can be tested end-to-end.
            # REMOVE this fallback once TF Serving is deployed.
            return self._mock_damage_predictions()

    async def predict_vehicle_type(
        self, images: List[np.ndarray]
    ) -> tuple[str, float]:
        """Classify vehicle type (sedan, SUV, pickup, etc.)"""
        if not images:
            return "unknown", 0.0

        # Use sharpest frame only for vehicle classification
        batch = [images[0].tolist()]
        payload = {"signature_name": "serving_default", "instances": batch}

        try:
            response = await self._post(
                f"/v1/models/{settings.vehicle_model_name}:predict",
                payload
            )
            preds    = response.get("predictions", [[]])[0]
            best_idx = int(np.argmax(preds))
            return VEHICLE_CLASSES[best_idx], float(preds[best_idx])

        except Exception as e:
            logger.warning(f"TF Serving vehicle call failed: {e}. Using mock.")
            return "suv", 0.91   # mock fallback

    def build_cost_estimate(
        self, detections: List[DamageDetection]
    ) -> tuple[float, Dict[str, float]]:
        """
        Combine per-damage cost ranges into a total estimate.
        Uses midpoint of range + labour overhead.

        CHANGE: labour_rate_per_hour to match local garage rates.
        CHANGE: labour_hours_per_damage to calibrate to your market.
        """
        labour_rate_per_hour  = 750    # NPR/hour — CHANGE to local rate
        labour_hours_per_damage = 1.5

        breakdown: Dict[str, float] = {}
        parts_total = 0.0

        for det in detections:
            mn, mx  = REPAIR_COSTS_NPR.get(det.damage_type, (5000, 20000))
            mid     = (mn + mx) / 2
            # Scale by severity
            if det.severity == "severe":
                cost = mx * 0.85
            elif det.severity == "moderate":
                cost = mid
            else:
                cost = mn * 1.3

            breakdown[det.damage_type] = round(cost)
            parts_total += cost

        labour = len(detections) * labour_hours_per_damage * labour_rate_per_hour
        breakdown["labour"] = round(labour)
        total = parts_total + labour

        return round(total), breakdown

    # ── Private helpers ────────────────────────────────────────────────────

    async def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def _parse_damage_predictions(
        self, raw_preds: List[List[float]]
    ) -> List[DamageDetection]:
        """
        Aggregate predictions across all frames.
        For each class, take the max confidence seen in any frame.
        Return only detections above CONFIDENCE_THRESHOLD.

        CHANGE: CONFIDENCE_THRESHOLD to trade off precision vs recall.
        0.45 = more detections (higher recall, more false positives)
        0.70 = fewer detections (higher precision, may miss damage)
        """
        CONFIDENCE_THRESHOLD = 0.50   # CHANGE this threshold

        if not raw_preds:
            return self._mock_damage_predictions()

        # max confidence per class across all frames
        class_max = np.max(np.array(raw_preds), axis=0)
        detections = []

        for idx, conf in enumerate(class_max):
            if conf >= CONFIDENCE_THRESHOLD and idx < len(DAMAGE_CLASSES):
                dtype  = DAMAGE_CLASSES[idx]
                sev    = DAMAGE_SEVERITY.get(dtype, "minor")
                mn, mx = REPAIR_COSTS_NPR.get(dtype, (5000, 20000))
                detections.append(DamageDetection(
                    damage_type    = dtype,
                    confidence     = float(conf),
                    severity       = sev,
                    location_hint  = self._infer_location(dtype),
                    repair_cost_min= mn,
                    repair_cost_max= mx,
                ))

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _infer_location(self, damage_type: str) -> str:
        """
        Heuristic location hint — replace with proper spatial detection
        once you add bounding box output to your model.
        """
        location_map = {
            "broken_glass":  "windscreen/window",
            "broken_lights": "front/rear lights",
            "dent":          "body panel",
            "scratch":       "body panel",
        }
        return location_map.get(damage_type, "vehicle body")

    def _mock_damage_predictions(self) -> List[DamageDetection]:
        """
        DEVELOPMENT ONLY — remove once TF Serving is live.
        Returns realistic-looking predictions for UI testing.
        """
        return [
            DamageDetection("dent",          0.94, "severe",   "front bumper",  8000,  25000),
            DamageDetection("scratch",        0.88, "minor",    "rear door",     2000,   8500),
            DamageDetection("broken_lights",  0.91, "moderate", "front left",    6000,  22000),
            DamageDetection("paint_damage",   0.82, "minor",    "front door",    3000,  12000),
        ]
