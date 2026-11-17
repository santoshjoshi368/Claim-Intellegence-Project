"""
ClaimLens — Fraud Detection Engine
Runs multiple fraud signal checks and produces a composite fraud score.

CHANGE: weights in SIGNAL_WEIGHTS to tune sensitivity.
Higher weight = that signal contributes more to the overall score.
"""
import math
import hashlib
import logging
import imagehash
from PIL import Image
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FraudSignal:
    name: str
    passed: bool
    score_contribution: float   # 0.0 = clean, 1.0 = maximum risk
    detail: str


@dataclass
class FraudReport:
    overall_score: float            # 0.0 (clean) – 1.0 (high risk)
    risk_level: str                 # low | medium | high | critical
    passed: bool                    # True = claim proceeds normally
    signals: List[FraudSignal]
    triage_override: Optional[str]  # force adjuster/fraud_review if flagged
    flags: List[str]                # human-readable flag list


# ── Signal weights — must sum to 1.0 ────────────────────────────────────────
# CHANGE these to tune how much each check influences the score
SIGNAL_WEIGHTS = {
    "video_continuity":    0.25,   # most important — edited video = clear fraud
    "gps_consistency":     0.20,   # location mismatch
    "timestamp_validity":  0.15,   # claim filed long after incident
    "device_trust":        0.10,   # unregistered device
    "duplicate_image":     0.20,   # reused photos from another claim
    "damage_consistency":  0.10,   # damage pattern vs accident type
}


class FraudEngine:

    def __init__(self, db_session=None):
        self.db = db_session

    async def analyse(
        self,
        # Video signals
        has_cuts: bool,
        continuity_score: float,
        # Location signals
        gps_lat: Optional[float],
        gps_lng: Optional[float],
        declared_location: Optional[str],
        # Timing signals
        recorded_at: Optional[datetime],
        submitted_at: Optional[datetime],
        # Device signals
        device_id: Optional[str],
        policy_device_id: Optional[str],
        # Image signals
        frame_images: List[np.ndarray],
        existing_hashes: List[str],
        # Damage consistency
        damage_types: List[str],
        accident_type: Optional[str],
    ) -> FraudReport:

        signals: List[FraudSignal] = []

        # ── 1. Video continuity ────────────────────────────────────────────
        signals.append(self._check_continuity(has_cuts, continuity_score))

        # ── 2. GPS consistency ─────────────────────────────────────────────
        signals.append(self._check_gps(gps_lat, gps_lng, declared_location))

        # ── 3. Timestamp validity ──────────────────────────────────────────
        signals.append(self._check_timestamp(recorded_at, submitted_at))

        # ── 4. Device trust ────────────────────────────────────────────────
        signals.append(self._check_device(device_id, policy_device_id))

        # ── 5. Duplicate image detection ───────────────────────────────────
        dup_signal, new_hashes = await self._check_duplicates(
            frame_images, existing_hashes
        )
        signals.append(dup_signal)

        # ── 6. Damage vs accident consistency ─────────────────────────────
        signals.append(self._check_damage_consistency(damage_types, accident_type))

        # ── Composite score ────────────────────────────────────────────────
        score = 0.0
        for sig in signals:
            weight = SIGNAL_WEIGHTS.get(sig.name, 0.0)
            score += sig.score_contribution * weight

        score = min(1.0, max(0.0, score))

        # ── Risk level ─────────────────────────────────────────────────────
        if score < 0.25:
            risk_level = "low"
            passed     = True
            override   = None
        elif score < 0.50:
            risk_level = "medium"
            passed     = True
            override   = "adjuster"     # route to adjuster even if STP threshold met
        elif score < 0.75:
            risk_level = "high"
            passed     = False
            override   = "fraud_review"
        else:
            risk_level = "critical"
            passed     = False
            override   = "fraud_review"

        flags = [s.detail for s in signals if not s.passed]

        return FraudReport(
            overall_score    = round(score, 3),
            risk_level       = risk_level,
            passed           = passed,
            signals          = signals,
            triage_override  = override,
            flags            = flags,
        )

    # ── Individual checks ──────────────────────────────────────────────────

    def _check_continuity(
        self, has_cuts: bool, continuity_score: float
    ) -> FraudSignal:
        """
        Edited videos are the primary fraud signal.
        CHANGE: continuity threshold if getting false positives on
        videos shot in changing lighting conditions (raise to 0.60).
        """
        if has_cuts:
            return FraudSignal(
                name="video_continuity",
                passed=False,
                score_contribution=1.0,
                detail=f"Video discontinuity detected — possible edit (continuity={continuity_score:.2f})"
            )
        if continuity_score < 0.70:
            return FraudSignal(
                name="video_continuity",
                passed=False,
                score_contribution=0.5,
                detail=f"Low video continuity score ({continuity_score:.2f}) — possible lighting manipulation"
            )
        return FraudSignal(
            name="video_continuity", passed=True,
            score_contribution=0.0,
            detail="Video continuity verified — no edits detected"
        )

    def _check_gps(
        self,
        lat: Optional[float],
        lng: Optional[float],
        declared: Optional[str],
    ) -> FraudSignal:
        """
        Compare GPS metadata with declared accident location.
        Full implementation requires a geocoding API (Google Maps, OpenStreetMap).
        CHANGE: integrate geocoding once API key is available.
        """
        if lat is None or lng is None:
            return FraudSignal(
                name="gps_consistency", passed=False,
                score_contribution=0.4,
                detail="GPS metadata missing from video — location unverified"
            )
        # TODO: resolve declared_location to lat/lng via geocoding API
        # then compute haversine distance and compare to FRAUD_GPS_TOLERANCE_KM
        return FraudSignal(
            name="gps_consistency", passed=True,
            score_contribution=0.0,
            detail=f"GPS recorded: {lat:.4f}, {lng:.4f}"
        )

    def _check_timestamp(
        self,
        recorded_at: Optional[datetime],
        submitted_at: Optional[datetime],
    ) -> FraudSignal:
        """
        Video should be recorded close to when claim is submitted.
        CHANGE: tolerance in .env FRAUD_TIMESTAMP_TOLERANCE_HR (default 4h).
        """
        from app.config import get_settings
        tolerance_hr = get_settings().fraud_timestamp_tolerance_hr

        if recorded_at is None:
            return FraudSignal(
                name="timestamp_validity", passed=False,
                score_contribution=0.3,
                detail="Video timestamp metadata missing"
            )

        now = submitted_at or datetime.utcnow()
        delta_hr = abs((now - recorded_at).total_seconds()) / 3600

        if delta_hr > tolerance_hr * 4:
            return FraudSignal(
                name="timestamp_validity", passed=False,
                score_contribution=0.8,
                detail=f"Video recorded {delta_hr:.0f}h before submission — well outside tolerance"
            )
        if delta_hr > tolerance_hr:
            return FraudSignal(
                name="timestamp_validity", passed=False,
                score_contribution=0.4,
                detail=f"Video recorded {delta_hr:.0f}h before submission"
            )
        return FraudSignal(
            name="timestamp_validity", passed=True,
            score_contribution=0.0,
            detail=f"Timestamp within acceptable range ({delta_hr:.1f}h)"
        )

    def _check_device(
        self,
        device_id: Optional[str],
        policy_device_id: Optional[str],
    ) -> FraudSignal:
        """
        Device should match the registered policyholder device.
        CHANGE: implement device registration flow in your auth system.
        """
        if not device_id:
            return FraudSignal(
                name="device_trust", passed=False,
                score_contribution=0.3,
                detail="Device ID not captured"
            )
        if policy_device_id and device_id != policy_device_id:
            return FraudSignal(
                name="device_trust", passed=False,
                score_contribution=0.6,
                detail="Video recorded on unregistered device"
            )
        return FraudSignal(
            name="device_trust", passed=True,
            score_contribution=0.0,
            detail="Registered device confirmed"
        )

    async def _check_duplicates(
        self,
        frames: List[np.ndarray],
        existing_hashes: List[str],
    ) -> Tuple[FraudSignal, List[str]]:
        """
        Compute perceptual hash for each frame and compare against
        all previously stored claim hashes in the database.
        CHANGE: FRAUD_IMAGE_HASH_THRESHOLD in .env (default 10).
        Lower = stricter (fewer false negatives, more false positives).
        """
        from app.config import get_settings
        threshold = get_settings().fraud_image_hash_threshold

        new_hashes = []
        matched_any = False

        for frame_arr in frames:
            try:
                pil_img = Image.fromarray(
                    (frame_arr * 255).astype(np.uint8)
                    if frame_arr.max() <= 1.0
                    else frame_arr.astype(np.uint8)
                )
                phash = str(imagehash.phash(pil_img))
                new_hashes.append(phash)

                for existing in existing_hashes:
                    dist = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing)
                    if dist <= threshold:
                        matched_any = True
                        break
            except Exception as e:
                logger.debug(f"Hash computation failed for frame: {e}")

        if matched_any:
            return FraudSignal(
                name="duplicate_image", passed=False,
                score_contribution=1.0,
                detail="Frame matches image from a previous claim — possible reuse"
            ), new_hashes

        return FraudSignal(
            name="duplicate_image", passed=True,
            score_contribution=0.0,
            detail="No duplicate frames found in claims database"
        ), new_hashes

    def _check_damage_consistency(
        self,
        damage_types: List[str],
        accident_type: Optional[str],
    ) -> FraudSignal:
        """
        Cross-check detected damage against declared accident type.
        E.g. rear-end collision should not show only front damage.
        CHANGE: extend INCONSISTENCY_RULES with more accident/damage pairs.
        """
        if not accident_type or not damage_types:
            return FraudSignal(
                name="damage_consistency", passed=True,
                score_contribution=0.1,
                detail="Damage consistency check skipped — insufficient data"
            )

        # CHANGE: add more rules based on your claims experience
        INCONSISTENCY_RULES: Dict[str, List[str]] = {
            "rear_end":     ["broken_glass"],   # rear-end rarely breaks windscreen
            "flood":        ["dent", "scratch"], # flood shouldn't cause dents
            "fire":         ["scratch"],         # fire unlikely to cause scratch only
        }

        suspect_damages = INCONSISTENCY_RULES.get(accident_type.lower(), [])
        triggered = [d for d in damage_types if d in suspect_damages]

        if triggered:
            return FraudSignal(
                name="damage_consistency", passed=False,
                score_contribution=0.5,
                detail=f"Damage types {triggered} inconsistent with declared accident '{accident_type}'"
            )

        return FraudSignal(
            name="damage_consistency", passed=True,
            score_contribution=0.0,
            detail=f"Damage pattern consistent with declared accident type '{accident_type}'"
        )
