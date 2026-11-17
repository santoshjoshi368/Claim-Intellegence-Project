"""
ClaimLens — Triage Engine
Decides claim routing based on AI outputs and fraud score.

Three outcomes:
  stp           → Straight-Through Processing (auto-approve & pay)
  adjuster      → Route to human adjuster for review
  fraud_review  → Escalate to Special Investigation Unit (SIU)

CHANGE: STP_MAX_COST and thresholds based on your claims policy.
"""
from dataclasses import dataclass
from typing import List, Optional
from app.tf_serving_client import DamageDetection


# ── CHANGE these to match your company's claims policy ───────────────────────
STP_MAX_COST          = 15_000   # NPR — claims below this auto-approve if clean
STP_MIN_CONFIDENCE    = 0.75     # minimum avg damage detection confidence for STP
STP_MAX_FRAUD_SCORE   = 0.25     # fraud score must be below this for STP
STP_ALLOWED_SEVERITIES = {"minor"}  # only minor damage qualifies for STP


@dataclass
class TriageDecision:
    decision: str           # stp | adjuster | fraud_review
    reason: str
    priority: str           # low | normal | high | urgent
    sla_hours: int          # expected resolution time


class TriageEngine:

    def decide(
        self,
        cost_estimate: float,
        detections: List[DamageDetection],
        fraud_score: float,
        fraud_passed: bool,
        fraud_override: Optional[str],
    ) -> TriageDecision:

        # ── Fraud override always wins ─────────────────────────────────────
        if not fraud_passed or fraud_override == "fraud_review":
            return TriageDecision(
                decision = "fraud_review",
                reason   = "Fraud signals detected — escalated to SIU",
                priority = "urgent",
                sla_hours= 24,
            )

        # ── Adjuster override from fraud engine (medium risk) ──────────────
        if fraud_override == "adjuster":
            return TriageDecision(
                decision = "adjuster",
                reason   = "Elevated fraud score — adjuster review required",
                priority = "high",
                sla_hours= 48,
            )

        # ── Severe damage always needs adjuster ───────────────────────────
        severities = {d.severity for d in detections}
        if "severe" in severities:
            return TriageDecision(
                decision = "adjuster",
                reason   = "Severe damage detected — physical inspection required",
                priority = "high",
                sla_hours= 48,
            )

        # ── High cost → adjuster ──────────────────────────────────────────
        if cost_estimate > STP_MAX_COST:
            return TriageDecision(
                decision = "adjuster",
                reason   = f"Estimated cost NPR {cost_estimate:,.0f} exceeds STP threshold",
                priority = "normal",
                sla_hours= 72,
            )

        # ── Low confidence → adjuster ──────────────────────────────────────
        if detections:
            avg_conf = sum(d.confidence for d in detections) / len(detections)
            if avg_conf < STP_MIN_CONFIDENCE:
                return TriageDecision(
                    decision = "adjuster",
                    reason   = f"Low average detection confidence ({avg_conf:.0%}) — human review needed",
                    priority = "normal",
                    sla_hours= 72,
                )

        # ── Fraud score borderline ─────────────────────────────────────────
        if fraud_score > STP_MAX_FRAUD_SCORE:
            return TriageDecision(
                decision = "adjuster",
                reason   = f"Fraud score {fraud_score:.2f} above STP threshold",
                priority = "normal",
                sla_hours= 48,
            )

        # ── All checks passed → STP ────────────────────────────────────────
        return TriageDecision(
            decision = "stp",
            reason   = "All AI checks passed — eligible for straight-through processing",
            priority = "low",
            sla_hours= 24,
        )
