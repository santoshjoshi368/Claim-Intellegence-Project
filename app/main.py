"""
ClaimLens — FastAPI Backend
Main application entry point with all API routes.

Run with:
    uvicorn app.main:app --reload --port 8000
"""
import uuid
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import init_db, get_db, Claim, Policy, FraudImageHash
from app.video_processor import VideoProcessor
from app.tf_serving_client import TFServingClient
from app.plate_ocr import PlateOCR
from app.fraud_engine import FraudEngine
from app.triage_engine import TriageEngine

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s"
)
logger = logging.getLogger("claimlens")

settings = get_settings()

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ClaimLens API",
    description = "AI-powered motor insurance claims processing backend",
    version     = "1.0.0",
    docs_url    = "/docs",      # Swagger UI — disable in production if desired
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = settings.allowed_origins.split(","),
    allow_methods    = ["*"],
    allow_headers    = ["*"],
    allow_credentials= True,
)

# ── Service instances ─────────────────────────────────────────────────────────
video_processor = VideoProcessor()
tf_client       = TFServingClient()
plate_ocr       = PlateOCR()
triage_engine   = TriageEngine()


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await init_db()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    logger.info("ClaimLens API started")


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0", "env": settings.app_env}


# ── Policy lookup (called when customer opens app) ────────────────────────────
@app.get("/api/policy/{policy_number}")
async def get_policy(
    policy_number: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Returns policy details for display on the dashboard.
    CHANGE: In production, fetch from your core insurance system via API
    instead of local DB. Replace this function body with an HTTP call
    to your policy management system.
    """
    result = await db.execute(
        select(Policy).where(Policy.policy_number == policy_number)
    )
    policy = result.scalar_one_or_none()

    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Policy {policy_number} not found"
        )

    return {
        "policy_number":  policy.policy_number,
        "holder_name":    policy.holder_name,
        "vehicle_reg":    policy.vehicle_reg,
        "vehicle_make":   policy.vehicle_make,
        "vehicle_model":  policy.vehicle_model,
        "vehicle_year":   policy.vehicle_year,
        "sum_insured":    policy.sum_insured,
        "coverage_type":  policy.coverage_type,
        "policy_end":     policy.policy_end,
        "is_active":      policy.is_active,
    }


# ── Claim history ─────────────────────────────────────────────────────────────
@app.get("/api/claims/{policy_number}")
async def get_claims(
    policy_number: str,
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(Claim)
        .where(Claim.policy_number == policy_number)
        .order_by(Claim.submitted_at.desc())
    )
    claims = result.scalars().all()
    return [
        {
            "claim_ref":      c.claim_ref,
            "status":         c.status,
            "cost_estimate":  c.cost_estimate,
            "submitted_at":   c.submitted_at,
            "triage_decision":c.triage_decision,
            "fraud_passed":   c.fraud_passed,
        }
        for c in claims
    ]


# ── VIDEO SUBMISSION — main claim filing endpoint ─────────────────────────────
@app.post("/api/claim/submit")
async def submit_claim(
    # ── Required fields ────────────────────────────────────────────────────
    video:          UploadFile = File(...),
    policy_number:  str        = Form(...),

    # ── Optional metadata sent by the mobile app ───────────────────────────
    # CHANGE: ensure your mobile app captures and sends these fields
    gps_lat:        Optional[float]  = Form(None),
    gps_lng:        Optional[float]  = Form(None),
    device_id:      Optional[str]    = Form(None),
    recorded_at:    Optional[str]    = Form(None),   # ISO datetime string
    accident_type:  Optional[str]    = Form(None),   # rear_end | flood | fire | other
    declared_location: Optional[str] = Form(None),
    incident_description: Optional[str] = Form(None),
    preferred_garage: Optional[str]  = Form(None),

    db: AsyncSession = Depends(get_db),
):
    """
    Main claim submission endpoint.
    1. Validates policy
    2. Saves video to disk
    3. Processes video → frames
    4. Runs AI models (damage, vehicle, plate)
    5. Runs fraud engine
    6. Makes triage decision
    7. Saves claim record
    8. Returns full AI analysis
    """

    # ── 1. Validate policy ─────────────────────────────────────────────────
    result = await db.execute(
        select(Policy).where(Policy.policy_number == policy_number)
    )
    policy = result.scalar_one_or_none()

    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found"
        )
    if not policy.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Policy is not active"
        )

    # ── 2. Save uploaded video ─────────────────────────────────────────────
    claim_id  = f"CLM-{datetime.now().year}-{str(uuid.uuid4())[:6].upper()}"
    video_dir = Path(settings.upload_dir) / claim_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"claim_video{Path(video.filename).suffix}"

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    logger.info(f"Video saved: {video_path} ({video_path.stat().st_size / 1e6:.1f} MB)")

    # ── 3. Process video → frames ──────────────────────────────────────────
    try:
        video_analysis = video_processor.process_video(str(video_path))
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video could not be processed: {str(e)}"
        )

    usable_images = video_processor.get_usable_images_as_arrays(video_analysis)
    logger.info(f"Frames: {len(usable_images)} usable / {video_analysis.total_frames} total")

    if len(usable_images) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Video too short or too blurry — please re-record"
        )

    # ── 4. Run AI models in parallel ───────────────────────────────────────
    import asyncio
    damage_task  = asyncio.create_task(tf_client.predict_damage(usable_images))
    vehicle_task = asyncio.create_task(tf_client.predict_vehicle_type(usable_images))
    damage_detections, (vehicle_type, vehicle_conf) = await asyncio.gather(
        damage_task, vehicle_task
    )

    # Plate OCR (CPU-bound — run synchronously)
    plate_result = plate_ocr.read_plate_from_frames(usable_images)

    # Verify plate against policy
    plate_match, plate_sim = plate_ocr.verify_plate_matches_policy(
        plate_result.plate_text, policy.vehicle_reg
    )
    logger.info(
        f"Plate: detected={plate_result.plate_formatted} "
        f"policy={policy.vehicle_reg} match={plate_match} sim={plate_sim:.2f}"
    )

    # Cost estimate
    cost_total, cost_breakdown = tf_client.build_cost_estimate(damage_detections)

    # ── 5. Fraud engine ────────────────────────────────────────────────────
    # Fetch existing hashes from DB for duplicate detection
    hash_result = await db.execute(select(FraudImageHash.frame_hash))
    existing_hashes = [row[0] for row in hash_result.all()]

    recorded_dt = None
    if recorded_at:
        try:
            recorded_dt = datetime.fromisoformat(recorded_at)
        except ValueError:
            pass

    fraud_engine = FraudEngine(db_session=db)
    fraud_report = await fraud_engine.analyse(
        has_cuts           = video_analysis.has_cuts,
        continuity_score   = video_analysis.continuity_score,
        gps_lat            = gps_lat,
        gps_lng            = gps_lng,
        declared_location  = declared_location,
        recorded_at        = recorded_dt,
        submitted_at       = datetime.utcnow(),
        device_id          = device_id,
        policy_device_id   = None,   # TODO: load from policy holder profile
        frame_images       = usable_images,
        existing_hashes    = existing_hashes,
        damage_types       = [d.damage_type for d in damage_detections],
        accident_type      = accident_type,
    )

    # ── 6. Triage decision ─────────────────────────────────────────────────
    # Add plate mismatch as fraud signal
    if not plate_match:
        fraud_report.flags.append(
            f"Plate mismatch: detected {plate_result.plate_formatted} "
            f"vs policy {policy.vehicle_reg}"
        )
        fraud_report.overall_score = min(1.0, fraud_report.overall_score + 0.3)
        fraud_report.passed = False

    triage = triage_engine.decide(
        cost_estimate   = cost_total,
        detections      = damage_detections,
        fraud_score     = fraud_report.overall_score,
        fraud_passed    = fraud_report.passed,
        fraud_override  = fraud_report.triage_override,
    )

    # ── 7. Save claim to DB ────────────────────────────────────────────────
    claim = Claim(
        claim_ref            = claim_id,
        policy_number        = policy_number,
        vehicle_reg          = policy.vehicle_reg,
        status               = "ai_assessed" if fraud_report.passed else "fraud_review",
        video_path           = str(video_path),
        video_duration       = video_analysis.duration_sec,
        gps_lat              = gps_lat,
        gps_lng              = gps_lng,
        device_id            = device_id,
        recorded_at          = recorded_dt,
        detected_plate       = plate_result.plate_text,
        plate_confidence     = plate_result.confidence,
        detected_make        = vehicle_type,
        vehicle_confidence   = vehicle_conf,
        damage_results       = [
            {
                "type":       d.damage_type,
                "confidence": d.confidence,
                "severity":   d.severity,
                "location":   d.location_hint,
            }
            for d in damage_detections
        ],
        extracted_frames     = len(usable_images),
        cost_estimate        = cost_total,
        cost_breakdown       = cost_breakdown,
        fraud_score          = fraud_report.overall_score,
        fraud_flags          = fraud_report.flags,
        fraud_passed         = fraud_report.passed,
        triage_decision      = triage.decision,
        incident_description = incident_description,
        preferred_garage     = preferred_garage,
        declared_location    = declared_location,
    )
    db.add(claim)

    # Save new frame hashes for future duplicate detection
    _, new_hashes = await fraud_engine._check_duplicates(usable_images, [])
    for h in new_hashes:
        db.add(FraudImageHash(claim_ref=claim_id, frame_hash=h))

    await db.flush()

    # ── 8. Return full response ────────────────────────────────────────────
    return {
        "claim_ref":    claim_id,
        "status":       claim.status,

        "vehicle": {
            "detected_plate":    plate_result.plate_formatted,
            "plate_confidence":  plate_result.confidence,
            "plate_match":       plate_match,
            "plate_similarity":  plate_sim,
            "policy_plate":      policy.vehicle_reg,
            "vehicle_type":      vehicle_type,
            "vehicle_confidence":vehicle_conf,
        },

        "video_analysis": {
            "duration_sec":     video_analysis.duration_sec,
            "frames_extracted": len(usable_images),
            "has_cuts":         video_analysis.has_cuts,
            "continuity_score": video_analysis.continuity_score,
            "cut_timestamps":   video_analysis.cut_timestamps,
        },

        "damage": [
            {
                "type":       d.damage_type,
                "confidence": round(d.confidence, 3),
                "severity":   d.severity,
                "location":   d.location_hint,
                "cost_min":   d.repair_cost_min,
                "cost_max":   d.repair_cost_max,
            }
            for d in damage_detections
        ],

        "cost_estimate": {
            "total_npr":  cost_total,
            "breakdown":  cost_breakdown,
        },

        "fraud": {
            "score":      fraud_report.overall_score,
            "risk_level": fraud_report.risk_level,
            "passed":     fraud_report.passed,
            "flags":      fraud_report.flags,
            "signals": [
                {
                    "name":    s.name,
                    "passed":  s.passed,
                    "detail":  s.detail,
                }
                for s in fraud_report.signals
            ],
        },

        "triage": {
            "decision":  triage.decision,
            "reason":    triage.reason,
            "priority":  triage.priority,
            "sla_hours": triage.sla_hours,
        },
    }


# ── Claim status update (for adjuster dashboard) ──────────────────────────────
class StatusUpdate(BaseModel):
    status: str
    adjuster_notes: Optional[str] = None
    final_settlement: Optional[float] = None

@app.patch("/api/claim/{claim_ref}/status")
async def update_claim_status(
    claim_ref: str,
    body: StatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Called by adjuster dashboard to update claim status.
    CHANGE: Add JWT auth here before going to production.
    """
    result = await db.execute(
        select(Claim).where(Claim.claim_ref == claim_ref)
    )
    claim = result.scalar_one_or_none()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    claim.status     = body.status
    claim.updated_at = datetime.utcnow()
    if body.final_settlement:
        claim.cost_estimate = body.final_settlement
    await db.flush()

    return {"claim_ref": claim_ref, "status": claim.status}


# ── Seed a test policy (development only) ─────────────────────────────────────
@app.post("/api/dev/seed-policy", include_in_schema=False)
async def seed_policy(db: AsyncSession = Depends(get_db)):
    """
    Creates a sample policy for testing.
    REMOVE this endpoint before going to production.
    """
    if settings.is_production:
        raise HTTPException(status_code=403, detail="Not available in production")

    policy = Policy(
        policy_number  = "NIC-MV-2024-087432",
        holder_name    = "Santosh Karmacharya",
        holder_phone   = "+977-9841000000",
        holder_email   = "santosh@example.com",
        vehicle_reg    = "BA1CHA2345",
        vehicle_make   = "Toyota",
        vehicle_model  = "Vigo D/Cab",
        vehicle_year   = 2019,
        vehicle_color  = "White",
        sum_insured    = 3_200_000.0,
        coverage_type  = "comprehensive",
        policy_start   = datetime(2024, 1, 1),
        policy_end     = datetime(2024, 12, 31),
        is_active      = True,
    )
    db.add(policy)
    await db.flush()
    return {"message": "Test policy created", "policy_number": policy.policy_number}
