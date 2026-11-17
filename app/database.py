"""
ClaimLens — Database Models
Uses SQLAlchemy async ORM.
Development: SQLite  |  Production: swap DATABASE_URL to PostgreSQL
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from app.config import get_settings

settings = get_settings()

# ── Engine ─────────────────────────────────────────────────────────────────
# CHANGE: for production set DATABASE_URL in .env to a PostgreSQL connection string
engine = create_async_engine(
    settings.database_url,
    echo=settings.app_env == "development",   # SQL logging in dev only
    # For PostgreSQL in production, add:
    # pool_size=10, max_overflow=20, pool_pre_ping=True
)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


# ── Models ─────────────────────────────────────────────────────────────────
class Policy(Base):
    """Policyholder and vehicle details — pre-populated from your insurance core system"""
    __tablename__ = "policies"

    id              = Column(Integer, primary_key=True, index=True)
    policy_number   = Column(String(50), unique=True, index=True, nullable=False)
    holder_name     = Column(String(100), nullable=False)
    holder_phone    = Column(String(20), nullable=False)
    holder_email    = Column(String(100))
    vehicle_reg     = Column(String(20), nullable=False, index=True)  # number plate
    vehicle_make    = Column(String(50))
    vehicle_model   = Column(String(50))
    vehicle_year    = Column(Integer)
    vehicle_color   = Column(String(20))
    sum_insured     = Column(Float)
    coverage_type   = Column(String(30))  # comprehensive | third-party
    policy_start    = Column(DateTime)
    policy_end      = Column(DateTime)
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)


class Claim(Base):
    """One claim record per submission"""
    __tablename__ = "claims"

    id              = Column(Integer, primary_key=True, index=True)
    claim_ref       = Column(String(30), unique=True, index=True)   # CLM-2024-001
    policy_number   = Column(String(50), index=True, nullable=False)
    vehicle_reg     = Column(String(20), nullable=False)
    status          = Column(String(30), default="submitted")
    # submitted | ai_assessed | adjuster_review | approved | rejected | settled

    # Video metadata
    video_path      = Column(String(500))
    video_duration  = Column(Float)
    gps_lat         = Column(Float)
    gps_lng         = Column(Float)
    device_id       = Column(String(100))
    recorded_at     = Column(DateTime)

    # AI outputs (stored as JSON)
    detected_plate  = Column(String(20))
    plate_confidence= Column(Float)
    detected_make   = Column(String(50))
    detected_model  = Column(String(50))
    vehicle_confidence = Column(Float)
    damage_results  = Column(JSON)        # list of damage detections
    extracted_frames= Column(Integer)
    cost_estimate   = Column(Float)
    cost_breakdown  = Column(JSON)        # itemised costs

    # Fraud signals
    fraud_score     = Column(Float)       # 0.0 (clean) – 1.0 (high risk)
    fraud_flags     = Column(JSON)        # list of triggered flags
    fraud_passed    = Column(Boolean, default=True)

    # Triage
    triage_decision = Column(String(30))  # stp | adjuster | fraud_review
    adjuster_id     = Column(Integer, nullable=True)

    # Incident details
    incident_description = Column(Text)
    preferred_garage     = Column(String(100))
    declared_location    = Column(String(200))

    submitted_at    = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FraudImageHash(Base):
    """Perceptual hash store for duplicate image detection"""
    __tablename__ = "fraud_image_hashes"

    id          = Column(Integer, primary_key=True)
    claim_ref   = Column(String(30), index=True)
    frame_hash  = Column(String(64), index=True)
    created_at  = Column(DateTime, default=datetime.utcnow)


# ── Dependency ─────────────────────────────────────────────────────────────
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db():
    """Create all tables. Called on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
