"""
ClaimLens — App Configuration
All settings are loaded from environment variables / .env file.
Change values in .env, not here.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path

class Settings(BaseSettings):
    # App
    app_env: str = "development"
    secret_key: str = "CHANGE_ME"
    allowed_origins: str = "http://localhost:3000"

    # Database
    database_url: str = "sqlite+aiosqlite:///./claimlens.db"

    # TF Serving
    tf_serving_host: str = "localhost"
    tf_serving_rest_port: int = 8501
    tf_serving_grpc_port: int = 8500
    damage_model_name: str = "damage_model"
    vehicle_model_name: str = "vehicle_model"
    plate_model_name: str = "plate_model"

    # Storage
    upload_dir: str = "./uploads"
    max_video_size_mb: int = 200
    max_video_duration_secs: int = 120

    # OCR
    tesseract_cmd: str = "/usr/bin/tesseract"

    # Fraud thresholds
    fraud_gps_tolerance_km: float = 50.0
    fraud_timestamp_tolerance_hr: int = 4
    fraud_image_hash_threshold: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def tf_serving_rest_url(self) -> str:
        return f"http://{self.tf_serving_host}:{self.tf_serving_rest_port}"

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
