# ClaimLens — AI Motor Insurance Claims Backend

End-to-end FastAPI backend for AI-powered motor insurance claim filing.
Accepts video uploads from the mobile app, runs multimodal AI analysis,
and returns structured claim assessments.

---

## Architecture

```
Mobile App  ──►  FastAPI (port 8000)
                    ├── Video Processor   (OpenCV)
                    ├── Plate OCR         (EasyOCR + Tesseract)
                    ├── TF Serving Client ──► TF Serving (port 8501)
                    │                           ├── damage_model
                    │                           ├── vehicle_model
                    │                           └── plate_model
                    ├── Fraud Engine
                    ├── Triage Engine
                    └── SQLite / PostgreSQL DB
```

---

## Folder Structure

```
claimlens/
├── app/
│   ├── main.py             ← FastAPI routes (start here)
│   ├── config.py           ← All settings (read from .env)
│   ├── database.py         ← DB models (Policy, Claim, FraudImageHash)
│   ├── video_processor.py  ← Frame extraction + cut detection
│   ├── tf_serving_client.py← Calls TF Serving REST API
│   ├── plate_ocr.py        ← Number plate OCR
│   ├── fraud_engine.py     ← 6-signal fraud analysis
│   └── triage_engine.py    ← STP / adjuster / fraud routing
├── models/
│   ├── tf_serving.config   ← TF Serving model registry
│   └── saved_models/       ← PUT YOUR EXPORTED MODELS HERE
│       ├── damage_model/1/
│       ├── vehicle_model/1/
│       └── plate_model/1/
├── scripts/
│   └── export_models.py    ← Run in Colab to export trained models
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example            ← Copy to .env and fill in values
```

---

## Setup — Step by Step

### Step 1 — Clone and configure
```bash
git clone <your-repo>
cd claimlens
cp .env.example .env
# Open .env and fill in your values — see comments inside
```

### Step 2 — Install dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 3 — Install Tesseract OCR
```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows — download installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then update TESSERACT_CMD in .env
```

### Step 4 — Export your trained models (run in Colab)
```python
# In Google Colab, after training, run:
# scripts/export_models.py
# This saves SavedModel format to Google Drive
# Then download and put in models/saved_models/
```

### Step 5 — Start with Docker (recommended)
```bash
docker-compose up --build
```

### Step 5 (alternative) — Run without Docker
```bash
# Terminal 1: Start FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2: Start TF Serving (requires Docker)
docker run -p 8501:8501 -p 8500:8500 \
  -v $(pwd)/models/saved_models/damage_model:/models/damage_model \
  -v $(pwd)/models/tf_serving.config:/models/tf_serving.config \
  tensorflow/serving:2.15.0 \
  --model_config_file=/models/tf_serving.config
```

### Step 6 — Seed test data
```bash
curl -X POST http://localhost:8000/api/dev/seed-policy
```

### Step 7 — Test the API
```bash
# Open Swagger UI
open http://localhost:8000/docs

# Test claim submission
curl -X POST http://localhost:8000/api/claim/submit \
  -F "video=@test_video.mp4" \
  -F "policy_number=NIC-MV-2024-087432" \
  -F "gps_lat=27.7172" \
  -F "gps_lng=85.3240" \
  -F "accident_type=rear_end"
```

---

## Where to Make Changes

### Change 1 — Fraud thresholds
Edit `.env`:
```
FRAUD_GPS_TOLERANCE_KM=50.0
FRAUD_TIMESTAMP_TOLERANCE_HR=4
FRAUD_IMAGE_HASH_THRESHOLD=10
```

### Change 2 — Repair cost database
Edit `app/tf_serving_client.py` → `REPAIR_COSTS_NPR` dict.
Update values to match current local garage rates in your market.

### Change 3 — Number plate regex
Edit `app/plate_ocr.py` → `PLATE_PATTERN`.
Update the regex for your country's plate format.
Currently set to Nepal format: BA 1 CHA 2345

### Change 4 — STP thresholds
Edit `app/triage_engine.py`:
```python
STP_MAX_COST       = 15_000   # NPR — raise to auto-approve more claims
STP_MAX_FRAUD_SCORE = 0.25    # lower = stricter
```

### Change 5 — TF Serving host
Edit `.env`:
```
TF_SERVING_HOST=your-server-ip
TF_SERVING_REST_PORT=8501
```

### Change 6 — Database (production)
Edit `.env`:
```
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/claimlens
```
Then run Alembic migrations:
```bash
alembic init migrations
alembic revision --autogenerate -m "initial"
alembic upgrade head
```

### Change 7 — Add JWT authentication
In `app/main.py`, add the `Depends(verify_token)` dependency to
all routes before going to production. See FastAPI security docs.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| GET | /api/policy/{policy_number} | Get policy details |
| GET | /api/claims/{policy_number} | Get claim history |
| POST | /api/claim/submit | **Submit video claim** (main endpoint) |
| PATCH | /api/claim/{claim_ref}/status | Update claim status (adjuster) |
| POST | /api/dev/seed-policy | Create test policy (dev only) |

---

## Mobile App Integration

The mobile app (claims_app.html) should:

1. Send video as `multipart/form-data` to `POST /api/claim/submit`
2. Include GPS coordinates from device location API
3. Include device ID for fraud trust scoring
4. Include `recorded_at` as ISO datetime from video metadata
5. Poll `GET /api/claims/{policy_number}` for status updates

### CHANGE in mobile app
In `claims_app.html`, replace the mock `goTo('screen-processing')` call
with a real fetch to the backend:

```javascript
const formData = new FormData();
formData.append('video', videoBlob, 'claim.mp4');
formData.append('policy_number', 'NIC-MV-2024-087432');
formData.append('gps_lat', position.coords.latitude);
formData.append('gps_lng', position.coords.longitude);
formData.append('device_id', getDeviceId());
formData.append('recorded_at', new Date().toISOString());

const response = await fetch('http://YOUR_SERVER:8000/api/claim/submit', {
  method: 'POST',
  body: formData,
});
const result = await response.json();
// Use result.damage, result.fraud, result.triage to populate UI
```

---

## Development Notes

- While TF Serving is not yet running, the backend uses **mock predictions**
  so you can test the full API pipeline end-to-end.
- Mock predictions are in `app/tf_serving_client.py` → `_mock_damage_predictions()`
- Remove mock fallbacks once real models are deployed.
