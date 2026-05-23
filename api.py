"""FastAPI REST endpoint for Skin Disease Detection Using Deep Learning.

Run:
    uvicorn api:app --reload --port 8000

    Then open http://localhost:8000 in your browser.

API Test:
    curl -X POST http://localhost:8000/predict \\
         -H "Authorization: Bearer <API_KEY>" \\
         -F "file=@skin.jpg"
"""
from __future__ import annotations

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

# ── PIL image bomb protection ─────────────────────────────────────────────────
from PIL import Image
Image.MAX_IMAGE_PIXELS = 50_000_000  # ~7000x7000 max

# ── Magic bytes for image validation ──────────────────────────────────────────
_IMAGE_SIGNATURES = {
    b"\xff\xd8\xff":       "jpeg",     # JPEG
    b"\x89PNG\r\n\x1a\n":  "png",      # PNG
    b"BM":                 "bmp",      # BMP
    b"RIFF":               "webp",     # WebP (RIFF container)
}

def _is_valid_image_bytes(data: bytes) -> bool:
    """Verify file is a real image by checking magic bytes, not just extension."""
    for sig in _IMAGE_SIGNATURES:
        if data[:len(sig)] == sig:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Model warmup on startup — avoids cold-start latency on first request
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the model into memory at startup, not on first request."""
    try:
        from predict import load_model_cached
        load_model_cached()
        print("[OK] Model pre-loaded successfully on startup.")
    except Exception as exc:
        print(f"[WARNING] Model warmup failed: {exc} -- will retry on first request.")
    yield


app = FastAPI(
    title="Skin Disease Detection Using Deep Learning - API",
    description=(
        "Upload a skin image and receive a disease classification "
        "with confidence score and severity level. "
        "Powered by a MobileNetV2 deep learning model."
    ),
    version="2.2.0",
    lifespan=lifespan,
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ── Security ──────────────────────────────────────────────────────────────────
security = HTTPBearer()

_ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# API key from environment — fail hard in production
_API_KEY = os.getenv("API_KEY", "")
if not _API_KEY:
    if _ENVIRONMENT == "production":
        raise RuntimeError(
            "API_KEY environment variable MUST be set in production! "
            "Set it before starting: export API_KEY=your-secret-key"
        )
    import warnings
    warnings.warn(
        "API_KEY environment variable not set! "
        "Set it before deploying to production: export API_KEY=your-secret-key",
        stacklevel=2,
    )
    _API_KEY = "dev-key-not-for-production"  # Dev-only fallback


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── CORS — restricted to known origins ────────────────────────────────────────
_CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:8501,http://localhost:3000,http://localhost:8000,http://localhost:8080",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS.split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ── Security headers ──────────────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    if _ENVIRONMENT == "production":
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com"
        )
    return response


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# ── Static files (HTML frontend) ─────────────────────────────────────────────
_STATIC_DIR = Path(__file__).resolve().parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["system"], include_in_schema=False)
async def root():
    """Redirect to the web UI."""
    return RedirectResponse(url="/static/index.html")


@app.get("/health", tags=["system"])
async def health_check():
    """Liveness probe for load balancers."""
    return {"status": "healthy", "version": "2.2.0"}


@app.post("/predict", tags=["inference"])
@limiter.limit("10/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key),
):
    """Classify a skin disease image.

    Args:
        file: JPEG or PNG image file (max 10 MB).

    Returns:
        JSON with disease name, confidence (0–1), severity label and colour.
        Includes reliability analysis with warnings for uncertain predictions.
    """
    # ── 1. Validate extension ─────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use JPG or PNG.",
        )

    # Sanitize suffix to prevent path traversal
    safe_suffix = ".jpg" if suffix in [".jpg", ".jpeg"] else ".png"

    # ── 2. Read and validate size ─────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10 MB.",
        )

    # ── 3. Validate magic bytes (is it actually an image?) ────────────────
    if not _is_valid_image_bytes(content):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. The file content does not match any supported image format.",
        )

    # ── 4. Write to temp file and predict ─────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from predict import predict_single_image, analyze_prediction_reliability
        from utils.disease_info import get_disease_info, get_severity
        from utils.ood_detector import is_skin_image
        from utils.image_utils import check_image_quality

        # ── 5. Image quality check ────────────────────────────────────
        _, quality_warnings = check_image_quality(tmp_path)

        # ── 6. OOD detection (is it actually skin?) ───────────────────
        is_skin, skin_ratio = is_skin_image(tmp_path)
        if not is_skin:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image does not appear to contain human skin. "
                    f"Skin pixel ratio: {skin_ratio:.1%}. "
                    "Please upload a clear photo of the affected skin area."
                ),
            )

        # ── 7. Prediction with TTA ─────────────────────────────────────
        tta_result = predict_single_image(tmp_path, use_tta=True)
        top_results, avg_preds, all_pass_probs = tta_result
        disease = top_results[0]["disease"]
        confidence = top_results[0]["confidence"]  # already 0-100

        # ── 8. Reliability analysis (with TTA disagreement) ───────────
        reliability = analyze_prediction_reliability(
            avg_preds, confidence, all_pass_probs=all_pass_probs
        )
        all_warnings = quality_warnings + reliability["warnings"]

        info = get_disease_info(disease)
        severity_label, severity_colour = get_severity(confidence, disease)

        return JSONResponse({
            "disease":         disease,
            "display_name":    info.get("display_name", disease),
            "confidence":      round(float(confidence / 100.0), 4),
            "severity":        severity_label,
            "severity_colour": severity_colour,
            "is_reliable":     reliability["is_reliable"],
            "prediction_entropy": reliability["entropy"],
            "prediction_margin":  reliability["margin"],
            "tta_used":         True,
            "tta_passes":       top_results[0].get("tta_passes", 8),
            "tta_agreement":    top_results[0].get("tta_agreement", 0),
            "tta_disagreement": reliability["tta_disagreement"],
            "latency_ms":       top_results[0].get("latency_ms", 0),
            "symptoms":        info.get("symptoms", []),
            "recommendations": info.get("recommendations", []),
            "quality_warnings": all_warnings,
            "disclaimer": (
                "This is an AI-assisted screening tool only. "
                "Consult a qualified dermatologist for medical advice."
            ),
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/predict/batch", tags=["inference"])
@limiter.limit("5/minute")
async def predict_batch(
    request: Request,
    files: list[UploadFile] = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key),
):
    """Classify multiple images in one request (max 20 files)."""
    if len(files) > 20:
        raise HTTPException(
            status_code=400,
            detail="Maximum 20 files per batch request.",
        )
    results = []
    for f in files:
        suffix = Path(f.filename or "image.jpg").suffix.lower()
        safe_suffix = ".jpg" if suffix in [".jpg", ".jpeg"] else ".png"
        content = await f.read()

        # Skip invalid files gracefully
        if not _is_valid_image_bytes(content):
            results.append({"filename": f.filename, "error": "Invalid image content"})
            continue

        with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            from predict import predict_single_image, analyze_prediction_reliability

            tta_result = predict_single_image(tmp_path, use_tta=True)
            top_results, avg_preds, all_pass_probs = tta_result
            disease = top_results[0]["disease"]
            confidence = top_results[0]["confidence"]

            reliability = analyze_prediction_reliability(
                avg_preds, confidence, all_pass_probs=all_pass_probs
            )

            results.append({
                "filename":       f.filename,
                "disease":        disease,
                "confidence":     round(float(confidence / 100.0), 4),
                "is_reliable":    reliability["is_reliable"],
                "tta_agreement":  top_results[0].get("tta_agreement", 0),
                "tta_disagreement": reliability["tta_disagreement"],
            })
        except Exception as exc:
            results.append({
                "filename": f.filename,
                "error":    str(exc),
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return JSONResponse({"results": results, "total": len(results)})


# ─────────────────────────────────────────────────────────────────────────────
# Web UI endpoint — no auth required (same-origin / local use)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/predict/web", tags=["inference"])
@limiter.limit("15/minute")
async def predict_web(
    request: Request,
    file: UploadFile = File(...),
):
    """Classify a skin disease image — used by the built-in web UI.

    Same logic as /predict but without API key authentication.
    Returns a frontend-friendly JSON shape with reliability analysis.
    """
    # ── 1. Validate extension ─────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use JPG or PNG.",
        )
    safe_suffix = ".jpg" if suffix in [".jpg", ".jpeg"] else ".png"

    # ── 2. Read and validate size ─────────────────────────────────────────
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    # ── 3. Validate magic bytes ───────────────────────────────────────────
    if not _is_valid_image_bytes(content):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file. The file content does not match any supported image format.",
        )

    # ── 4. Write to temp file and predict ─────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=safe_suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from predict import predict_single_image, analyze_prediction_reliability
        from utils.disease_info import get_disease_info, get_severity
        from utils.ood_detector import is_skin_image
        from utils.image_utils import check_image_quality

        # ── 5. Image quality check ────────────────────────────────────
        _, quality_warnings = check_image_quality(tmp_path)

        # ── 6. OOD detection ──────────────────────────────────────────
        is_skin, skin_ratio = is_skin_image(tmp_path)
        if not is_skin:
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image does not appear to contain human skin. "
                    f"Skin pixel ratio: {skin_ratio:.1%}. "
                    "Please upload a clear photo of the affected skin area."
                ),
            )

        # ── 7. Prediction with TTA ─────────────────────────────────────
        tta_result = predict_single_image(tmp_path, use_tta=True)
        top_results, avg_preds, all_pass_probs = tta_result
        disease = top_results[0]["disease"]
        confidence = top_results[0]["confidence"]  # 0-100

        # ── 8. Reliability analysis (with TTA disagreement) ───────────
        reliability = analyze_prediction_reliability(
            avg_preds, confidence, all_pass_probs=all_pass_probs
        )
        all_warnings = quality_warnings + reliability["warnings"]

        info = get_disease_info(disease)
        severity_label, severity_colour = get_severity(confidence, disease)

        # ── Build frontend-friendly response with TTA metadata ────────
        return JSONResponse({
            "disease":          disease,
            "display_name":     info.get("display_name", disease),
            "confidence":       round(float(confidence / 100.0), 4),
            "description":      info.get("description", ""),
            "severity":         severity_label,
            "severity_colour":  severity_colour,
            "specialist":       info.get("specialist", "Dermatologist"),
            "is_reliable":      reliability["is_reliable"],
            "prediction_entropy": reliability["entropy"],
            "prediction_margin":  reliability["margin"],
            "tta_used":          True,
            "tta_passes":        top_results[0].get("tta_passes", 8),
            "tta_agreement":     top_results[0].get("tta_agreement", 0),
            "tta_disagreement":  reliability["tta_disagreement"],
            "latency_ms":        top_results[0].get("latency_ms", 0),
            "symptoms":         info.get("symptoms", []),
            "recommendations":  info.get("recommendations", []),
            "quality_warnings": all_warnings,
            "top_predictions":  [
                {"disease": r["disease"], "confidence": round(float(r["confidence"] / 100.0), 4)}
                for r in top_results
            ],
            "disclaimer": (
                "This is an AI-assisted screening tool only. "
                "Consult a qualified dermatologist for medical advice."
            ),
        })
    finally:
        Path(tmp_path).unlink(missing_ok=True)

