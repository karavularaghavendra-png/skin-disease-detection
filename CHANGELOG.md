# Changelog

All notable changes are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [2.2.0] - 2026-04-10

### Fixed
- **Critical**: Batch endpoint (`/predict/batch`) now uses TTA with correct 3-tuple unpack — TTA disagreement was silently always 0
- **Critical**: CI/CD pipeline (`ci.yml`) now targets correct root-level files instead of removed `src/` directory
- **Fairness**: OOD skin detector now uses multi-range YCrCb thresholds supporting Fitzpatrick I–VI skin tones
- Fixed `TROUBLESHOOTING.md` API key reference (was `default-key-change-in-production`, now correct `dev-key-not-for-production`)
- Fixed bandit security scanner path in `.pre-commit-config.yaml` (was scanning removed `src/`)
- Fixed `test_production_safeguards.py` to use `tmp_path` fixture instead of writing to project root
- Centralized TF warning suppression in `logger.py`

### Added
- Security headers middleware (X-Content-Type-Options, X-Frame-Options, Referrer-Policy, CSP in production)
- Model warmup on FastAPI startup (avoids cold-start latency)
- Production-mode API key enforcement (`ENVIRONMENT=production` requires `API_KEY`)
- MIT LICENSE file
- Honest accuracy disclaimer in README and static frontend

### Changed
- Updated CHANGELOG with missing v2.0 and v2.1 entries
- Static frontend "99.4% Confidence Rate" → "95%+ Accuracy" with verification note
- README performance benchmarks now include verification disclaimer
- Version bumped to 2.2.0

## [2.1.0] - 2026-04-09

### Added
- **Test-Time Augmentation (TTA)**: 8-pass deterministic augmentation pipeline for robust inference
- TTA reliability analysis: entropy, margin, and cross-pass disagreement metrics
- `tta_agreement`, `tta_disagreement`, and `latency_ms` fields in API responses
- Streamlit UI: TTA metrics display (agreement %, entropy, latency)
- Static web frontend: TTA metrics bar with color-coded agreement

### Changed
- `/predict` and `/predict/web` endpoints now use TTA by default
- Prediction module refactored to support both single-pass and TTA modes

## [2.0.0] - 2026-04-05

### Added
- Custom static HTML/CSS/JS frontend (`static/index.html`) with glassmorphism design
- `/predict/web` endpoint — no-auth variant for built-in web UI
- OOD detection: YCrCb color-space skin detector (`utils/ood_detector.py`)
- Image quality gating: blur, brightness, resolution checks (`utils/image_utils.py`)
- Prediction reliability analysis (confidence threshold, entropy, margin)
- Production safeguard tests (`tests/test_production_safeguards.py`)
- Multi-stage Docker build with non-root user
- Dependabot configuration for automated dependency updates

### Changed
- Project restructured: removed `src/` package, flat layout with root-level modules
- Disease knowledge base consolidated into `utils/disease_info.py` (single source of truth)
- All model inference goes through `predict.py` (removed duplicate predict modules)

### Removed
- `src/` package directory (modules moved to project root)
- Redundant `model/model.py`, `model/predict.py`, `model/train.py`
- Root-level `severity_logic.py` and `medication_map.py` (consolidated into `utils/`)

## [1.2.0] - 2026-03-16

### Fixed
- **Critical**: Class name mismatch — `fungal` and `normal` now map correctly in disease database
- Added missing `h5py` to requirements.txt
- Fixed misleading `image_bytes: bytes` type hint in `predict_single_image()`

### Added
- `logger.py` — centralised logging (replaces print statements)
- `explainability.py` — Grad-CAM++ heatmaps
- `utils/medication_map.py` — OTC medication guidance surfaced in UI
- `api.py` — FastAPI REST endpoint with `/predict`, `/predict/batch`, `/health`
- `tools/convert_to_tflite.py` — TFLite quantised model export
- `Dockerfile` and `.dockerignore` for containerised deployment
- `.github/workflows/ci.yml` — automated CI with lint, type-check, tests
- `tests/` — pytest suite with conftest.py
- `CONTRIBUTING.md` and `CHANGELOG.md`
- Prediction history in sidebar
- File size validation (10 MB limit)
- Mobile-responsive CSS breakpoints

### Changed
- Dead code removed: stray `]` file, archived `webapp/`

## [1.0.0] - 2026-01-01

### Added
- Initial release: 5-class CNN skin disease classifier
- MobileNetV2 transfer learning pipeline
- Streamlit web interface with custom CSS theming
- Auto-filter and manual image reviewer tools
- Evaluation reports (confusion matrix, classification report)
