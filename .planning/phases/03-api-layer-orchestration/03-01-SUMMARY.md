---
phase: 03-api-layer-orchestration
plan: 01
subsystem: api
tags: [pydantic, fastapi, pytorch, inference, schemas]

# Dependency graph
requires:
  - phase: 02-ml-model-training
    provides: trained model weights (latest_model.pth), scaler stats (scaler_stats.json), metrics (metrics.json), FinancialRiskModel class, engineer_features() and FEATURE_NAMES

provides:
  - Pydantic v2 request/response schemas for /predict and /health endpoints
  - Predictor class that orchestrates the full inference pipeline
  - MonthlyEntry, PredictRequest, PredictResponse, HealthResponse, ComputedFeatures, InsightsObject models
  - Domain-threshold insight generation producing plain-language risk factors
  - API-to-training column mapping (expenses -> total_expenses, running savings computation)

affects: [03-api-layer-orchestration, 04-frontend-dashboard]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pydantic v2 field_validator with @classmethod for field-level validation"
    - "Pydantic v2 model_validator(mode='after') for cross-field validation"
    - "Predictor class encapsulates all ML state (no global model variables)"
    - "Domain-threshold analysis for insights (no SHAP/LIME needed)"
    - "Running cumulative savings with max(0, ...) floor to match training data"

key-files:
  created:
    - backend/api/schemas.py
    - backend/ml/predictor.py
  modified: []

key-decisions:
  - "Threshold-based insight analysis chosen over SHAP/LIME — no extra libraries, maps directly to stress definition, educational value"
  - "Predictor encapsulates all ML state — no global model variables — instantiated once via FastAPI lifespan"
  - "Running savings starts at 0 and is floored at 0 to match synthetic generator behavior"
  - "Risk categories: high (>=0.65), medium (0.35-0.64), low (<0.35) — wider bands than training threshold to communicate uncertainty"
  - "debt_payment_defaulted flags any month where debt_payment==0.0 (catches both explicit zero and omitted field)"

patterns-established:
  - "API column mapping: expenses (API) -> total_expenses (training CSV)"
  - "Feature vector assembled via [features[name] for name in FEATURE_NAMES] to guarantee order matches scaler"
  - "Z-score: (feature_vector - self.mean) / self.scale matches train.py exactly"

# Metrics
duration: 3min
completed: 2026-03-02
---

# Phase 3 Plan 01: Schemas and Predictor Summary

**Pydantic v2 schemas with time-series validation and a Predictor class orchestrating PyTorch inference with domain-threshold insights generation**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-03-02T07:53:20Z
- **Completed:** 2026-03-02T07:56:03Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created 6 Pydantic v2 models covering the full /predict and /health API contracts, with validators rejecting negative income/expenses, month counts outside 6-12, and credit scores outside 300-850
- Created Predictor class that loads model/scaler/metrics from disk with fail-fast RuntimeError, runs the complete inference pipeline, and returns all response fields
- Implemented domain-threshold insight engine that generates plain-language risk factor sentences and summary paragraphs without any additional libraries

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Pydantic schemas for predict and health endpoints** - `10b3ef3` (feat)
2. **Task 2: Create Predictor class with full inference pipeline** - `95a2b0f` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `backend/api/schemas.py` - All 6 Pydantic v2 models: MonthlyEntry, PredictRequest, ComputedFeatures, InsightsObject, PredictResponse, HealthResponse
- `backend/ml/predictor.py` - Predictor class with __init__ (artifact loading), predict(), health(), _build_dataframe(), _compute_insights()

## Decisions Made

- Used domain-threshold analysis for insights instead of SHAP/LIME — maps directly to the stress definition used during training, requires no additional dependencies, and is more educational
- Risk category thresholds set wider than model's 0.5 threshold: high (>=0.65), medium (0.35-0.64), low (<0.35) to communicate model uncertainty meaningfully
- Running savings computation starts at 0 and is floored at 0 via max(0, ...) to match synthetic generator behavior and avoid out-of-distribution inputs
- debt_payment_defaulted is True if any month.debt_payment == 0.0, catching both explicitly-zero and omitted values

## Deviations from Plan

None - plan executed exactly as written. The minor import cleanup (removed accidental AnnotatedString import) was caught before the first verification run.

## Issues Encountered

- `python` command not found on Windows — used `py` instead (Windows Python Launcher). This is expected and not a code issue.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- schemas.py and predictor.py are fully functional and importable
- Predictor tested with both high-risk and low-risk profiles producing correct scores and categories
- Ready for Plan 02: FastAPI app creation (main.py + routes.py) which will wire these pieces together
- Server must be started from project root (`uvicorn backend.main:app`) for relative model paths to resolve correctly

## Self-Check: PASSED

- FOUND: backend/api/schemas.py
- FOUND: backend/ml/predictor.py
- FOUND: .planning/phases/03-api-layer-orchestration/03-01-SUMMARY.md
- FOUND commit: 10b3ef3 (Task 1 - schemas)
- FOUND commit: 95a2b0f (Task 2 - predictor)

---
*Phase: 03-api-layer-orchestration*
*Completed: 2026-03-02*
