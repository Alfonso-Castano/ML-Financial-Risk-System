# Project State

## Current Status

**Phase**: 03 - API Layer & Orchestration
**Current Plan**: Plan 01 complete (1/3 plans done)
**Last Updated**: 2026-03-02

## Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1: Foundation & Synthetic Data | ✓ Complete | 100% (1/1 plans) |
| 2: ML Model & Training | ✓ Complete | 100% (4/4 plans complete) |
| 3: API Layer & Orchestration | ◆ In Progress | 33% (1/3 plans complete) |
| 4: Frontend Dashboard | ○ Pending | 0% |
| 5: Deployment & Documentation | ○ Pending | 0% |

**Legend**: ○ Pending | ◆ In Progress | ✓ Complete

## Project Context

**Core Value**: Learn how to integrate ML models into applications with clean architecture
**Current Focus**: Phase 3 Plan 01 complete — schemas and predictor built. Next: Plan 02 (FastAPI app + routes).

See `.planning/PROJECT.md` for full architecture and constraints.

## Performance Metrics

| Phase | Plan | Duration | Tasks | Files | Completed |
|-------|------|----------|-------|-------|-----------|
| 01 | 01 | 3min 22sec | 2 | 12 | 2026-02-17 |
| 02 | 01 | 1min | 1 | 1 | 2026-02-23 |
| 02 | 02 | 4min | 1 | 2 | 2026-02-23 |
| 02 | 03 | 2min | 2 | 4 | 2026-02-23 |
| 02 | 04 | 5min | 2 | 5 | 2026-02-23 |
| 03 | 01 | 3min | 2 | 2 | 2026-03-02 |

## Recent Decisions

- 2026-03-02: Threshold-based insight analysis chosen over SHAP/LIME — maps to stress definition, educational, no extra libraries
- 2026-03-02: Predictor encapsulates all ML state, no global model variables — instantiated once via FastAPI lifespan
- 2026-03-02: Running savings starts at 0 and floored at 0 via max(0,...) to match synthetic generator behavior
- 2026-03-02: Risk categories: high (>=0.65), medium (0.35-0.64), low (<0.35) — wider than model's 0.5 threshold to communicate uncertainty
- 2026-03-02: debt_payment_defaulted flags any month where debt_payment==0.0, catching both explicit zero and omitted field
- 2026-02-23: evaluate.py runs standalone (python -m backend.ml.evaluate), not called by train.py - architectural isolation decision
- 2026-02-23: matplotlib.use('Agg') placed before pyplot import for headless rendering on all platforms
- 2026-02-23: All sklearn metrics wrapped with float() to prevent numpy scalar JSON serialization errors
- 2026-02-23: Test set reconstructed deterministically in evaluate.py using same seed/ratios as train.py
- 2026-02-23: squeeze(-1) used instead of squeeze() to handle single-sample last-batch shape mismatch in BCELoss
- 2026-02-23: training_history.json saved at end of __main__ so evaluate.py can generate loss plots independently
- 2026-02-23: num_workers=0 in DataLoaders for Windows compatibility with PyTorch multiprocessing
- 2026-02-23: Feature set locked to 9: avg_income, avg_expenses, final_savings, debt_payment, credit_score, debt_ratio, liquidity_ratio, net_cash_flow, consec_negative_months
- 2026-02-23: debt_ratio denominator is free cash flow (income - expenses), not gross income
- 2026-02-23: Named layer attributes (fc1, relu1, dropout1...) over nn.Sequential for educational clarity
- 2026-02-23: Scaling is external to FinancialDataset - StandardScaler fits in train.py to prevent data leakage
- 2026-02-23: input_size parameterized (default=9) so model is not hard-coded to feature count
- 2026-02-17: Used 4 financial archetypes (struggling, getting_by, stable, comfortable) matching user's salary bands
- 2026-02-17: Calibrated archetype parameters iteratively to achieve 36.2% stress ratio (target: ~35%)
- 2026-02-17: Applied long format (one row per person-month) for time series compatibility
- 2026-02-17: Fixed debt payment per profile, variable income/expenses with 15% monthly variance
- 2026-02-13: Project initialized with minimal GSD structure
- 2026-02-13: Architecture locked (no service layer, no database, vanilla JS)
- 2026-02-13: Complexity guardrails added to CLAUDE.md

## Next Action

**Phase 3 Plan 02**: FastAPI application setup (main.py + routes.py)

**Phase 3 Plan 01 artifacts ready for Plan 02:**
- `backend/api/schemas.py` - all 6 Pydantic v2 models (MonthlyEntry, PredictRequest, PredictResponse, HealthResponse, ComputedFeatures, InsightsObject)
- `backend/ml/predictor.py` - Predictor class (load, predict, health, _build_dataframe, _compute_insights)

**Architecture for Plan 02:**
- `backend/main.py` - FastAPI app with lifespan, CORS, custom 422 handler, static file mount
- `backend/api/routes.py` - thin route handlers for /predict and /health

**Note:** Start server from project root: `uvicorn backend.main:app` (paths in settings.py are relative to project root)

## Last Session

**Session timestamp:** 2026-03-02T07:56:03Z
**Stopped at:** Completed 03-api-layer-orchestration/03-01-PLAN.md
**Status:** Phase 3 Plan 01 complete - schemas and predictor built, both verified working

---

*Updated: 2026-03-02 after Phase 3 Plan 01 completion*
