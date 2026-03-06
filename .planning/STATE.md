# Project State

## Current Status

**Phase**: 04 - Frontend Dashboard
**Current Plan**: 04-02 complete (2/3 plans done)
**Last Updated**: 2026-03-06

## Progress

| Phase | Status | Progress |
|-------|--------|----------|
| 1: Foundation & Synthetic Data | ✓ Complete | 100% (1/1 plans) |
| 2: ML Model & Training | ✓ Complete | 100% (4/4 plans complete) |
| 3: API Layer & Orchestration | ✓ Complete | 100% (2/2 plans complete) |
| 4: Frontend Dashboard | ◆ In Progress | 67% (2/3 plans complete) |
| 5: Deployment & Documentation | ○ Pending | 0% |

**Legend**: ○ Pending | ◆ In Progress | ✓ Complete

## Project Context

**Core Value**: Learn how to integrate ML models into applications with clean architecture
**Current Focus**: Phase 4 in progress — 04-02 complete (full JavaScript dashboard). Next: 04-03 (final phase review / polish if needed, then Phase 5 deployment).

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
| 03 | 02 | 3min | 1 | 3 | 2026-03-02 |
| 04 | 01 | 4min 14sec | 2 | 2 | 2026-03-04 |
| 04 | 02 | ~25min | 1+verify | 1 | 2026-03-06 |

## Recent Decisions

- 2026-03-06: retryBtn calls handlePredict() (not submitPrediction directly) so validation re-runs on retry
- 2026-03-06: CSV upload populates grid but does not auto-submit — user reviews pre-filled data and clicks Predict
- 2026-03-06: fetchModelHealth() failure is non-fatal — About tab shows fallback text, Predict tab unaffected
- 2026-03-06: Known issue: CSV upload grid population works but downstream prediction deferred — user confirmed low-priority
- 2026-03-04: Rows 1-6 as core required months; rows 7-12 optional-row (opacity 0.6) with divider — UX clarity without hiding months
- 2026-03-04: SVG gauge hand-rolled (not svg-gauge library) — 5-line math, no CDN dependency, educational
- 2026-03-04: Results panel state controlled by class on container (.is-empty/.is-loading/.has-results/.is-error)
- 2026-03-04: CSS shimmer via 5-stop gradient background-size: 300% + @keyframes — pure CSS, no JS
- 2026-03-04: Insight cards use max-height 0/300px CSS transition (not max-height: auto which cannot animate)
- 2026-03-02: Routes contain zero business logic — thin delegation to Predictor via app.state
- 2026-03-02: Lifespan context manager for model loading (fail-fast on startup)
- 2026-03-02: Static files mounted AFTER router so API routes take precedence
- 2026-03-02: 422 errors flattened to {"error": "message"} format for simplicity
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

**Continue Phase 4**: 04-03 (final plan — likely UAT / end-to-end verification or Dockerfile)

**04-02 artifacts ready for 04-03:**
- `frontend/app.js` — complete JS behavior, all states working
- `frontend/index.html` + `frontend/styles.css` — unchanged, verified
- Full prediction flow confirmed: input → validate → POST /predict → gauge + insight cards + features grid

**Start server:** `uvicorn backend.main:app` from project root

## Last Session

**Session timestamp:** 2026-03-06T00:00:00Z
**Stopped at:** Completed 04-frontend-dashboard/04-02-PLAN.md
**Status:** Phase 4 Plan 02 complete — full JavaScript dashboard implemented and user-verified

---

*Updated: 2026-03-06 after Phase 4 Plan 02 completion*
