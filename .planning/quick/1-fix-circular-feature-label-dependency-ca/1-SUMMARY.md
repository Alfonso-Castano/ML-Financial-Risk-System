---
phase: quick-1
plan: 01
subsystem: feature-engineering, ml-training, frontend
tags: [bug-fix, circular-dependency, feature-engineering, sigmoid-saturation]
key-files:
  modified:
    - backend/data/feature_engineering.py
    - backend/ml/predictor.py
    - frontend/app.js
    - frontend/index.html
  regenerated:
    - models/latest_model.pth
    - models/metrics.json
    - models/scaler_stats.json
decisions:
  - "Replaced liquidity_ratio with expense_volatility (coefficient of variation of monthly expenses)"
  - "Replaced consec_negative_months with savings_trend (linear regression slope of cumulative savings)"
  - "ROC-AUC partial improvement: 0.998 -> 0.9951 — circular features removed but archetype separation persists"
metrics:
  completed: 2026-03-06
---

# Quick Task 1: Fix Circular Feature-Label Dependency — Summary

**One-liner:** Replaced two circular features (liquidity_ratio, consec_negative_months) with non-circular alternatives (expense_volatility, savings_trend) — circular dependency broken but archetype-level data separation still limits probability spread.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Replace circular features in feature_engineering.py | d8652e6 | Complete |
| 2 | Update frontend labels, retrain, evaluate | af78ecd | Partial — see blocker |

## What Was Done

### Task 1: Feature Replacement

Removed the two circular features from `backend/data/feature_engineering.py`:

- **Removed:** `compute_liquidity_ratio()` — directly encoded label condition 1 (savings < expenses)
- **Removed:** `compute_consecutive_negative_months()` — directly encoded label condition 2 (3+ negative streak)

Added two non-circular replacements:

- **Added:** `compute_expense_volatility(monthly_expenses)` — coefficient of variation (std/mean) of monthly expenses. Measures spending erraticism without encoding any label threshold.
- **Added:** `compute_savings_trend(cumulative_savings)` — linear regression slope of cumulative savings over the window ($/month). Captures trajectory without threshold comparison.

Updated `engineer_features()` to use the new functions and updated `FEATURE_NAMES` to reflect the swap. Updated `predictor.py _compute_insights()` to remove the two deleted feature references and add threshold-based insight checks for the new features.

### Task 2: Frontend and Retraining

Updated `frontend/app.js` FEATURE_LABELS map and `formatFeatureValue()` switch:
- `liquidity_ratio` -> `expense_volatility` (displayed as 2-decimal ratio)
- `consec_negative_months` -> `savings_trend ($/mo)` (displayed as signed $/month)

Updated `frontend/index.html` About tab feature list with new descriptions.

Retrained model: `python -m backend.ml.train` — training converged in 46/75 epochs, best val loss 0.0775.

Ran evaluation: `python -m backend.ml.evaluate`.

## Blocker: ROC-AUC Still Too High

**Verification criterion:** ROC-AUC < 0.995
**Actual result:** ROC-AUC = 0.9951 (just above threshold)
**Probability distribution:** 82.9% of predictions are extreme (< 0.05 or > 0.95), only 2.7% in the [0.35, 0.65] middle zone.

### Root Cause Analysis

The circular features were successfully removed, producing a small improvement (0.998 → 0.9951). However, the remaining 7 features still provide near-perfect class separation because the **four archetypes are too cleanly partitioned by income and expense ratio**:

| Feature | Single-feature accuracy |
|---------|------------------------|
| final_savings | 96.1% |
| savings_trend | 94.5% |
| net_cash_flow | 94.4% |
| avg_income | 86.0% |
| debt_ratio | 81.0% |
| credit_score | 79.4% |

The problem: "struggling" profiles ($30K-45K, expense_ratio 0.95-1.10) and "stable" profiles ($55K-85K, expense_ratio 0.60-0.75) are so structurally different that `final_savings` alone (healthy mean $42,470 vs stressed mean $1,360) provides 96% accuracy. This is a **data generation problem**, not a feature engineering problem.

### Fix Options (Requires User Decision)

**Option A: Widen archetype overlap in settings.py**
- Increase income range overlap between "struggling" and "getting_by" archetypes
- Increase `MONTHLY_VARIANCE` from 0.15 to 0.25-0.35 to create more borderline profiles
- Regenerate training data and retrain
- Expected ROC-AUC: ~0.80-0.88

**Option B: Introduce a "borderline" archetype**
- Add a 5th archetype with income/expense parameters that straddle the stress boundary
- Weight it at 20-25% to ensure meaningful intermediate cases
- Regenerate and retrain

**Option C: Change label conditions to use future-state**
- Label based on month 13+ outcomes rather than month 12 state
- Requires generating 13-18 months of data, using months 1-12 as features
- More complex but creates genuine uncertainty in features vs labels

**Option D: Accept current result**
- The circular dependency IS broken (both offending features removed)
- ROC-AUC improvement from 0.998 to 0.9951 is real, just not dramatic
- The model is now learning from genuine financial signals, not label-encoding rules
- The saturation persists due to clean archetype separation, not circular logic

## Self-Check

Files committed:
- `backend/data/feature_engineering.py` — verified: `d8652e6`
- `backend/ml/predictor.py` — verified: `d8652e6`
- `frontend/app.js` — verified: `af78ecd`
- `frontend/index.html` — verified: `af78ecd`

FEATURE_NAMES verified: 9 entries, contains expense_volatility and savings_trend, no liquidity_ratio or consec_negative_months.

## Self-Check: PARTIAL PASS

- Feature replacement: PASSED (9 features, correct names, assertions verified)
- Frontend labels: PASSED (app.js and index.html updated)
- ROC-AUC verification: FAILED (0.9951 > 0.995 threshold)
- Probability distribution: FAILED (2.7% in middle zone, target >10%)

**Awaiting user decision on Option A/B/C/D before ROC-AUC criterion can be met.**
