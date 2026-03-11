---
status: resolved
trigger: "POST /predict returns 500 Internal Server Error due to Pydantic ResponseValidationError — the response schema still references old feature names (liquidity_ratio, consec_negative_months) that were replaced with new features (expense_volatility, savings_trend) in Quick Task 1."
created: 2026-03-09T00:00:00Z
updated: 2026-03-09T00:00:00Z
---

## Current Focus

hypothesis: ComputedFeatures Pydantic model in schemas.py has liquidity_ratio and consec_negative_months as required float fields, but predictor.py now returns expense_volatility and savings_trend — causing Pydantic to raise ResponseValidationError on two missing fields.
test: Read schemas.py and feature_engineering.py, cross-reference FEATURE_NAMES vs ComputedFeatures fields
expecting: Mismatch confirmed — fix by replacing the two stale field names in ComputedFeatures
next_action: Apply fix to schemas.py

## Symptoms

expected: POST /predict returns 200 with risk prediction including computed_features containing the 9 current feature names (avg_income, avg_expenses, final_savings, debt_payment, credit_score, debt_ratio, expense_volatility, net_cash_flow, savings_trend)
actual: POST /predict returns 500 with ResponseValidationError: 2 validation errors — 'liquidity_ratio' Field required, 'consec_negative_months' Field required
errors: fastapi.exceptions.ResponseValidationError: 2 validation errors at ('response', 'computed_features', 'liquidity_ratio') and ('response', 'computed_features', 'consec_negative_months')
reproduction: Start server (uvicorn backend.main:app), open localhost:8000, fill months 1-6 with any values, click Predict Risk. Server returns 500.
timeline: Started after Quick Task 1 replaced circular features (liquidity_ratio → expense_volatility, consec_negative_months → savings_trend) in feature_engineering.py and predictor.py but missed updating schemas.py

## Eliminated

- hypothesis: Bug in predictor.py or feature_engineering.py
  evidence: feature_engineering.py FEATURE_NAMES correctly lists expense_volatility and savings_trend; engineer_features() returns the correct 9-key dict. The compute functions are correct.
  timestamp: 2026-03-09T00:00:00Z

## Evidence

- timestamp: 2026-03-09T00:00:00Z
  checked: backend/api/schemas.py ComputedFeatures model (lines 69-84)
  found: Fields are avg_income, avg_expenses, final_savings, debt_payment, credit_score, debt_ratio, liquidity_ratio, net_cash_flow, consec_negative_months
  implication: liquidity_ratio (line 82) and consec_negative_months (line 84) are the two stale field names that were not updated after Quick Task 1

- timestamp: 2026-03-09T00:00:00Z
  checked: backend/data/feature_engineering.py FEATURE_NAMES (lines 25-35)
  found: FEATURE_NAMES = [avg_income, avg_expenses, final_savings, debt_payment, credit_score, debt_ratio, expense_volatility, net_cash_flow, savings_trend]
  implication: The authoritative feature list uses expense_volatility and savings_trend — schemas.py must match exactly

- timestamp: 2026-03-09T00:00:00Z
  checked: grep for liquidity_ratio and consec_negative_months across entire project
  found: Only backend/api/schemas.py has these names in active backend code. .planning docs, .tmp scripts, and .claude/skills references are historical and do not affect runtime.
  implication: Single-file fix in schemas.py is sufficient to resolve the 500 error

## Resolution

root_cause: ComputedFeatures Pydantic model in backend/api/schemas.py was not updated when Quick Task 1 replaced liquidity_ratio with expense_volatility and consec_negative_months with savings_trend in feature_engineering.py. Pydantic requires all declared fields to be present in the response dict; the two missing keys cause a ResponseValidationError before the response reaches the client.
fix: Replace liquidity_ratio: float with expense_volatility: float and consec_negative_months: float with savings_trend: float in ComputedFeatures (schemas.py lines 82 and 84)
verification: ComputedFeatures fields verified against FEATURE_NAMES from feature_engineering.py — exact 9-field match in identical order. No other active backend files reference the old names.
files_changed:
  - backend/api/schemas.py
