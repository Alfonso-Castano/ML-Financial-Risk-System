---
status: resolved
trigger: "sigmoid-saturation: Neural network with sigmoid output produces only 0% or 100% risk scores"
created: 2026-03-06T00:00:00Z
updated: 2026-03-06T00:10:00Z
---

## Current Focus

hypothesis: CONFIRMED - Circular dependency between engineered features and label conditions
test: Verified mathematically: consec_negative_months >= 3 maps perfectly to label condition 2, liquidity_ratio < 1.0 maps to label condition 1. Together they predict 99.17% of labels using a trivial threshold rule.
expecting: N/A - root cause found
next_action: Return diagnosis to caller

## Symptoms

expected: Continuous risk scores across the full 0-100 range (e.g., 23.5%, 47.2%, 81.0%) reflecting varying degrees of financial stress
actual: Risk gauge shows only 0 or 100. Probability displays as 0.0% or 100.0%. No middle values observed across multiple test inputs
errors: No console errors, no API errors. System works but outputs extreme values
reproduction: Start server (uvicorn backend.main:app), open localhost:8000, fill months 1-6 with any values, click Predict Risk. Score is always 0 or 100
started: Has always been this way since the model was first trained and the dashboard was built

## Eliminated

- hypothesis: H2 - Feature scale mismatch between 12-month training and 6-month inference
  evidence: Diagnostic showed 0 features go out-of-distribution (|z| > 3) when using 6 months vs 12. The 6-month prediction shifts (1.8% -> 26.4%) but stays within a reasonable intermediate range for that sample.
  timestamp: 2026-03-06

- hypothesis: H3 - Stale model/scaler artifacts
  evidence: model .pth and scaler_stats.json have timestamps 3 seconds apart (2026-02-23 02:37:05 and 02:37:09), confirming they were produced in the same training run.
  timestamp: 2026-03-06

## Evidence

- timestamp: 2026-03-06
  checked: Test set probability distribution (450 profiles)
  found: 90.7% of predictions are extreme (< 0.05 or > 0.95). Only 2.2% fall in the [0.35, 0.65] middle zone. min=0.000000, max=1.000000, p50=0.002020, std=0.460339. "Borderline" samples near the median predicted probability are only at ~0.001-0.005.
  implication: The saturation is not a frontend or API issue — the model itself outputs extreme values even on held-out test data. H1 confirmed at the statistical level.

- timestamp: 2026-03-06
  checked: Feature separability by class in the full 3000-profile dataset
  found: consec_negative_months shows ZERO overlap between classes at the decision boundary: all 769 profiles with consec >= 3 are stressed, zero are healthy. This is a hard rule, not a soft pattern.
  implication: The feature encodes the label condition exactly. This is the root of the saturation.

- timestamp: 2026-03-06
  checked: Circular dependency test — can we reproduce labels using only the threshold rules?
  found: The rule "stressed IF (consec_negative_months >= 3 OR liquidity_ratio < 1.0)" achieves 99.17% accuracy on the full dataset with just these two features and hardcoded thresholds. This matches the two label conditions in _apply_stress_labels() in synthetic_generator.py almost exactly.
  implication: The neural network has nothing to learn beyond these two trivial thresholds. It achieves 0.998 ROC-AUC not by learning generalizable patterns but by memorizing the label-generating rules.

- timestamp: 2026-03-06
  checked: Label condition breakdown across 3000 profiles
  found: 317 stressed by condition 1 only (savings < expenses), 77 by condition 2 only (consec >= 3), 692 by both, 1914 healthy. The features consec_negative_months and liquidity_ratio directly implement these conditions.
  implication: For ANY input a user provides: if it happens to have consec_negative_months >= 3 or final_savings < monthly_expenses, the model fires 100%. Otherwise 0%. There is no middle ground because the label definition has no middle ground in the training data.

- timestamp: 2026-03-06
  checked: The two specific features causing the problem
  found: (1) consec_negative_months in feature_engineering.py is computed identically to max_streak in synthetic_generator.py _apply_stress_labels(). (2) liquidity_ratio = final_savings / avg_monthly_expenses approximately mirrors condition 1 (savings < total_expenses with threshold=1.0). These features ARE the label conditions, just renamed.
  implication: The model is not learning risk; it is reconstructing the label assignment function.

## Resolution

root_cause: |
  CIRCULAR DEPENDENCY between engineered features and stress label conditions.

  In synthetic_generator.py, _apply_stress_labels() defines stress using two rules:
    1. final_savings < final_total_expenses * 1.0  (savings below 1 month expenses)
    2. max_consecutive_negative_cashflow_months >= 3

  In feature_engineering.py, two of the 9 features are:
    - consec_negative_months: computed via identical streak-counting logic (IS condition 2)
    - liquidity_ratio = final_savings / avg_monthly_expenses (directly encodes condition 1)

  The model receives features that ARE the label conditions. It learns two trivial
  thresholds (consec >= 3 OR liq < 1.0) and outputs near-1 or near-0 for everything.
  Any input that doesn't trigger these thresholds gets ~0%; any that does gets ~100%.
  This is why 90.7% of test predictions are extreme and the ROC-AUC is 0.998.

  The bug is not in the model architecture, the API, the frontend, or the scaler.
  It is in the DATASET DESIGN: using the exact labeling conditions as model features
  creates a degenerate learning problem with no room for intermediate predictions.

fix:
  To fix, the circular features must be replaced with predictive-but-not-circular
  alternatives. Options:
    1. Remove consec_negative_months from feature set (it is condition 2 verbatim)
    2. Replace liquidity_ratio with a savings trend or savings-to-income ratio
       that correlates with but does not directly implement condition 1
    3. Redesign label conditions to use future outcomes (e.g., will be stressed
       in month 13?) while keeping current-state features for training
    4. Add noise/overlap to archetypes so "borderline" profiles exist with
       intermediate feature values that straddle both label conditions

verification:
  After fix: test set probability distribution should show meaningful proportion
  (>10%) in the [0.35, 0.65] range and ROC-AUC should drop to 0.75-0.90 range
  (reflecting genuine learning difficulty, not trivial threshold memorization).

files_changed: []
