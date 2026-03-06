---
phase: quick-1
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - backend/data/feature_engineering.py
  - backend/config/settings.py
  - frontend/app.js
  - frontend/index.html
autonomous: false
must_haves:
  truths:
    - "Prediction probability distribution contains meaningful intermediate values (>10% of predictions in [0.35, 0.65] range)"
    - "Feature count remains exactly 9 (locked architecture)"
    - "No circular dependency between features and stress label conditions"
    - "Dashboard FEATURE_LABELS and About tab descriptions match new feature names"
  artifacts:
    - path: "backend/data/feature_engineering.py"
      provides: "expense_volatility and savings_trend compute functions; engineer_features returns both"
    - path: "backend/config/settings.py"
      provides: "FEATURE_NAMES with expense_volatility and savings_trend replacing old names"
    - path: "models/latest_model.pth"
      provides: "Retrained model weights using non-circular features"
    - path: "models/metrics.json"
      provides: "Updated metrics showing ROC-AUC in realistic 0.75-0.90 range"
  key_links:
    - from: "backend/data/feature_engineering.py"
      to: "backend/config/settings.py"
      via: "FEATURE_NAMES list must match engineer_features() dict keys exactly"
    - from: "backend/ml/predictor.py"
      to: "backend/data/feature_engineering.py"
      via: "imports FEATURE_NAMES from feature_engineering — no separate copy to update"
---

<objective>
Replace the two circular features (consec_negative_months and liquidity_ratio) that
directly encode the stress label conditions with non-circular alternatives
(expense_volatility and savings_trend), then retrain and verify the model produces
continuous intermediate predictions.

Purpose: The model currently outputs only 0% or 100% because two of its nine features
ARE the label conditions verbatim. Replacing them breaks the circular dependency and
forces the network to learn genuine risk patterns from financial signals.

Output: Retrained model with probability distribution showing meaningful spread across
the [0, 1] range.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@.planning/debug/sigmoid-saturation.md
@backend/data/feature_engineering.py
@backend/config/settings.py
@backend/ml/predictor.py
@frontend/app.js
@frontend/index.html
</context>

<tasks>

<task type="auto">
  <name>Task 1: Replace circular features in feature_engineering.py and settings.py</name>
  <files>
    backend/data/feature_engineering.py
    backend/config/settings.py
  </files>
  <action>
    In backend/data/feature_engineering.py:

    1. REMOVE compute_liquidity_ratio() function entirely.

    2. REMOVE compute_consecutive_negative_months() function entirely.

    3. ADD compute_expense_volatility() function:
       - Signature: compute_expense_volatility(monthly_expenses: List[float]) -> float
       - Formula: std(monthly_expenses) / mean(monthly_expenses) — coefficient of variation
       - Returns 0.0 if mean is zero or fewer than 2 data points
       - Docstring: "Measures expense instability as standard deviation relative to the mean.
         Higher values indicate erratic spending patterns, a risk signal that correlates with
         but does NOT directly encode the label conditions."

    4. ADD compute_savings_trend() function:
       - Signature: compute_savings_trend(cumulative_savings: List[float]) -> float
       - Formula: linear regression slope of cumulative savings over time.
         Use numpy: np.polyfit(range(len(savings)), savings, 1)[0]
         This gives the average monthly change in savings (dollars per month).
         Positive = growing savings, negative = declining savings.
         Returns 0.0 if fewer than 2 data points.
       - Docstring: "Captures savings trajectory as the linear slope of cumulative savings
         over the observation window. Negative slope signals deteriorating savings
         without directly implementing any label threshold."

    5. UPDATE engineer_features():
       - Replace: cash_flows = (group['income'] - group['total_expenses']).tolist()
         Keep this line — it's still needed for nothing now, but remove it if unused.
       - Add: monthly_expenses = group['total_expenses'].tolist()
       - Add: savings_series = group['savings'].tolist()
       - Replace compute calls:
         OLD: liquidity_ratio = compute_liquidity_ratio(final_savings, avg_expenses)
         NEW: expense_volatility = compute_expense_volatility(monthly_expenses)
         OLD: consec_negative_months = compute_consecutive_negative_months(cash_flows)
         NEW: savings_trend = compute_savings_trend(savings_series)
       - Remove cash_flows line if no longer used by any remaining call.
       - Update return dict: replace 'liquidity_ratio' key with 'expense_volatility',
         replace 'consec_negative_months' key with 'savings_trend'.

    6. UPDATE module docstring (top of file) to reflect new feature set:
       Replace the two old feature lines with:
         expense_volatility    - Coefficient of variation of monthly expenses (std/mean)
         savings_trend         - Linear slope of cumulative savings ($/month)

    7. UPDATE FEATURE_NAMES list (in feature_engineering.py):
       Replace 'liquidity_ratio' with 'expense_volatility'
       Replace 'consec_negative_months' with 'savings_trend'
       Keep all other 7 features and their order unchanged.

    In backend/config/settings.py:
       This file does NOT contain a FEATURE_NAMES list — confirmed during review.
       No changes needed to settings.py.

    NOTE: backend/ml/predictor.py imports FEATURE_NAMES from feature_engineering directly
    (line: from backend.data.feature_engineering import engineer_features, FEATURE_NAMES).
    No changes needed to predictor.py — it will automatically pick up the updated list.
    However, update _compute_insights() in predictor.py to remove references to the
    deleted features:
      - Remove the liquidity_ratio check block (lines ~226-231)
      - Remove the consec_negative_months check block (lines ~233-238)
      - These were the two checks that mirrored the label conditions — removing them
        is correct since the features no longer exist.
  </action>
  <verify>
    python -c "
    from backend.data.feature_engineering import FEATURE_NAMES, engineer_features
    import pandas as pd
    assert len(FEATURE_NAMES) == 9, f'Expected 9 features, got {len(FEATURE_NAMES)}'
    assert 'expense_volatility' in FEATURE_NAMES, 'expense_volatility missing'
    assert 'savings_trend' in FEATURE_NAMES, 'savings_trend missing'
    assert 'liquidity_ratio' not in FEATURE_NAMES, 'liquidity_ratio still present'
    assert 'consec_negative_months' not in FEATURE_NAMES, 'consec_negative_months still present'
    print('FEATURE_NAMES:', FEATURE_NAMES)
    print('All assertions passed')
    "
  </verify>
  <done>
    FEATURE_NAMES has exactly 9 entries containing expense_volatility and savings_trend.
    liquidity_ratio and consec_negative_months are absent.
    engineer_features() returns a dict with the 9 updated keys.
    predictor.py _compute_insights() no longer references the deleted features.
  </done>
</task>

<task type="auto">
  <name>Task 2: Update frontend labels and About tab, then retrain and evaluate</name>
  <files>
    frontend/app.js
    frontend/index.html
  </files>
  <action>
    In frontend/app.js, update FEATURE_LABELS map (around line 17-27):
      Replace:
        liquidity_ratio:        'Liquidity Ratio',
        consec_negative_months: 'Consecutive Negative Months',
      With:
        expense_volatility:     'Expense Volatility',
        savings_trend:          'Savings Trend ($/mo)',

    In frontend/index.html, update the About tab feature list (around lines 410-412):
      Replace:
        <li><strong>liquidity_ratio</strong> — Final savings ÷ average monthly expenses</li>
      With:
        <li><strong>expense_volatility</strong> — Std dev ÷ mean of monthly expenses (coefficient of variation). Higher = more erratic spending.</li>

      Replace:
        <li><strong>consec_negative_months</strong> — Longest run of consecutive months where income &lt; expenses + debt_payment</li>
      With:
        <li><strong>savings_trend</strong> — Linear slope of cumulative savings over the window ($/month). Negative = declining savings.</li>

    After frontend changes, retrain the model:
      python -m backend.ml.train

    Then re-evaluate:
      python -m backend.ml.evaluate

    Capture the probability distribution from evaluate output. The fix is working if:
      - ROC-AUC drops from ~0.998 to somewhere in the 0.75-0.92 range
      - The console output or metrics.json shows the model is no longer trivially separating classes
  </action>
  <verify>
    python -c "
    import json
    with open('models/metrics.json') as f:
        m = json.load(f)
    roc = m['roc_auc']
    print(f'ROC-AUC: {roc:.4f}')
    assert roc < 0.995, f'ROC-AUC {roc:.4f} still suspiciously high — circular dependency may persist'
    print('ROC-AUC check passed: model is no longer memorizing label rules')
    "
  </verify>
  <done>
    frontend/app.js FEATURE_LABELS shows expense_volatility and savings_trend.
    frontend/index.html About tab describes the two new features.
    models/latest_model.pth and models/metrics.json are regenerated from the retrained model.
    ROC-AUC is below 0.995, indicating the model is learning genuine patterns.
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking">
  <what-built>
    Replaced two circular features (liquidity_ratio, consec_negative_months) with
    non-circular alternatives (expense_volatility, savings_trend), updated frontend
    labels, and retrained the model.
  </what-built>
  <how-to-verify>
    1. Start the server: uvicorn backend.main:app --reload
    2. Open http://localhost:8000
    3. Fill in months 1-6 with values that previously gave 100% (e.g., income=3000,
       expenses=3500 for all 6 months with credit score 650)
    4. Click Predict Risk — score should no longer be exactly 100%
    5. Fill in 6 months with clearly healthy values (income=6000, expenses=2000,
       credit score 750, debt payment 200)
    6. Click Predict Risk — score should be low but not exactly 0%
    7. Try a borderline case (income=4000, expenses=3600, credit score 680) — you
       should now see an intermediate score like 25-75% rather than 0 or 100
    8. Check the Computed Features panel — it should show "Expense Volatility" and
       "Savings Trend ($/mo)" instead of the old labels
    9. Visit the About tab — confirm the feature list shows the two new feature descriptions
  </how-to-verify>
  <resume-signal>Type "approved" if intermediate scores appear, or describe the scores you're seeing</resume-signal>
</task>

</tasks>

<verification>
After all tasks complete:
- FEATURE_NAMES in feature_engineering.py has exactly 9 entries
- expense_volatility and savings_trend are present; liquidity_ratio and consec_negative_months are absent
- models/metrics.json ROC-AUC is below 0.995
- Frontend computed features panel shows updated labels
- Prediction produces continuous scores, not just 0% and 100%
</verification>

<success_criteria>
The model produces non-degenerate predictions: meaningful proportion of scores
fall in the intermediate range (not exclusively 0% or 100%). ROC-AUC reflects
genuine learning difficulty rather than trivial threshold memorization.
Feature count remains exactly 9. Frontend labels and documentation match new features.
</success_criteria>

<output>
No SUMMARY.md required for quick plans. Update .planning/STATE.md if desired to
note the fix was applied and the model was retrained.
</output>
