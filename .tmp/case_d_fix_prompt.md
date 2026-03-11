# Fix Case D: High-Income High-Burn Leniency

## Your Role

You are the **orchestrator**. Do NOT do all the work yourself. Delegate to your sub-agents and skills, coordinate their outputs, and make architectural decisions. Use the Agent tool to spawn specialists and your `/senior-data-scientist` and `/senior-data-engineer` skills for statistical analysis and data pipeline work.

## The Problem

Case D ($10,000 income / $10,000 expenses / 700 credit) predicts **4.2% risk** — expected is **40-60%**. The model is dangerously lenient on high-income profiles that spend everything they earn.

## Root Cause (Already Diagnosed)

The model over-indexes on `avg_income` (Cohen's d = -1.248, the strongest feature). High income is a powerful "healthy" signal that overrides `expense_ratio = 1.0`. This is an **OOD training data problem**: no high-income profiles with expense_ratio near 1.0 exist in the training data. The model has never seen someone earning $10k/month who also spends $10k/month.

**Evidence from prior session:**
- Training data archetypes cap high-income profiles ("comfortable": $65k-$130k/yr) at expense_ratio 0.40-0.70
- No archetype covers high-income + high-expense-ratio simultaneously
- When the model sees income=$10k + expense_ratio=1.0, the income signal dominates because it's never learned that high earners can also be stressed

## Current System State

**Model performance:** ROC-AUC 0.9584, Recall 0.7704, Precision 0.8889

**Feature set (9 features, in order):**
`avg_income, avg_expenses, final_savings, expense_ratio, credit_score, savings_months, expense_volatility, net_cash_flow, savings_trend`

**Current archetypes** (in `backend/config/settings.py`):
```python
ARCHETYPES = {
    "poverty":     {"income_range": (15000, 30000),  "expense_ratio": (0.90, 1.20), "weight": 0.10},
    "struggling":  {"income_range": (28000, 58000),  "expense_ratio": (0.85, 1.12), "weight": 0.25},
    "getting_by":  {"income_range": (35000, 72000),  "expense_ratio": (0.72, 1.02), "weight": 0.23},
    "stable":      {"income_range": (48000, 92000),  "expense_ratio": (0.55, 0.88), "weight": 0.26},
    "comfortable": {"income_range": (65000, 130000), "expense_ratio": (0.40, 0.70), "weight": 0.16},
}
```

**Key files:**
- `backend/config/settings.py` — Archetypes, hyperparameters, all config
- `backend/data/synthetic_generator.py` — Generates training CSV from archetypes
- `backend/data/feature_engineering.py` — FEATURE_NAMES, all compute_* functions, build_feature_matrix
- `backend/ml/train.py` — Training loop
- `backend/ml/evaluate.py` — Evaluation metrics
- `backend/ml/predictor.py` — API inference orchestration (noise injection at 15%)
- `models/scaler_stats.json` — Scaler mean/scale for z-score normalization
- `models/metrics.json` — Current metrics

**Regression test cases (ALL must pass after fix):**

| Case | Income | Expenses | Credit | Expected | Current |
|------|--------|----------|--------|----------|---------|
| A (deep deficit) | $1,000 | $2,500 | 650 | >80% | ~100% PASS |
| B (breaking even) | $3,000 | $3,000 | 680 | 55-70% | 50.7% ~PASS |
| C (comfortable) | $6,000 | $3,500 | 720 | <30% | 0.1% PASS |
| D (high burn) | $10,000 | $10,000 | 700 | 40-60% | 4.2% FAIL |
| E (frugal) | $2,000 | $1,200 | 660 | 25-40% | 0.1% PASS |

## Constraints (from CLAUDE.md)

- **Locked architecture:** Hidden layers 128 → 64 (ReLU + Dropout) → Sigmoid. Do NOT change.
- **INPUT_SIZE = 9** — can be changed if needed, but the current 9 features are solid. Prefer fixing the data.
- **Stress definition is locked:** savings < 1 month expenses OR 3+ consecutive negative cash flow months.
- **No new files** unless absolutely necessary. Modify existing files.
- **Keep it simple** — this is a learning project, not production.
- Run on Windows. Use `py` not `python` for commands.

## Execution Protocol

### Phase 1: Investigate (use agents in parallel)

Spawn these agents simultaneously:

1. **Debugger agent** — Analyze the training data distribution gap. Quantify: how many profiles in training data have income > $8k/month AND expense_ratio > 0.85? What's their stress rate? Confirm the OOD hypothesis with numbers. Files to read: `backend/config/settings.py`, `data/synthetic_train.csv`, `backend/data/feature_engineering.py`.

2. **Explore agent** — Search the codebase for any hardcoded assumptions about income levels or expense ratios that could constrain the fix. Check if any other code besides settings.py and synthetic_generator.py would need changes. Check the stress labeling logic in synthetic_generator.py to confirm high-income high-expense profiles would correctly trigger stress labels.

### Phase 2: Design the Fix

Based on investigation results, design a training data fix. The most likely solution is adding a new archetype (e.g., "high_burn" or "overspender") to `settings.py:ARCHETYPES` that covers high-income profiles with high expense ratios. Consider:

- Income range should overlap with "comfortable" and extend higher
- Expense ratio should be 0.85-1.15+ (these people spend nearly everything or more)
- Savings buffer should be low (0-2 months)
- Credit score can be moderate (660-720) — high earners with bad habits often maintain decent credit
- Weight should be meaningful enough to create training examples (0.08-0.12)
- CRITICAL: Rebalance other archetype weights so total = 1.0

Use `/senior-data-scientist` to validate that the proposed archetype parameters will produce the right stress distribution and won't break existing cases.

### Phase 3: Implement and Retrain

1. Add the new archetype to `backend/config/settings.py`
2. Regenerate training data: `py -m backend.data.synthetic_generator`
3. Retrain: `py -m backend.ml.train`
4. Evaluate: `py -m backend.ml.evaluate`

### Phase 4: Regression Test

Run ALL 5 test cases (A-E) through the model. Use `/senior-data-scientist` to write and run a regression test script. The script should:
- Test each case with deterministic inputs (no noise) for the baseline
- Test each case with noise (multiple seeds) for variance analysis
- Report pass/fail against expected ranges

**Success criteria:**
- Case D: 40-60% risk (PRIMARY TARGET)
- Cases A, B, C, E: must remain within their expected ranges
- ROC-AUC > 0.93
- Recall > 0.80

### Phase 5: Iterate if Needed

If Case D doesn't hit 40-60% or other cases regress:
- Use the debugger agent to analyze what went wrong
- Adjust archetype parameters (income range, expense ratio range, weight)
- Retrain and retest
- Maximum 3 iterations before escalating to the user

## Important Notes

- The 15% noise injection in `predictor.py:_build_dataframe()` causes variance for borderline cases (income ~ expenses). Case B with noise: mean 75%, std 25%. This is a known side effect — do NOT try to fix it in this session.
- `debt_payment` still exists in the CSV as 20% of total_expenses — it's just not extracted as a separate model feature anymore. Don't touch this.
- After retraining, `models/scaler_stats.json` will update automatically. Don't manually edit it.
- Use `.tmp/` for any scratch scripts. These are disposable.
