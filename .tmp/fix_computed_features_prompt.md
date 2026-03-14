# Fix Computed Feature Errors + Feature Weight Analysis Report

## Your Role

You are the **orchestrator** — a senior software engineer and senior architect. Do NOT do all the work yourself. Delegate to your sub-agents and skills, coordinate their outputs, and make architectural decisions.

**Sub-agents to use:**
- **Debugger agent** (`subagent_type: debugger`) — for root cause analysis, hypothesis testing, reproducing bugs
- **Explore agent** (`subagent_type: Explore`) — for codebase searches, tracing data flows
- `/senior-data-scientist` skill — for statistical analysis, feature importance reporting, validating fixes
- `/senior-data-engineer` skill — for data pipeline integrity, verifying data transformations

**Execution discipline:** Follow each step below in order. Do NOT skip ahead. Complete each step and report its findings before moving to the next.

---

## Part 1: Fix Computed Feature Bugs

### The Bugs (from live application testing)

A user entered 12 months of detailed financial data into the frontend and got incorrect computed features back:

**User input:**
| Month | Income ($) | Expenses ($) |
|-------|-----------|-------------|
| 1     | 2,100     | 2,000       |
| 2     | 2,050     | 2,020       |
| 3     | 2,150     | 2,100       |
| 4     | 2,000     | 2,050       |
| 5     | 2,060     | 2,000       |
| 6     | 2,100     | 2,050       |
| 7-12  | 4,500     | 3,200       |
| Credit Score: 700 |

**Expected computed features (calculated by hand):**
- Avg monthly income: (2100+2050+2150+2000+2060+2100+4500*6)/12 = **$3,288**
- Avg monthly expenses: (2000+2020+2100+2050+2000+2050+3200*6)/12 = **$2,637**
- Net cash flow: $3,288 - $2,637 = **+$652/month**

**What the app returned:**
- Avg monthly income: **$1,834** (WRONG — should be ~$3,288)
- Avg monthly expenses: **$2,077** (WRONG — should be ~$2,637)
- Net cash flow: **-$242** (WRONG — should be positive ~+$652)
- Final savings: **$81** (suspect — seems low given months 7-12 have $1,300 surplus each)
- Expense ratio: **1.13** (WRONG — derived from wrong averages; should be ~0.80)

### Step 1: Identify the Source of Error

**Spawn a Debugger agent** to do root cause analysis. The agent should:

1. Read the full inference pipeline: `backend/ml/predictor.py` (especially `_build_dataframe()`), `backend/data/feature_engineering.py` (`engineer_features()`), `backend/api/schemas.py`, and `backend/api/routes.py`
2. Trace the data flow from API request → `_build_dataframe()` → `engineer_features()` → response
3. **Primary hypothesis to test:** The 15% noise injection in `predictor.py:_build_dataframe()` (lines 211-212) applies `rng.normal(1.0, 0.15)` multipliers to EVERY request, including those with detailed monthly data that already has natural variance. This would corrupt user-provided values and produce wrong averages.
4. **Reproduce the bug:** Write a script in `.tmp/` that sends the exact user input from the table above through `_build_dataframe()` and `engineer_features()`, printing both the noised values and the computed features. Compare against the hand-calculated expected values.
5. **Verify:** Check if the noise injection is the sole cause, or if there are other issues (e.g., months 7-12 not being sent correctly from the frontend, or `engineer_features` sorting/indexing bugs).

**Expected finding:** The noise injection is unconditional — it distorts user-provided multi-month data that already has natural variance. It was designed for the simple 3-input case (constant monthly values produce `expense_volatility=0`, an OOD value) but should NOT be applied when the user provides detailed monthly data.

### Step 2: Identify Possible Solutions

Based on the debugger's findings, enumerate possible solutions. Consider at minimum:

1. **Conditional noise injection** — Detect whether the user provided detailed monthly data (natural variance exists) vs. constant values (all months identical). Only inject noise when variance is near zero.
2. **Remove noise injection entirely** — Accept that API inputs may produce OOD `expense_volatility=0` and let the model handle it. This is simpler but could degrade predictions for simple inputs.
3. **Separate code paths** — One path for simple 3-field inputs (inject noise), one for detailed monthly data (pass through as-is). May add complexity.

For each solution, evaluate:
- Does it fix all three bugs?
- Does it break the simple input case (where noise was intentionally added)?
- How complex is the implementation?
- Does it respect the locked architecture (no new files, no service layers)?

### Step 3: Choose the Best Solution

Use `/senior-data-engineer` to evaluate the solutions against these criteria:
- **Correctness**: User-provided data must pass through unmodified
- **OOD protection**: Simple constant inputs still need variance to avoid `expense_volatility=0`
- **Simplicity**: Minimal code changes, no new abstractions
- **Testability**: Easy to verify both code paths work

Present the chosen solution with a clear rationale.

### Step 4: Create a Plan

Write a step-by-step implementation plan:
1. Which file(s) to modify
2. What the change looks like (pseudocode is fine)
3. How to test (both simple-input and detailed-input cases)
4. What regression risks exist

### Step 5: Execute the Fix

Implement the chosen solution. Then verify by:
1. **Replaying the screenshot input** — Run the exact 12-month input from the bug report through the fixed pipeline. Confirm avg_income ≈ $3,288, avg_expenses ≈ $2,637, NCF ≈ +$652.
2. **Testing the simple input case** — Run Case D ($10,000/$10,000/700 credit) through the fixed pipeline. Confirm `expense_volatility > 0` (noise still injected for constant inputs).
3. **Running the full regression test** — Run the 5-case regression test (`.tmp/regression_test.py`) to confirm Case D still lands in 40-60% range.

---

## Part 2: Feature Weight Analysis Report

After fixing the bugs, generate a report on how the ML model weighs different features and their impact on risk predictions.

**Use `/senior-data-scientist`** to produce this analysis. The report should include:

### 2a. Feature Importance Ranking

Using the trained model (`models/latest_model.pth`), compute feature importance via **permutation importance** on the test set:
- For each of the 9 features, shuffle that feature's values across all test profiles while keeping others fixed
- Measure how much the ROC-AUC drops — bigger drop = more important feature
- Rank all 9 features from most to least impactful

Also compute **mean absolute z-scores** per feature for stressed vs. healthy profiles (a simpler proxy for feature separability).

### 2b. Deep Dive: High-Impact Features

The user has observed that these features seem to carry heavy weight. For each one, report:

1. **Income level (avg_income)** — How does the model treat high vs. low income? What z-score range does the model see? At what income threshold does the model shift from "risky" to "safe"? How does the new `high_burn` archetype affect this?

2. **Income volatility** — This isn't a direct feature, but income variance feeds into `expense_volatility` and `savings_trend` indirectly. How sensitive is the model to unstable income? What happens when income is stable months 1-6 but jumps in months 7-12 (like the screenshot input)?

3. **Spending volatility (expense_volatility)** — What's the z-score threshold where this feature flips from "safe" to "risky"? How much does it contribute to the final prediction vs. other features?

4. **Expense ratio** — The dominant feature for stress prediction. Show the relationship between expense_ratio values and predicted risk probability. Where is the inflection point?

5. **Net cash flow vs. savings_months** — These are correlated. Does the model treat them redundantly or does each add unique predictive power?

### 2c. Feature Interaction Effects

Show how feature COMBINATIONS affect predictions. Create a small grid:
- High income + high expense_ratio → what risk?
- Low income + low expense_ratio → what risk?
- High income + low expense_ratio → what risk?
- Low income + high expense_ratio → what risk?

This reveals whether the model uses additive logic or has learned non-linear interactions.

### 2d. Report Format

Save the report to `.tmp/feature_weight_report.md` with:
- Tables for quantitative results
- Clear interpretation in plain language
- Actionable insights (e.g., "expense_ratio dominates — consider if the model over-relies on this single feature")

---

## Constraints (from CLAUDE.md)

- **Locked architecture:** Hidden layers 128 -> 64 (ReLU + Dropout) -> Sigmoid. Do NOT change.
- **No new files** unless absolutely necessary. Modify existing files.
- **Keep it simple** — this is a learning project, not production.
- Run on Windows. Use `py` not `python` for commands.
- Use `.tmp/` for any scratch scripts.
- Do NOT modify the model architecture, feature set, or training data. Only fix the inference pipeline bug.
- After fixing, the model does NOT need to be retrained — this is a serving-layer bug, not a training bug.

## Key Files

| File | Role |
|------|------|
| `backend/ml/predictor.py` | Inference orchestration — `_build_dataframe()` is where noise injection lives (lines 210-212) |
| `backend/data/feature_engineering.py` | `engineer_features()` computes the 9 features from a DataFrame |
| `backend/api/schemas.py` | Pydantic request/response models |
| `backend/api/routes.py` | Thin API route handlers |
| `backend/config/settings.py` | All configuration (archetypes, hyperparameters) |
| `models/scaler_stats.json` | Scaler mean/scale for z-score normalization |
| `frontend/app.js` | Frontend JavaScript — check how months 7-12 are sent |

## Success Criteria

1. User-provided detailed monthly data passes through `_build_dataframe()` UNMODIFIED (no noise applied)
2. Simple constant inputs (e.g., same income/expenses every month) still get noise injection to avoid OOD `expense_volatility=0`
3. All 5 regression test cases (A-E) remain within their expected ranges
4. Feature weight report is saved to `.tmp/feature_weight_report.md`