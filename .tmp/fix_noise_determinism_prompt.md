# Fix Noise Injection Determinism Bugs

## Your Role

You are the **orchestrator**. Delegate to sub-agents, coordinate their findings, and make architectural decisions. Do NOT do the heavy lifting yourself.

**Sub-agents to use:**
- **Debugger agent** (`subagent_type: debugger`) -- for root cause analysis, hypothesis testing, reproducing bugs
- **Explore agent** (`subagent_type: Explore`) -- for codebase searches, tracing data flows
- `/senior-data-scientist` skill -- for evaluating statistical properties of proposed fixes
- `/senior-data-engineer` skill -- for data pipeline integrity checks

**Execution discipline:** Complete each step and report findings before moving to the next.

---

## The Bugs (from live application testing)

Two nearly identical inputs produce **paradoxically opposite** risk scores:

### Screenshot A (Risk Score: 78, Probability: 78.1%)
- Months 1-6: Income $2,150 / Expenses **$2,150** (equal)
- Months 7-12: NOT filled (placeholder values "4500"/"3200" shown in gray, but inputs are empty)
- Credit Score: 700
- Computed features returned:
  - Avg Monthly Income: **$2,261** (wrong -- should be $2,150 if 6 months sent)
  - Avg Monthly Expenses: **$2,185** (wrong -- should be $2,150)
  - Expense Ratio: **0.97** (wrong -- should be 1.00)
  - Net Cash Flow: **$76** (wrong -- should be $0)

### Screenshot B (Risk Score: 96, Probability: 95.6%)
- Months 1-6: Income $2,150 / Expenses **$2,100** (expenses $50 LOWER)
- Months 7-12: NOT filled (same placeholder situation)
- Credit Score: 700
- Computed features returned:
  - Avg Monthly Income: **$1,939** (wrong -- should be $2,150)
  - Avg Monthly Expenses: **$2,017** (wrong -- should be $2,100)
  - Expense Ratio: **1.04** (wrong -- should be 0.977)
  - Net Cash Flow: **-$78** (wrong -- should be +$50)

### Why this is broken

1. **Illogical ordering:** Screenshot B has LOWER expenses than A, yet gets a HIGHER risk score (95.6% vs 78.1%). A user spending less should never be rated riskier than one spending more, all else equal.

2. **Computed features are wrong:** The displayed averages don't match the user's input. For Screenshot A, the user entered $2,150/$2,150 for 6 months -- the average should be exactly $2,150/$2,150, not $2,261/$2,185.

3. **Root cause (confirmed):** The 15% noise injection in `predictor.py:_build_dataframe()` corrupts constant-value inputs with random multipliers derived from a deterministic seed. Different expense values ($2,150 vs $2,100) produce different seeds, which draw different noise multipliers. Screenshot B's seed happens to draw multipliers that push income DOWN and expenses UP, flipping the expense_ratio above 1.0, while Screenshot A's seed draws more balanced multipliers. This creates the paradox: random noise artifacts, not the actual financial data, drive the risk score.

---

## Step 1: Identify the Source of Error

**Spawn a Debugger agent** to confirm the root cause. The agent should:

1. Read the noise injection code in `backend/ml/predictor.py` (`_build_dataframe()`, lines 195-240)
2. Read the conditional noise check we recently added (CV threshold at 0.01)
3. **Reproduce the bug:** Write a script in `.tmp/` that:
   - Sends both Screenshot A input (6 months of $2,150/$2,150, credit 700) and Screenshot B input (6 months of $2,150/$2,100, credit 700) through `_build_dataframe()` + `engineer_features()`
   - Prints the per-month noised values for both cases
   - Prints computed features for both cases
   - Confirms that the noise is the sole cause of the paradox (expense_ratio flip from <1.0 to >1.0)
4. **Explain the seed mechanism:** Show how the hash-based seed produces different RNG streams for inputs that differ by only $50, causing divergent noise multipliers
5. **Check whether the CV threshold is working correctly:** Both Screenshot A and B have CV=0 for their expenses (all 6 months identical), so `needs_noise=true` is technically correct per the current logic. The bug isn't that noise is applied when it shouldn't be -- it's that noise-driven distortion produces illogical results for similar inputs.

**Expected finding:** The noise injection is working "as designed" (CV=0 triggers noise), but the DESIGN is flawed for this case. Constant-value inputs that differ by small amounts should produce proportionally similar risk scores, not divergent ones.

---

## Step 2: Identify Possible Solutions

Based on the debugger's findings, evaluate these candidate solutions:

### Option A: Remove noise injection entirely
- Accept that constant inputs produce `expense_volatility=0` (z-score = -4.55)
- The model handles this as an extreme value, which may actually be acceptable since:
  - expense_volatility has negligible permutation importance (AUC drop: -0.0012, per the feature weight report)
  - The model barely uses this feature
- **Pro:** Simplest fix. Computed features always match user input exactly. No paradoxes possible.
- **Con:** expense_volatility=0 is technically OOD. May affect edge cases.

### Option B: Reduce noise magnitude
- Use 5% noise instead of 15% to limit distortion while still producing nonzero expense_volatility
- **Pro:** Smaller distortions mean similar inputs produce similar outputs
- **Con:** Still produces wrong computed features. Still possible for noise to flip expense_ratio across 1.0 for borderline cases (just less likely).

### Option C: Fix the seed to be input-independent
- Use a constant seed (e.g., 42) instead of hashing the input, so all constant-value inputs get the SAME noise pattern
- **Pro:** Eliminates the paradox for similar inputs (both would get same multipliers)
- **Con:** Computed features still wrong. Doesn't fix the "displayed values don't match input" problem.

### Option D: Apply noise ONLY to expense_volatility computation, not to the raw values
- Pass raw user values through for avg_income, avg_expenses, NCF, etc.
- Only inject synthetic variance when computing the expense_volatility feature specifically
- **Pro:** Computed features match user input exactly. expense_volatility gets a non-zero value. No paradoxes.
- **Con:** Slightly more code complexity. savings_trend and final_savings also affected by noise currently.

### Option E: Hybrid -- remove noise, clamp expense_volatility to training floor
- Don't inject noise at all. If expense_volatility=0, replace it with the training distribution's minimum (e.g., 0.05) or mean (0.15) before scaling
- **Pro:** Simple. Computed features correct. No paradox. expense_volatility gets a reasonable value.
- **Con:** Slightly dishonest about the feature value, but expense_volatility is nearly unused by the model anyway.

For each solution, evaluate:
- Does it eliminate the paradox (similar inputs -> similar outputs)?
- Are displayed computed features correct (match user input)?
- Does it preserve model accuracy for the simple-input case?
- How complex is the implementation?
- Does it respect the locked architecture?

---

## Step 3: Choose the Best Solution

Use `/senior-data-scientist` to evaluate the top 2-3 solutions. The evaluation should consider:

1. **The feature weight report finding:** expense_volatility has permutation importance of -0.0012 (effectively zero). The model does NOT use this feature meaningfully. This dramatically changes the cost/benefit of protecting against OOD expense_volatility.
2. **User trust:** Displaying computed features that don't match the user's input destroys confidence in the system. This is a learning project where transparency matters.
3. **Determinism:** The same-ish inputs must produce same-ish outputs. Random-looking behavior is unacceptable.

Present the chosen solution with clear rationale.

---

## Step 4: Create a Plan

Write a step-by-step implementation plan:
1. Which file(s) to modify (expect: `backend/ml/predictor.py`)
2. What the change looks like (pseudocode or diff)
3. How to test both the paradox case and the regression cases
4. What risks exist

---

## Step 5: Execute the Fix

Implement the chosen solution. Then verify by:

1. **Paradox test:** Run Screenshot A ($2,150/$2,150) and Screenshot B ($2,150/$2,100) through the fixed pipeline.
   - Confirm avg_income = exactly $2,150 for both
   - Confirm avg_expenses = exactly $2,150 and $2,100 respectively
   - Confirm Screenshot A risk >= Screenshot B risk (the logically correct ordering)
   - Confirm expense_ratio = 1.00 and 0.977 respectively (exact, no noise)

2. **Varied-input test:** Run the 12-month detailed input from the previous bug report (months 1-6: varied, months 7-12: $4,500/$3,200). Confirm values pass through unmodified (this was already fixed).

3. **Regression test:** Run `.tmp/regression_test.py` (5-case test A-E). Report any changes in scores.

---

## Constraints (from CLAUDE.md)

- **Locked architecture:** 128 -> 64 (ReLU + Dropout) -> Sigmoid. Do NOT change.
- **No new files** unless absolutely necessary. Modify existing files.
- **Keep it simple** -- this is a learning project, not production.
- Use `py` not `python` for commands (Windows).
- Use `.tmp/` for scratch scripts.
- Do NOT modify the model architecture, feature set, or training data.
- Do NOT retrain the model -- this is a serving-layer fix.

## Key Files

| File | Role |
|------|------|
| `backend/ml/predictor.py` | `_build_dataframe()` -- noise injection lives here (lines 202-226) |
| `backend/data/feature_engineering.py` | `engineer_features()` computes the 9 features |
| `backend/api/schemas.py` | Pydantic request/response models |
| `frontend/index.html` | Month grid with placeholder values on months 7-12 (lines 116-148) |
| `models/scaler_stats.json` | Scaler mean/scale for z-score normalization |
| `.tmp/feature_weight_report.md` | Feature importance analysis (expense_volatility is negligible) |
| `.tmp/regression_test.py` | 5-case regression test |

## Success Criteria

1. Screenshot A ($2,150/$2,150) has risk >= Screenshot B ($2,150/$2,100) -- logically correct ordering
2. Computed features displayed to the user match their actual input exactly (no noise distortion)
3. All 5 regression test cases (A-E) remain within their expected ranges (or improve)
4. expense_volatility gets a reasonable non-OOD value even for constant inputs
