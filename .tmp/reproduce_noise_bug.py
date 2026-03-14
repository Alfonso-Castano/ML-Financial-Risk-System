"""
Reproduce the noise injection determinism bug in predictor.py.

The bug: two nearly identical inputs produce paradoxically different risk-relevant
features because different input values yield different hash-based seeds, which
drive different RNG streams, which produce different noise multipliers.

Case A: $2,150 income / $2,150 expenses  (income == expenses)
Case B: $2,150 income / $2,100 expenses  (income > expenses by $50)

Expected: Case B should have a LOWER expense_ratio than Case A (more headroom).
Actual:   The different seeds can flip Case B to a WORSE expense_ratio than Case A.
"""

import sys
import os
import numpy as np
import pandas as pd

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data.feature_engineering import engineer_features

# ---------------------------------------------------------------------------
# Minimal stand-ins for the Pydantic request objects used in predictor.py
# ---------------------------------------------------------------------------

class MonthData:
    def __init__(self, income: float, expenses: float):
        self.income = income
        self.expenses = expenses


class FakeRequest:
    def __init__(self, months, credit_score: int):
        self.months = months
        self.credit_score = credit_score


# ---------------------------------------------------------------------------
# Replicate _build_dataframe() exactly as written in predictor.py (lines 195-240)
# ---------------------------------------------------------------------------

def build_dataframe(request) -> pd.DataFrame:
    """Exact copy of Predictor._build_dataframe() with no modifications."""

    # Lines 196-200: hash-based seed
    seed_value = int(abs(hash((
        tuple((m.income, m.expenses) for m in request.months),
        request.credit_score
    )))) % (2**31)
    rng = np.random.default_rng(seed_value)

    # Lines 206-209: CV check
    raw_expenses = [m.expenses for m in request.months]
    mean_exp = np.mean(raw_expenses) if raw_expenses else 0.0
    cv = float(np.std(raw_expenses) / mean_exp) if mean_exp > 0 else 0.0
    needs_noise = cv < 0.01

    rows = []
    cumulative_savings = 0.0

    for i, month in enumerate(request.months):
        if needs_noise:
            # Lines 219-221
            inference_variance = 0.15
            income = month.income * rng.normal(1.0, inference_variance)
            expenses = month.expenses * rng.normal(1.0, inference_variance)
        else:
            income = month.income
            expenses = month.expenses

        monthly_net = income - expenses
        cumulative_savings = max(0.0, cumulative_savings + monthly_net)

        rows.append({
            'month': i + 1,
            'income': round(income, 2),
            'total_expenses': round(expenses, 2),
            'savings': round(cumulative_savings, 2),
            'credit_score': request.credit_score,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build the two requests
# ---------------------------------------------------------------------------

N_MONTHS = 6
CREDIT_SCORE = 700

months_A = [MonthData(income=2150.0, expenses=2150.0) for _ in range(N_MONTHS)]
months_B = [MonthData(income=2150.0, expenses=2100.0) for _ in range(N_MONTHS)]

request_A = FakeRequest(months=months_A, credit_score=CREDIT_SCORE)
request_B = FakeRequest(months=months_B, credit_score=CREDIT_SCORE)


# ---------------------------------------------------------------------------
# Compute seeds (so we can show they differ)
# ---------------------------------------------------------------------------

seed_A = int(abs(hash((
    tuple((m.income, m.expenses) for m in request_A.months),
    request_A.credit_score
)))) % (2**31)

seed_B = int(abs(hash((
    tuple((m.income, m.expenses) for m in request_B.months),
    request_B.credit_score
)))) % (2**31)


# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------

df_A = build_dataframe(request_A)
df_B = build_dataframe(request_B)

features_A = engineer_features(df_A)
features_B = engineer_features(df_B)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

DIVIDER = "=" * 65

print(DIVIDER)
print("NOISE INJECTION DETERMINISM BUG — REPRODUCTION SCRIPT")
print(DIVIDER)

print("\n--- SEED DERIVATION ---")
print(f"  Case A seed  (income=$2150, expenses=$2150): {seed_A}")
print(f"  Case B seed  (income=$2150, expenses=$2100): {seed_B}")
print(f"  Seeds are different: {seed_A != seed_B}")

print("\n--- CASE A: $2,150 income / $2,150 expenses per month ---")
print(f"  {'Month':<7} {'Noised Income':>14} {'Noised Expenses':>16} {'Monthly Net':>12}")
for _, row in df_A.iterrows():
    net = row['income'] - row['total_expenses']
    print(f"  {int(row['month']):<7} {row['income']:>14.2f} {row['total_expenses']:>16.2f} {net:>12.2f}")

print("\n--- CASE B: $2,150 income / $2,100 expenses per month ---")
print(f"  {'Month':<7} {'Noised Income':>14} {'Noised Expenses':>16} {'Monthly Net':>12}")
for _, row in df_B.iterrows():
    net = row['income'] - row['total_expenses']
    print(f"  {int(row['month']):<7} {row['income']:>14.2f} {row['total_expenses']:>16.2f} {net:>12.2f}")

print("\n--- ENGINEERED FEATURES COMPARISON ---")
print(f"  {'Feature':<22} {'Case A':>12} {'Case B':>12}  Note")
print(f"  {'-'*22} {'-'*12} {'-'*12}  {'-'*30}")

for name in ['avg_income', 'avg_expenses', 'expense_ratio', 'net_cash_flow',
             'savings_months', 'final_savings', 'expense_volatility', 'savings_trend']:
    va = features_A[name]
    vb = features_B[name]
    note = ""
    if name == 'expense_ratio':
        note = "<-- KEY FEATURE"
    fmt = "  {:<22} {:>12.4f} {:>12.4f}  {}"
    print(fmt.format(name, va, vb, note))

print()
print(DIVIDER)
print("THE PARADOX")
print(DIVIDER)

ratio_A = features_A['expense_ratio']
ratio_B = features_B['expense_ratio']

print(f"\n  Case A raw expenses = $2,150/mo  ->  noised expense_ratio = {ratio_A:.4f}")
print(f"  Case B raw expenses = $2,100/mo  ->  noised expense_ratio = {ratio_B:.4f}")
print()

if ratio_B > ratio_A:
    print("  PARADOX CONFIRMED: Case B (LOWER expenses) has a HIGHER expense_ratio.")
    print("  The $50/month cheaper profile looks MORE stressed to the model.")
    print()
    print("  Root cause:")
    print("    - Different inputs -> different hash seeds -> different RNG streams")
    print(f"    - seed_A={seed_A} drives noise multipliers that happen to keep")
    print( "      expenses below income for Case A.")
    print(f"    - seed_B={seed_B} drives noise multipliers that push expenses above")
    print( "      income for Case B, even though raw expenses were $50 lower.")
    print()
    print("  Impact: A user who spends $50 LESS per month can receive a WORSE")
    print("  risk score solely due to an accident of hash arithmetic.")
elif ratio_A > ratio_B:
    print("  Case B (lower expenses) correctly has a lower expense_ratio this run.")
    print("  NOTE: the paradox is hash-value dependent and may not trigger on every")
    print("  Python version/platform. The structural fragility remains: two inputs")
    print("  that differ by $50 share no relationship between their RNG streams,")
    print("  so ordering is arbitrary and may flip on a different interpreter.")
else:
    print("  expense_ratios are equal this run (uncommon). The structural bug")
    print("  (unrelated RNG seeds for similar inputs) still exists.")

print()
print("  Fix direction: replace hash-based seed with a seed derived from a")
print("  CANONICAL representation (e.g., mean income/expenses rounded to the")
print("  nearest $100), so that inputs that differ by small amounts land on")
print("  the same RNG stream and produce monotonically consistent features.")
print(DIVIDER)
