"""Verify the conditional noise injection fix for both cases."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from types import SimpleNamespace
from backend.ml.predictor import Predictor
from backend.data.feature_engineering import engineer_features

# ── Helper: build a mock request ──
def mock_request(months_data, credit_score):
    months = [SimpleNamespace(income=inc, expenses=exp) for inc, exp in months_data]
    return SimpleNamespace(months=months, credit_score=credit_score)

# ── Instantiate predictor (loads model) ──
predictor = Predictor()

print("=" * 70)
print("TEST 1: Detailed 12-month input (should NOT have noise)")
print("=" * 70)

detailed_months = [
    (2100, 2000), (2050, 2020), (2150, 2100),
    (2000, 2050), (2060, 2000), (2100, 2050),
    (4500, 3200), (4500, 3200), (4500, 3200),
    (4500, 3200), (4500, 3200), (4500, 3200),
]
req = mock_request(detailed_months, 700)
df = predictor._build_dataframe(req)
features = engineer_features(df)

expected_avg_income = (2100+2050+2150+2000+2060+2100+4500*6) / 12
expected_avg_expenses = (2000+2020+2100+2050+2000+2050+3200*6) / 12
expected_ncf = expected_avg_income - expected_avg_expenses

print(f"  avg_income:    {features['avg_income']:.2f}  (expected: {expected_avg_income:.2f})")
print(f"  avg_expenses:  {features['avg_expenses']:.2f}  (expected: {expected_avg_expenses:.2f})")
print(f"  net_cash_flow: {features['net_cash_flow']:.2f}  (expected: {expected_ncf:.2f})")
print(f"  expense_ratio: {features['expense_ratio']:.4f}  (expected: {expected_avg_expenses/expected_avg_income:.4f})")
print(f"  expense_vol:   {features['expense_volatility']:.4f}")

# Check exact match (no noise)
assert abs(features['avg_income'] - expected_avg_income) < 0.01, "FAIL: avg_income mismatch"
assert abs(features['avg_expenses'] - expected_avg_expenses) < 0.01, "FAIL: avg_expenses mismatch"
assert features['net_cash_flow'] > 0, "FAIL: NCF should be positive"
print("  PASS - values match hand-calculated expectations exactly")

print()
print("=" * 70)
print("TEST 2: Constant input (should HAVE noise)")
print("=" * 70)

constant_months = [(10000, 10000)] * 12
req2 = mock_request(constant_months, 700)
df2 = predictor._build_dataframe(req2)
features2 = engineer_features(df2)

print(f"  avg_income:       {features2['avg_income']:.2f}  (raw: 10000)")
print(f"  avg_expenses:     {features2['avg_expenses']:.2f}  (raw: 10000)")
print(f"  expense_vol:      {features2['expense_volatility']:.4f}  (should be > 0)")
print(f"  net_cash_flow:    {features2['net_cash_flow']:.2f}")

assert features2['expense_volatility'] > 0, "FAIL: expense_volatility should be > 0 with noise"
# Values should differ from raw 10000 due to noise
income_vals = df2['income'].tolist()
has_variance = any(abs(v - 10000) > 1 for v in income_vals)
assert has_variance, "FAIL: noise not applied to constant input"
print("  PASS - noise injected for constant values, expense_volatility > 0")

print()
print("=" * 70)
print("TEST 3: 6-month detailed input (should NOT have noise)")
print("=" * 70)

six_months = [
    (2100, 2000), (2050, 2020), (2150, 2100),
    (2000, 2050), (2060, 2000), (2100, 2050),
]
req3 = mock_request(six_months, 700)
df3 = predictor._build_dataframe(req3)
features3 = engineer_features(df3)

expected_avg_inc_6 = (2100+2050+2150+2000+2060+2100) / 6
expected_avg_exp_6 = (2000+2020+2100+2050+2000+2050) / 6

print(f"  avg_income:    {features3['avg_income']:.2f}  (expected: {expected_avg_inc_6:.2f})")
print(f"  avg_expenses:  {features3['avg_expenses']:.2f}  (expected: {expected_avg_exp_6:.2f})")

assert abs(features3['avg_income'] - expected_avg_inc_6) < 0.01, "FAIL: 6-month avg_income mismatch"
assert abs(features3['avg_expenses'] - expected_avg_exp_6) < 0.01, "FAIL: 6-month avg_expenses mismatch"
print("  PASS - 6-month detailed data passes through unmodified")

print()
print("ALL TESTS PASSED")
