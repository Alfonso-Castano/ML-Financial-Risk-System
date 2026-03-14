"""Test that the noise removal fix eliminates the paradox.

Screenshot A: 6 months $2,150/$2,150, credit 700 (equal income/expenses)
Screenshot B: 6 months $2,150/$2,100, credit 700 (expenses $50 lower)

Expected: A risk >= B risk (spending more should be riskier)
Expected: Computed features match user input exactly
"""
import sys
import json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, ".")

from backend.ml.model import FinancialRiskModel
from backend.data.feature_engineering import engineer_features, FEATURE_NAMES

# Load model and scaler
model = FinancialRiskModel(input_size=9)
model.load_state_dict(torch.load("models/latest_model.pth", weights_only=True))
model.eval()

with open("models/scaler_stats.json") as f:
    stats = json.load(f)
mean_arr = np.array(stats["mean"], dtype=np.float32)
scale_arr = np.array(stats["scale"], dtype=np.float32)

EXPENSE_VOL_TRAIN_MEAN = 0.1497


def predict_fixed(inc, exp, credit, n_months=6):
    """Predict using the fixed pipeline (no noise, clamped expense_volatility)."""
    rows = []
    cum_savings = 0.0
    for i in range(n_months):
        net = inc - exp
        cum_savings = max(0.0, cum_savings + net)
        rows.append({
            "month": i + 1,
            "income": round(inc, 2),
            "total_expenses": round(exp, 2),
            "savings": round(cum_savings, 2),
            "credit_score": credit,
        })
    df = pd.DataFrame(rows)
    features = engineer_features(df)

    # Clamp expense_volatility for model input (same as predictor.py fix)
    model_features = {
        name: (EXPENSE_VOL_TRAIN_MEAN if name == 'expense_volatility'
               and features[name] < 0.01 else features[name])
        for name in FEATURE_NAMES
    }
    fv = np.array([model_features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean_arr) / scale_arr

    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())

    return prob * 100, features


# === PARADOX TEST ===
print("=" * 60)
print("PARADOX TEST: Screenshot A vs Screenshot B")
print("=" * 60)

score_a, feats_a = predict_fixed(2150, 2150, 700)
score_b, feats_b = predict_fixed(2150, 2100, 700)

print(f"\nScreenshot A ($2,150/$2,150):")
print(f"  Risk Score: {score_a:.1f}%")
print(f"  avg_income:  ${feats_a['avg_income']:,.2f}  (expected: $2,150.00)")
print(f"  avg_expenses: ${feats_a['avg_expenses']:,.2f}  (expected: $2,150.00)")
print(f"  expense_ratio: {feats_a['expense_ratio']:.4f}  (expected: 1.0000)")
print(f"  net_cash_flow: ${feats_a['net_cash_flow']:,.2f}  (expected: $0.00)")
print(f"  expense_volatility: {feats_a['expense_volatility']:.4f}  (honest value for display)")

print(f"\nScreenshot B ($2,150/$2,100):")
print(f"  Risk Score: {score_b:.1f}%")
print(f"  avg_income:  ${feats_b['avg_income']:,.2f}  (expected: $2,150.00)")
print(f"  avg_expenses: ${feats_b['avg_expenses']:,.2f}  (expected: $2,100.00)")
print(f"  expense_ratio: {feats_b['expense_ratio']:.4f}  (expected: 0.9767)")
print(f"  net_cash_flow: ${feats_b['net_cash_flow']:,.2f}  (expected: $50.00)")
print(f"  expense_volatility: {feats_b['expense_volatility']:.4f}  (honest value for display)")

print(f"\n--- VERIFICATION ---")
# Check 1: A risk >= B risk (logically correct ordering)
paradox_ok = score_a >= score_b
print(f"  A ({score_a:.1f}%) >= B ({score_b:.1f}%): {'PASS' if paradox_ok else 'FAIL'}")

# Check 2: Computed features match input exactly
income_a_ok = abs(feats_a['avg_income'] - 2150) < 0.01
income_b_ok = abs(feats_b['avg_income'] - 2150) < 0.01
exp_a_ok = abs(feats_a['avg_expenses'] - 2150) < 0.01
exp_b_ok = abs(feats_b['avg_expenses'] - 2100) < 0.01
ratio_a_ok = abs(feats_a['expense_ratio'] - 1.0) < 0.001
ratio_b_ok = abs(feats_b['expense_ratio'] - 2100/2150) < 0.001
ncf_a_ok = abs(feats_a['net_cash_flow'] - 0) < 0.01
ncf_b_ok = abs(feats_b['net_cash_flow'] - 50) < 0.01

print(f"  A avg_income = $2,150: {'PASS' if income_a_ok else 'FAIL'}")
print(f"  B avg_income = $2,150: {'PASS' if income_b_ok else 'FAIL'}")
print(f"  A avg_expenses = $2,150: {'PASS' if exp_a_ok else 'FAIL'}")
print(f"  B avg_expenses = $2,100: {'PASS' if exp_b_ok else 'FAIL'}")
print(f"  A expense_ratio = 1.00: {'PASS' if ratio_a_ok else 'FAIL'}")
print(f"  B expense_ratio = 0.977: {'PASS' if ratio_b_ok else 'FAIL'}")
print(f"  A net_cash_flow = $0: {'PASS' if ncf_a_ok else 'FAIL'}")
print(f"  B net_cash_flow = $50: {'PASS' if ncf_b_ok else 'FAIL'}")

all_pass = all([paradox_ok, income_a_ok, income_b_ok, exp_a_ok, exp_b_ok,
                ratio_a_ok, ratio_b_ok, ncf_a_ok, ncf_b_ok])
print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
