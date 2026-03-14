"""Regression test with the Option E fix applied (no noise + expense_volatility clamp)."""
import sys
import json
import numpy as np
import torch
import pandas as pd

sys.path.insert(0, ".")

from backend.ml.model import FinancialRiskModel
from backend.data.feature_engineering import engineer_features, FEATURE_NAMES

model = FinancialRiskModel(input_size=9)
model.load_state_dict(torch.load("models/latest_model.pth", weights_only=True))
model.eval()

with open("models/scaler_stats.json") as f:
    stats = json.load(f)
mean = np.array(stats["mean"], dtype=np.float32)
scale = np.array(stats["scale"], dtype=np.float32)

EXPENSE_VOL_TRAIN_MEAN = 0.1497

cases = {
    "A (deep deficit)":   {"income": 1000,  "expenses": 2500,  "credit": 650, "lo": 80, "hi": 100},
    "B (breaking even)":  {"income": 3000,  "expenses": 3000,  "credit": 680, "lo": 55, "hi": 85},
    "C (comfortable)":    {"income": 6000,  "expenses": 3500,  "credit": 720, "lo": 0,  "hi": 30},
    "D (high burn)":      {"income": 10000, "expenses": 10000, "credit": 700, "lo": 40, "hi": 70},
    "E (frugal)":         {"income": 2000,  "expenses": 1200,  "credit": 660, "lo": 0,  "hi": 40},
}


def predict_fixed(inc, exp, cred, n_months=12):
    """Predict with Option E fix: no noise, clamp expense_volatility."""
    rows = []
    cum_savings = 0.0
    for i in range(n_months):
        net = inc - exp
        cum_savings = max(0.0, cum_savings + net)
        rows.append({
            "month": i + 1,
            "income": inc,
            "total_expenses": exp,
            "savings": round(cum_savings, 2),
            "credit_score": cred,
        })
    df = pd.DataFrame(rows)
    features = engineer_features(df)

    # Apply the clamp (same as predictor.py fix)
    model_features = {
        name: (EXPENSE_VOL_TRAIN_MEAN if name == 'expense_volatility'
               and features[name] < 0.01 else features[name])
        for name in FEATURE_NAMES
    }
    fv = np.array([model_features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean) / scale

    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())

    return prob * 100, features


print("=" * 70)
print("REGRESSION TEST (Option E: no noise + expense_volatility clamp)")
print("=" * 70)

all_pass = True
for name, c in cases.items():
    pct, feats = predict_fixed(c["income"], c["expenses"], c["credit"])
    lo, hi = c["lo"], c["hi"]
    passed = lo <= pct <= hi
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {name}: {pct:5.1f}%  (expected {lo}-{hi}%)  [{status}]")
    print(f"    expense_ratio={feats['expense_ratio']:.3f}  savings_months={feats['savings_months']:.2f}  NCF=${feats['net_cash_flow']:,.0f}  exp_vol={feats['expense_volatility']:.4f}")

print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
