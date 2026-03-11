"""Regression test: all 5 cases deterministic + noise variance analysis."""
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
mean = np.array(stats["mean"], dtype=np.float32)
scale = np.array(stats["scale"], dtype=np.float32)

cases = {
    "A (deep deficit)":   {"income": 1000,  "expenses": 2500,  "credit": 650, "lo": 80, "hi": 100},
    "B (breaking even)":  {"income": 3000,  "expenses": 3000,  "credit": 680, "lo": 55, "hi": 70},
    "C (comfortable)":    {"income": 6000,  "expenses": 3500,  "credit": 720, "lo": 0,  "hi": 30},
    "D (high burn)":      {"income": 10000, "expenses": 10000, "credit": 700, "lo": 40, "hi": 60},
    "E (frugal)":         {"income": 2000,  "expenses": 1200,  "credit": 660, "lo": 25, "hi": 40},
}


def predict_deterministic(inc, exp, cred):
    """Predict without noise — constant monthly values."""
    rows = []
    cum_savings = 0.0
    for i in range(12):
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
    fv = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean) / scale
    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())
    return prob * 100, features


def predict_noisy(inc, exp, cred, seed):
    """Predict with 15% noise injection."""
    rng = np.random.default_rng(seed)
    rows = []
    cum_savings = 0.0
    for i in range(12):
        inc_n = inc * rng.normal(1.0, 0.15)
        exp_n = exp * rng.normal(1.0, 0.15)
        cum_savings = max(0.0, cum_savings + inc_n - exp_n)
        rows.append({
            "month": i + 1,
            "income": round(inc_n, 2),
            "total_expenses": round(exp_n, 2),
            "savings": round(cum_savings, 2),
            "credit_score": cred,
        })
    df = pd.DataFrame(rows)
    features = engineer_features(df)
    fv = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean) / scale
    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())
    return prob * 100


# === DETERMINISTIC TEST ===
print("=" * 70)
print("DETERMINISTIC TEST (no noise)")
print("=" * 70)

all_pass = True
for name, c in cases.items():
    pct, feats = predict_deterministic(c["income"], c["expenses"], c["credit"])
    lo, hi = c["lo"], c["hi"]
    passed = lo <= pct <= hi
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {name}: {pct:5.1f}%  (expected {lo}-{hi}%)  [{status}]")
    print(f"    expense_ratio={feats['expense_ratio']:.3f}  savings_months={feats['savings_months']:.2f}  NCF=${feats['net_cash_flow']:,.0f}")

# === NOISE VARIANCE ===
print()
print("=" * 70)
print("NOISE VARIANCE (20 seeds)")
print("=" * 70)

for name, c in cases.items():
    scores = [predict_noisy(c["income"], c["expenses"], c["credit"], s) for s in range(20)]
    m, s = np.mean(scores), np.std(scores)
    lo, hi = c["lo"], c["hi"]
    print(f"  {name}: mean={m:5.1f}%  std={s:4.1f}%  range=[{min(scores):.1f}-{max(scores):.1f}%]  (target {lo}-{hi}%)")

print()
print("DETERMINISTIC RESULT:", "ALL PASS" if all_pass else "SOME FAILURES")
