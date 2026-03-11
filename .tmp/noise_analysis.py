"""
Analyze how the 15% noise injection affects Cases B and D
when income ~ expenses. Run multiple seeds to show variance.
"""
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

def predict_case(income, expenses, credit, seed):
    rng = np.random.default_rng(seed)
    rows = []
    cumulative_savings = 0.0
    for i in range(12):
        inc_noisy = income * rng.normal(1.0, 0.15)
        exp_noisy = expenses * rng.normal(1.0, 0.15)
        monthly_net = inc_noisy - exp_noisy
        cumulative_savings = max(0.0, cumulative_savings + monthly_net)
        rows.append({
            'month': i + 1,
            'income': round(inc_noisy, 2),
            'total_expenses': round(exp_noisy, 2),
            'savings': round(cumulative_savings, 2),
            'credit_score': credit,
        })
    profile_df = pd.DataFrame(rows)
    features = engineer_features(profile_df)
    fv = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean) / scale
    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())
    return prob, features

def predict_no_noise(income, expenses, credit):
    """Predict with constant monthly values (no noise)."""
    rows = []
    cumulative_savings = 0.0
    for i in range(12):
        monthly_net = income - expenses
        cumulative_savings = max(0.0, cumulative_savings + monthly_net)
        rows.append({
            'month': i + 1,
            'income': round(income, 2),
            'total_expenses': round(expenses, 2),
            'savings': round(cumulative_savings, 2),
            'credit_score': credit,
        })
    profile_df = pd.DataFrame(rows)
    features = engineer_features(profile_df)
    fv = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
    scaled = (fv - mean) / scale
    with torch.no_grad():
        t = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        prob = float(model(t).item())
    return prob, features


print("=" * 70)
print("NOISE IMPACT ANALYSIS")
print("=" * 70)

cases = {
    "B": {"income": 3000, "expenses": 3000, "credit": 680},
    "D": {"income": 10000, "expenses": 10000, "credit": 700},
}

for label, c in cases.items():
    inc, exp, cred = c["income"], c["expenses"], c["credit"]
    print(f"\n  Case {label}: income=${inc}, expenses=${exp}, credit={cred}")

    # No noise
    prob_nn, feat_nn = predict_no_noise(inc, exp, cred)
    print(f"\n  WITHOUT noise:")
    print(f"    expense_ratio={feat_nn['expense_ratio']:.3f}, NCF=${feat_nn['net_cash_flow']:.0f}, savings_months={feat_nn['savings_months']:.2f}")
    print(f"    expense_volatility={feat_nn['expense_volatility']:.4f} (z={(feat_nn['expense_volatility'] - mean[6])/scale[6]:.2f})")
    print(f"    Risk: {prob_nn*100:.1f}%")

    # With noise - multiple seeds
    print(f"\n  WITH 15% noise (20 random seeds):")
    probs = []
    for seed in range(20):
        prob, feat = predict_case(inc, exp, cred, seed)
        probs.append(prob)
    probs = np.array(probs)
    print(f"    Mean risk:   {probs.mean()*100:.1f}%")
    print(f"    Std risk:    {probs.std()*100:.1f}%")
    print(f"    Min risk:    {probs.min()*100:.1f}%")
    print(f"    Max risk:    {probs.max()*100:.1f}%")
    print(f"    Median risk: {np.median(probs)*100:.1f}%")

    # Show a few seeds
    print(f"\n  Sample seeds:")
    for seed in [0, 1, 2, 3, 4]:
        prob, feat = predict_case(inc, exp, cred, seed)
        print(f"    seed={seed}: risk={prob*100:.1f}%, exp_ratio={feat['expense_ratio']:.3f}, NCF=${feat['net_cash_flow']:.0f}, sav_mo={feat['savings_months']:.2f}")

print("\n" + "=" * 70)
print("ANALYSIS: Training data expense_ratio distribution near 1.0")
print("=" * 70)

from backend.data.feature_engineering import build_feature_matrix
df = pd.read_csv("data/synthetic_train.csv")
X, y, _ = build_feature_matrix(df)

er = X[:, 3]  # expense_ratio
# Profiles with expense_ratio between 0.9 and 1.1
mask_near1 = (er >= 0.9) & (er <= 1.1)
n_near1 = mask_near1.sum()
stress_near1 = y[mask_near1].mean()
print(f"\n  Profiles with expense_ratio 0.9-1.1: {n_near1} ({n_near1/len(er):.1%})")
print(f"  Stress rate in this group: {stress_near1:.1%}")

mask_above1 = er > 1.0
stress_above1 = y[mask_above1].mean()
print(f"\n  Profiles with expense_ratio > 1.0: {mask_above1.sum()} ({mask_above1.sum()/len(er):.1%})")
print(f"  Stress rate in this group: {stress_above1:.1%}")

mask_95_105 = (er >= 0.95) & (er <= 1.05)
stress_95_105 = y[mask_95_105].mean()
print(f"\n  Profiles with expense_ratio 0.95-1.05: {mask_95_105.sum()}")
print(f"  Stress rate in this group: {stress_95_105:.1%}")
