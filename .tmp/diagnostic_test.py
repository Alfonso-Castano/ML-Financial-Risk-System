"""Diagnostic script: Phase 1 - Quantify the leniency problem."""
import requests
import json
import numpy as np

BASE_URL = "http://127.0.0.1:8000"

# Load scaler stats for z-score analysis
with open("models/scaler_stats.json") as f:
    scaler = json.load(f)
mean = np.array(scaler["mean"])
scale = np.array(scaler["scale"])
feature_names = scaler["feature_names"]

# Test cases: (name, income, expenses, debt, credit, months, human_expectation)
CASES = {
    "A: Deep deficit":          (1000, 2500, 100, 650, 6, ">80%"),
    "B: Tight breaking even":   (3000, 2800, 200, 680, 6, "55-70%"),
    "C: Comfortable surplus":   (6000, 3500, 400, 720, 6, "<30%"),
    "D: High income high burn": (10000, 9500, 500, 700, 6, "40-60%"),
    "E: Low income frugal":     (2000, 1200, 100, 660, 6, "25-40%"),
}

def build_request(income, expenses, debt, credit, n_months):
    return {
        "months": [{"income": income, "expenses": expenses, "debt_payment": debt}] * n_months,
        "credit_score": credit,
    }

print("=" * 90)
print("PHASE 1: MODEL LENIENCY DIAGNOSTIC")
print("=" * 90)

results = {}
for name, (inc, exp, debt, credit, months, human) in CASES.items():
    req = build_request(inc, exp, debt, credit, months)
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=req, timeout=10)
        data = resp.json()
        results[name] = data
        
        print(f"\n{'─' * 90}")
        print(f"CASE {name}")
        print(f"  Input: income=${inc}, expenses=${exp}, debt=${debt}, credit={credit}, {months}mo")
        print(f"  Human expectation: {human}")
        print(f"  Model output:      {data['risk_score']}% ({data['risk_category']})")
        print(f"  Raw probability:   {data['probability']}")
        
        # Compute z-scores
        feats = data['computed_features']
        feat_vec = np.array([feats[fn] for fn in feature_names])
        z_scores = (feat_vec - mean) / scale
        
        print(f"  Computed features:")
        for fn, val, z in zip(feature_names, feat_vec, z_scores):
            flag = " ⚠️ OOD" if abs(z) > 2.0 else ""
            print(f"    {fn:25s} = {val:12.2f}  (z = {z:+6.2f}){flag}")
            
        # Gap analysis
        print(f"  LENIENCY GAP: model={data['risk_score']}% vs human={human}")
        
    except Exception as e:
        print(f"\n  CASE {name}: ERROR - {e}")

print(f"\n{'=' * 90}")
print("SUMMARY TABLE")
print("=" * 90)
print(f"{'Case':<30} {'Model':>8} {'Category':>10} {'Human':>12} {'Gap':>20}")
print("─" * 90)
for name, (inc, exp, debt, credit, months, human) in CASES.items():
    if name in results:
        d = results[name]
        print(f"{name:<30} {d['risk_score']:>7.1f}% {d['risk_category']:>10} {human:>12} {'INVESTIGATE' if d['risk_category'] != 'high' and '>80%' in human else ''}")

