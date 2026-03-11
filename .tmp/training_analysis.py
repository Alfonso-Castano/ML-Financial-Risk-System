"""Phase 2: Training data distribution analysis for leniency diagnosis."""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from backend.data.feature_engineering import build_feature_matrix, FEATURE_NAMES

# Load training data
print("Loading training data...")
df = pd.read_csv("data/synthetic_train.csv")
X, y, profile_ids = build_feature_matrix(df)
print(f"Loaded {len(profile_ids)} profiles, {int(y.sum())} stressed ({y.mean():.1%})")

# Create feature DataFrame
feat_df = pd.DataFrame(X, columns=FEATURE_NAMES)
feat_df['is_stressed'] = y
feat_df['profile_id'] = profile_ids

stressed = feat_df[feat_df['is_stressed'] == 1]
healthy = feat_df[feat_df['is_stressed'] == 0]
print(f"Stressed: {len(stressed)}, Healthy: {len(healthy)}")

# 1. Distribution analysis
print("\n" + "=" * 100)
print("1. FEATURE DISTRIBUTIONS BY CLASS")
print("=" * 100)

percentiles = [1, 5, 25, 50, 75, 95, 99]

for fn in FEATURE_NAMES:
    print(f"\n--- {fn} ---")
    for label, subset in [("Stressed", stressed), ("Healthy", healthy), ("All", feat_df)]:
        vals = subset[fn].values
        pcts = np.percentile(vals, percentiles)
        print(f"  {label:10s}: mean={np.mean(vals):10.2f}  std={np.std(vals):10.2f}  "
              f"min={np.min(vals):10.2f}  max={np.max(vals):10.2f}")
        pct_str = "  ".join([f"p{p}={v:.2f}" for p, v in zip(percentiles, pcts)])
        print(f"             {pct_str}")

# 2. Cohen's d
print("\n" + "=" * 100)
print("2. COHEN'S D EFFECT SIZE (Stressed vs Healthy)")
print("=" * 100)
print(f"  {'Feature':25s} {'Cohen d':>10s} {'Interpretation':>20s}")
print("  " + "-" * 60)

for fn in FEATURE_NAMES:
    s_vals = stressed[fn].values
    h_vals = healthy[fn].values
    n1, n2 = len(s_vals), len(h_vals)
    pooled_std = np.sqrt(((n1-1)*np.std(s_vals, ddof=1)**2 + (n2-1)*np.std(h_vals, ddof=1)**2) / (n1+n2-2))
    d = (np.mean(s_vals) - np.mean(h_vals)) / pooled_std if pooled_std > 0 else 0.0
    interp = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "LARGE"
    print(f"  {fn:25s} {d:+10.3f} {interp:>20s}")

# 3. OOD analysis for Case A
print("\n" + "=" * 100)
print("3. OOD ANALYSIS: Case A vs Training Distribution")
print("=" * 100)

case_a = {
    'avg_income': 1000, 'avg_expenses': 2500, 'final_savings': 0,
    'debt_payment': 100, 'credit_score': 650, 'debt_ratio': 0.0,
    'expense_volatility': 0.0, 'net_cash_flow': -1600, 'savings_trend': 0
}

print(f"  {'Feature':25s} {'Case A':>10s} {'Train Min':>10s} {'Train Max':>10s} {'Percentile':>12s} {'Status':>10s}")
print("  " + "-" * 80)
for fn in FEATURE_NAMES:
    val = case_a[fn]
    train_vals = feat_df[fn].values
    pctl = scipy_stats.percentileofscore(train_vals, val)
    tmin = np.min(train_vals)
    tmax = np.max(train_vals)
    if val < tmin or val > tmax:
        status = "** OOD **"
    elif pctl < 1 or pctl > 99:
        status = "EXTREME"
    elif pctl < 5 or pctl > 95:
        status = "edge"
    else:
        status = "in-dist"
    print(f"  {fn:25s} {val:10.2f} {tmin:10.2f} {tmax:10.2f} {pctl:10.1f}% {status:>10s}")

# 4. expense_volatility deep dive
print("\n" + "=" * 100)
print("4. EXPENSE VOLATILITY DEEP DIVE")
print("=" * 100)

ev = feat_df['expense_volatility'].values
print(f"  Min expense_volatility in training:  {np.min(ev):.6f}")
print(f"  Max expense_volatility in training:  {np.max(ev):.6f}")
print(f"  Mean:                                {np.mean(ev):.6f}")
print(f"  Std:                                 {np.std(ev):.6f}")
print(f"  Profiles with EV < 0.05:             {np.sum(ev < 0.05)}")
print(f"  Profiles with EV < 0.08:             {np.sum(ev < 0.08)}")
print(f"  Profiles with EV < 0.10:             {np.sum(ev < 0.10)}")
print(f"  p1 = {np.percentile(ev, 1):.6f}")
print(f"  p5 = {np.percentile(ev, 5):.6f}")
print(f"  p10 = {np.percentile(ev, 10):.6f}")
print(f"\n  API sends EV=0.0, but training min is {np.min(ev):.6f}")
z_ev = (0 - np.mean(ev)) / np.std(ev)
print(f"  Z-score of EV=0: (0 - {np.mean(ev):.4f}) / {np.std(ev):.4f} = {z_ev:.2f}")
print(f"  The model has NEVER seen expense_volatility near 0!")

# 5. net_cash_flow analysis
print("\n" + "=" * 100)
print("5. NET CASH FLOW CLASS-CONDITIONAL ANALYSIS")
print("=" * 100)

ncf = feat_df['net_cash_flow'].values
print(f"  Overall: Min={np.min(ncf):.2f}, Max={np.max(ncf):.2f}, Mean={np.mean(ncf):.2f}")

for threshold in [0, -500, -1000, -1600]:
    subset = feat_df[feat_df['net_cash_flow'] < threshold]
    if len(subset) > 0:
        print(f"  NCF < {threshold:6d}: {len(subset):4d} profiles ({len(subset)/len(feat_df):.1%}), stressed: {subset['is_stressed'].mean():.1%}")
    else:
        print(f"  NCF < {threshold:6d}:    0 profiles ** NO TRAINING DATA **")

# 6. debt_ratio masking
print("\n" + "=" * 100)
print("6. DEBT_RATIO MASKING ANALYSIS")
print("=" * 100)

dr = feat_df['debt_ratio'].values
neg_mask = feat_df['net_cash_flow'] < 0
pos_mask = feat_df['net_cash_flow'] > 0

print(f"  NCF < 0 AND debt_ratio == 0: {np.sum(neg_mask & (feat_df['debt_ratio'] == 0.0))}")
print(f"  NCF > 0 AND debt_ratio == 0: {np.sum(pos_mask & (feat_df['debt_ratio'] == 0.0))}")
if neg_mask.sum() > 0:
    print(f"  NCF < 0: mean debt_ratio = {feat_df.loc[neg_mask, 'debt_ratio'].mean():.4f}")
if pos_mask.sum() > 0:
    print(f"  NCF > 0: mean debt_ratio = {feat_df.loc[pos_mask, 'debt_ratio'].mean():.4f}")
print(f"  Overall: Mean={np.mean(dr):.4f}, Std={np.std(dr):.4f}, Min={np.min(dr):.4f}, Max={np.max(dr):.4f}")
print(f"  p50={np.median(dr):.4f}, p95={np.percentile(dr, 95):.4f}, p99={np.percentile(dr, 99):.4f}")
print(f"  Profiles with DR > 10: {np.sum(dr > 10)} ({np.sum(dr > 10)/len(dr):.1%})")
print(f"  Profiles with DR > 100: {np.sum(dr > 100)} ({np.sum(dr > 100)/len(dr):.1%})")

# 7. Income range
print("\n" + "=" * 100)
print("7. INCOME RANGE CHECK")
print("=" * 100)
inc = feat_df['avg_income'].values
print(f"  Min income: ${np.min(inc):.0f}/mo (=${np.min(inc)*12:.0f}/yr)")
print(f"  Profiles < $1500/mo: {np.sum(inc < 1500)}")
print(f"  Profiles < $2000/mo: {np.sum(inc < 2000)}")
print(f"  Profiles < $2500/mo: {np.sum(inc < 2500)}")
print(f"  Struggling archetype: $28k-$58k/yr = $2333-$4833/mo")
print(f"  Case A income $1000/mo = $12k/yr -- BELOW all archetypes!")

print("\n" + "=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
