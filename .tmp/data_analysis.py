"""Analyze training data distribution gap for Case D fix."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/synthetic_train.csv")

# Per-profile aggregation
profiles = df.groupby("profile_id").agg(
    avg_income=("income", "mean"),
    avg_expenses=("total_expenses", "mean"),
    is_stressed=("is_stressed", "first"),
).reset_index()
profiles["expense_ratio"] = profiles["avg_expenses"] / profiles["avg_income"]

print(f"Total profiles: {len(profiles)}")
print(f"Overall stress rate: {profiles['is_stressed'].mean():.1%}\n")

# Primary question: income > $8k/month AND expense_ratio > 0.85
mask = (profiles["avg_income"] > 8000) & (profiles["expense_ratio"] > 0.85)
subset = profiles[mask]
print(f"Profiles with income > $8k AND expense_ratio > 0.85: {len(subset)} ({len(subset)/len(profiles):.1%})")
if len(subset) > 0:
    print(f"  Stress rate: {subset['is_stressed'].mean():.1%}")
else:
    print("  NO PROFILES FOUND — confirms OOD hypothesis")

# Case D exact neighborhood
mask2 = (profiles["avg_income"].between(8000, 12000)) & (profiles["expense_ratio"].between(0.90, 1.10))
subset2 = profiles[mask2]
print(f"\nCase D neighborhood (income $8k-$12k, ratio 0.90-1.10): {len(subset2)} profiles")

# Income distribution for high earners
high_income = profiles[profiles["avg_income"] > 5000]
print(f"\nHigh income (>$5k/month): {len(high_income)} profiles")
print(f"  Max expense_ratio: {high_income['expense_ratio'].max():.3f}")
print(f"  Mean expense_ratio: {high_income['expense_ratio'].mean():.3f}")
print(f"  Stress rate: {high_income['is_stressed'].mean():.1%}")

# Crosstab: income bands x expense_ratio bands
profiles["income_band"] = pd.cut(profiles["avg_income"],
    bins=[0, 2000, 4000, 6000, 8000, 12000],
    labels=["<2k", "2-4k", "4-6k", "6-8k", "8k+"])
profiles["ratio_band"] = pd.cut(profiles["expense_ratio"],
    bins=[0, 0.5, 0.7, 0.85, 1.0, 2.0],
    labels=["<0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00", "1.00+"])

print("\n=== Count crosstab (income band x expense_ratio band) ===")
ct = pd.crosstab(profiles["income_band"], profiles["ratio_band"], margins=True)
print(ct)

print("\n=== Stress rate crosstab ===")
sr = profiles.groupby(["income_band", "ratio_band"])["is_stressed"].mean().unstack(fill_value=0)
print(sr.round(2))
