"""
Deep-dive: Why is the data over-separable?

The archetype check revealed consec_negative_months >= 3 perfectly predicts stress.
This script confirms the circular dependency: a feature derived from the SAME
condition used to create the stress label is included in the feature set.

Also checks the other label condition: savings < total_expenses (STRESS_SAVINGS_THRESHOLD=1.0)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.data.feature_engineering import build_feature_matrix, FEATURE_NAMES, engineer_features


def main():
    df = pd.read_csv(settings.DATA_PATH)
    X, y, profile_ids = build_feature_matrix(df)

    print("=" * 65)
    print("ROOT CAUSE ANALYSIS: Feature-Label Circular Dependency")
    print("=" * 65)

    # --- Feature 1: consec_negative_months ---
    # Label condition 2: max_streak >= STRESS_NEGATIVE_STREAK (3)
    consec_idx = FEATURE_NAMES.index('consec_negative_months')
    consec_vals = X[:, consec_idx]

    print("\n[Feature: consec_negative_months]")
    print("Label condition 2: 3+ consecutive negative cash flow months = stressed")
    print(f"\nPerfect threshold test (consec >= 3 -> stressed):")

    stressed_with_consec3 = np.sum((consec_vals >= 3) & (y == 1))
    healthy_with_consec3  = np.sum((consec_vals >= 3) & (y == 0))
    stressed_without      = np.sum((consec_vals < 3)  & (y == 1))
    healthy_without       = np.sum((consec_vals < 3)  & (y == 0))

    print(f"  consec >= 3 AND stressed:  {stressed_with_consec3}")
    print(f"  consec >= 3 AND healthy:   {healthy_with_consec3}  <-- should be 0 if circular")
    print(f"  consec < 3  AND stressed:  {stressed_without}  (these come from condition 1)")
    print(f"  consec < 3  AND healthy:   {healthy_without}")

    # --- Feature 2: liquidity_ratio (relates to label condition 1) ---
    # Label condition 1: final_savings < 1.0 * total_expenses (last month)
    # liquidity_ratio = final_savings / avg_monthly_expenses
    # If condition 1 is true: final_savings < final_total_expenses ~= avg_total_expenses
    # So liquidity_ratio < ~1.0 correlates strongly with stress
    liq_idx = FEATURE_NAMES.index('liquidity_ratio')
    liq_vals = X[:, liq_idx]

    print("\n[Feature: liquidity_ratio]")
    print("Label condition 1: savings < 1 month expenses = stressed")
    print(f"\nThreshold test (liquidity_ratio < 1.0 -> stressed):")

    stressed_liq_lt1 = np.sum((liq_vals < 1.0) & (y == 1))
    healthy_liq_lt1  = np.sum((liq_vals < 1.0) & (y == 0))
    stressed_liq_ge1 = np.sum((liq_vals >= 1.0) & (y == 1))
    healthy_liq_ge1  = np.sum((liq_vals >= 1.0) & (y == 0))

    print(f"  liq < 1.0  AND stressed: {stressed_liq_lt1}")
    print(f"  liq < 1.0  AND healthy:  {healthy_liq_lt1}  <-- should be 0 if circular")
    print(f"  liq >= 1.0 AND stressed: {stressed_liq_ge1}  (these come from condition 2)")
    print(f"  liq >= 1.0 AND healthy:  {healthy_liq_ge1}")

    # --- Combined: can we perfectly separate with just these two features? ---
    print("\n[Combined circular dependency test]")
    print("Prediction: stressed IF (consec >= 3 OR liquidity < 1.0)")

    pred = ((consec_vals >= 3) | (liq_vals < 1.0)).astype(float)
    accuracy = np.mean(pred == y)
    errors = np.sum(pred != y)
    print(f"  Accuracy using only label conditions: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Errors: {errors}")

    if accuracy > 0.99:
        print("\n  *** CIRCULAR DEPENDENCY CONFIRMED ***")
        print("  The training features directly encode the label conditions.")
        print("  The model learns trivial threshold rules, not generalizable patterns.")
        print("  This is why it produces 0% or 100% for all inputs.")

    # --- How many profiles are stressed SOLELY due to condition 2 (consec)? ---
    print("\n[Breakdown: which stress condition fires for each profile]")

    # We need per-profile data for this
    profile_cond1 = {}
    profile_cond2 = {}

    for profile_id, group in df.groupby('profile_id'):
        group = group.sort_values('month')
        final_row = group.iloc[-1]

        # Condition 1
        c1 = final_row['savings'] < (final_row['total_expenses'] * settings.STRESS_SAVINGS_THRESHOLD)

        # Condition 2
        cash_flows = group['income'] - group['total_expenses']
        max_streak = 0
        streak = 0
        for cf in cash_flows:
            if cf < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        c2 = max_streak >= settings.STRESS_NEGATIVE_STREAK

        is_stressed = group.iloc[0]['is_stressed']
        profile_cond1[profile_id] = c1
        profile_cond2[profile_id] = c2

    c1_only  = sum(1 for pid in profile_cond1 if profile_cond1[pid] and not profile_cond2[pid])
    c2_only  = sum(1 for pid in profile_cond2 if profile_cond2[pid] and not profile_cond1[pid])
    both     = sum(1 for pid in profile_cond1 if profile_cond1[pid] and profile_cond2[pid])
    neither  = sum(1 for pid in profile_cond1 if not profile_cond1[pid] and not profile_cond2[pid])

    total_stressed = c1_only + c2_only + both
    print(f"  Condition 1 only (low savings):        {c1_only:5d} profiles")
    print(f"  Condition 2 only (consec neg months):  {c2_only:5d} profiles")
    print(f"  Both conditions:                       {both:5d} profiles")
    print(f"  Neither (healthy):                     {neither:5d} profiles")
    print(f"  Total stressed:                        {total_stressed:5d} profiles")

    # --- Show what the model WOULD need to learn for generalization ---
    print("\n[What genuine generalization would look like]")
    print("  A well-trained model on non-circular data would learn:")
    print("  - TREND in income (declining income = future risk)")
    print("  - VOLATILITY in cash flow (unstable finances = future risk)")
    print("  - RATIO features that predict future stress, not label it retrospectively")
    print()
    print("  Instead, this model learned:")
    print("  - consec_negative_months >= 3 -> 100% stressed (this IS condition 2)")
    print("  - liquidity_ratio < 1.0 -> 100% stressed (this mirrors condition 1)")
    print("  - Everything else -> 0% stressed")
    print()
    print("  The model achieves 0.998 ROC-AUC by memorizing label conditions, not")
    print("  by learning generalizable financial risk patterns.")


if __name__ == "__main__":
    main()
