"""
Diagnostic script for sigmoid-saturation bug.

Tests H1: Training data too separable (model outputs extreme probabilities).
Tests H2: Feature scale mismatch between 12-month training and 6-month inference.

Run from project root:
    python .tmp/debug_probs.py
"""

import json
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add project root to path so backend imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import settings
from backend.data.feature_engineering import build_feature_matrix, FEATURE_NAMES, engineer_features
from backend.ml.model import FinancialRiskModel


def load_model_and_scaler():
    """Load the trained model and scaler stats."""
    model = FinancialRiskModel(input_size=settings.INPUT_SIZE)
    state_dict = torch.load(settings.MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with open(settings.SCALER_PATH) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    scale = np.array(stats["scale"], dtype=np.float32)

    return model, mean, scale


def run_inference(model, X_scaled):
    """Run model inference on a batch of scaled features."""
    with torch.no_grad():
        tensor = torch.tensor(X_scaled, dtype=torch.float32)
        probs = model(tensor).squeeze(-1).numpy()
    return probs


def bucket_count(probs, label, bins):
    """Print probability distribution in buckets."""
    print(f"\n  {label} bucket distribution:")
    for lo, hi in bins:
        count = np.sum((probs >= lo) & (probs < hi))
        pct = count / len(probs) * 100
        bar = "#" * int(pct / 2)
        print(f"    [{lo:.2f}, {hi:.2f}): {count:5d} ({pct:5.1f}%) {bar}")


def h1_test():
    """H1: Test if training data creates too-separable clusters."""
    print("=" * 60)
    print("H1: Probability distribution on the TEST SET")
    print("=" * 60)

    # Reproduce training split exactly as train.py does
    df = pd.read_csv(settings.DATA_PATH)
    X, y, profile_ids = build_feature_matrix(df)
    indices = np.arange(len(y))

    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=42,
    )
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=settings.VAL_SIZE,
        stratify=y[idx_train_val],
        random_state=42,
    )

    X_train = X[idx_train]
    X_test = X[idx_test]
    y_test = y[idx_test]

    # Fit scaler on train only (same as training)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # Load model and run inference
    model, mean, scale = load_model_and_scaler()
    probs = run_inference(model, X_test_scaled)

    print(f"\n  Test set size: {len(probs)} profiles")
    print(f"  Label distribution: {y_test.sum():.0f} stressed, {(1-y_test).sum():.0f} healthy")

    print(f"\n  Probability statistics:")
    print(f"    min:    {probs.min():.6f}")
    print(f"    max:    {probs.max():.6f}")
    print(f"    mean:   {probs.mean():.6f}")
    print(f"    std:    {probs.std():.6f}")
    print(f"    p5:     {np.percentile(probs, 5):.6f}")
    print(f"    p25:    {np.percentile(probs, 25):.6f}")
    print(f"    p50:    {np.percentile(probs, 50):.6f}")
    print(f"    p75:    {np.percentile(probs, 75):.6f}")
    print(f"    p95:    {np.percentile(probs, 95):.6f}")

    bins = [
        (0.00, 0.05),
        (0.05, 0.35),
        (0.35, 0.65),
        (0.65, 0.95),
        (0.95, 1.01),
    ]
    bucket_count(probs, "Test set", bins)

    # Inspect borderline samples: sort by predicted probability and show transition zone
    sorted_idx = np.argsort(probs)
    print(f"\n  Borderline samples (near 0.5, sorted by predicted probability):")
    mid = len(sorted_idx) // 2
    window = sorted_idx[max(0, mid-10):mid+10]
    print(f"  {'idx':>6}  {'true_label':>10}  {'pred_prob':>10}  {'pred_category':>14}")
    for i in window:
        cat = "HIGH" if probs[i] >= 0.65 else "MED" if probs[i] >= 0.35 else "LOW"
        print(f"  {i:6d}  {int(y_test[i]):10d}  {probs[i]:10.6f}  {cat:>14}")

    print(f"\n  VERDICT:")
    extreme = np.sum((probs < 0.05) | (probs > 0.95))
    middle = np.sum((probs >= 0.35) & (probs <= 0.65))
    print(f"    Extreme (< 0.05 or > 0.95): {extreme}/{len(probs)} = {extreme/len(probs)*100:.1f}%")
    print(f"    Middle zone [0.35, 0.65]:    {middle}/{len(probs)} = {middle/len(probs)*100:.1f}%")

    if extreme / len(probs) > 0.90:
        print("    -> H1 CONFIRMED: Data is over-separable, model saturates on test set too")
    else:
        print("    -> H1 NOT CONFIRMED: Test set has intermediate values")

    return probs, y_test, X_test, X_train, scaler


def h2_test(X_train_raw, scaler_trained):
    """H2: Test feature scale mismatch between 12-month training and 6-month inference."""
    print("\n" + "=" * 60)
    print("H2: Feature scale mismatch (12-month training vs 6-month inference)")
    print("=" * 60)

    # Grab a "medium" profile from the training data — use a getting_by archetype
    # by picking a profile with moderate income (~$4000/month)
    df = pd.read_csv(settings.DATA_PATH)

    # Find a profile that we'd consider "borderline" — moderate income, moderate expenses
    profile_stats = df.groupby('profile_id').agg({
        'income': 'mean',
        'total_expenses': 'mean',
        'is_stressed': 'first'
    }).reset_index()

    # Find a profile near median income that isn't stressed
    median_income = profile_stats['income'].median()
    candidates = profile_stats[
        (profile_stats['income'].between(median_income * 0.9, median_income * 1.1)) &
        (profile_stats['is_stressed'] == 0)
    ]

    if len(candidates) == 0:
        print("  Could not find borderline unstressed candidate profile")
        return

    sample_profile_id = candidates.iloc[0]['profile_id']
    sample_df = df[df['profile_id'] == sample_profile_id].sort_values('month')

    print(f"\n  Using profile_id={sample_profile_id} (median income ~${median_income:.0f}/mo, unstressed)")

    # Compute features using ALL 12 months
    features_12 = engineer_features(sample_df)
    vec_12 = np.array([features_12[n] for n in FEATURE_NAMES], dtype=np.float32)
    scaled_12 = scaler_trained.transform(vec_12.reshape(1, -1)).flatten()

    # Compute features using only FIRST 6 months
    sample_df_6 = sample_df.iloc[:6].copy()
    features_6 = engineer_features(sample_df_6)
    vec_6 = np.array([features_6[n] for n in FEATURE_NAMES], dtype=np.float32)
    scaled_6 = scaler_trained.transform(vec_6.reshape(1, -1)).flatten()

    print(f"\n  {'Feature':<25}  {'12-month raw':>13}  {'6-month raw':>12}  {'12-mo z':>8}  {'6-mo z':>8}  {'Out-of-dist?':>12}")
    print("  " + "-" * 85)
    for i, name in enumerate(FEATURE_NAMES):
        ood = "YES ***" if abs(scaled_6[i]) > 3.0 else ""
        print(f"  {name:<25}  {vec_12[i]:13.2f}  {vec_6[i]:12.2f}  {scaled_12[i]:8.3f}  {scaled_6[i]:8.3f}  {ood:>12}")

    # Run inference on both
    model, _, _ = load_model_and_scaler()
    prob_12 = run_inference(model, scaled_12.reshape(1, -1).astype(np.float32))[0]
    prob_6 = run_inference(model, scaled_6.reshape(1, -1).astype(np.float32))[0]

    print(f"\n  Model prediction (12-month input): {prob_12:.6f} -> {prob_12*100:.1f}%")
    print(f"  Model prediction (6-month input):  {prob_6:.6f} -> {prob_6*100:.1f}%")

    ood_count = np.sum(np.abs(scaled_6) > 3.0)
    print(f"\n  Out-of-distribution features (|z| > 3): {ood_count}/{len(FEATURE_NAMES)}")

    if ood_count > 0 and abs(prob_6 - prob_12) > 0.2:
        print("  -> H2 CONFIRMED: 6-month inputs shift features out of training distribution")
    elif ood_count > 0:
        print("  -> H2 PARTIAL: Some OOD features but prediction similar")
    else:
        print("  -> H2 NOT CONFIRMED: 6-month features remain within training distribution")

    # Also check what happens with simulated "all zeros" debt_payment (a common user input)
    print("\n  --- Bonus: What if user enters 0 for debt_payment? ---")
    vec_zero_debt = vec_12.copy()
    vec_zero_debt[FEATURE_NAMES.index('debt_payment')] = 0.0
    vec_zero_debt[FEATURE_NAMES.index('debt_ratio')] = 0.0  # debt_ratio also changes
    scaled_zero_debt = scaler_trained.transform(vec_zero_debt.reshape(1, -1)).flatten()
    prob_zero_debt = run_inference(model, scaled_zero_debt.reshape(1, -1).astype(np.float32))[0]
    print(f"  debt_payment z-score when 0: {scaled_zero_debt[FEATURE_NAMES.index('debt_payment')]:.3f}")
    print(f"  Model prediction with 0 debt: {prob_zero_debt:.6f} -> {prob_zero_debt*100:.1f}%")


def h3_test():
    """H3: Check file timestamps for model/scaler staleness."""
    print("\n" + "=" * 60)
    print("H3: Model and scaler artifact timestamp check")
    print("=" * 60)
    import os
    model_mtime = os.path.getmtime(settings.MODEL_PATH)
    scaler_mtime = os.path.getmtime(settings.SCALER_PATH)
    from datetime import datetime
    print(f"  Model  (latest_model.pth):  {datetime.fromtimestamp(model_mtime)}")
    print(f"  Scaler (scaler_stats.json): {datetime.fromtimestamp(scaler_mtime)}")
    if abs(model_mtime - scaler_mtime) > 60:
        print("  -> H3 WARNING: Files have different timestamps (> 1 min apart)")
    else:
        print("  -> H3 CLEAR: Files were written at the same time")


def check_archetype_separability():
    """Check how separable the archetypes actually are in feature space."""
    print("\n" + "=" * 60)
    print("ARCHETYPE SEPARABILITY CHECK")
    print("=" * 60)

    df = pd.read_csv(settings.DATA_PATH)
    X, y, _ = build_feature_matrix(df)

    # Check overlap in key features between stressed and unstressed
    print(f"\n  Stressed profiles: {y.sum():.0f} ({y.mean():.1%})")
    print(f"  Unstressed profiles: {(1-y).sum():.0f} ({(1-y.mean()):.1%})")

    print(f"\n  Feature ranges by class:")
    print(f"  {'Feature':<25}  {'Stressed min':>12}  {'Stressed max':>12}  {'Healthy min':>11}  {'Healthy max':>11}  {'Overlap?':>9}")
    print("  " + "-" * 90)

    for i, name in enumerate(FEATURE_NAMES):
        stressed_vals = X[y == 1, i]
        healthy_vals = X[y == 0, i]
        s_min, s_max = stressed_vals.min(), stressed_vals.max()
        h_min, h_max = healthy_vals.min(), healthy_vals.max()
        overlap = "YES" if s_min < h_max and h_min < s_max else "NO"
        print(f"  {name:<25}  {s_min:12.1f}  {s_max:12.1f}  {h_min:11.1f}  {h_max:11.1f}  {overlap:>9}")

    # Check consec_negative_months — this is directly in the label definition
    print(f"\n  consec_negative_months distribution (stressed vs healthy):")
    for val in range(0, 13):
        s_count = np.sum((X[y == 1, FEATURE_NAMES.index('consec_negative_months')] == val))
        h_count = np.sum((X[y == 0, FEATURE_NAMES.index('consec_negative_months')] == val))
        if s_count > 0 or h_count > 0:
            print(f"    {val:2d} months: stressed={s_count:5d}  healthy={h_count:5d}")


if __name__ == "__main__":
    probs, y_test, X_test, X_train_raw, scaler = h1_test()
    check_archetype_separability()
    h2_test(X_train_raw, scaler)
    h3_test()
