"""
Synthetic financial data generator.

Generates realistic financial profiles with stress labeling for ML training.
Each profile contains 12 months of financial history with income, expenses,
savings, debt, and credit score data.

Stress is labeled based on two conditions:
1. Final month savings < final month total expenses
2. 3+ consecutive months with negative cash flow
"""

import numpy as np
import pandas as pd
from pathlib import Path
from backend.config.settings import (
    NUM_PROFILES,
    RANDOM_SEED,
    MONTHS_HISTORY,
    MONTHLY_VARIANCE,
    STRESS_SAVINGS_THRESHOLD,
    STRESS_NEGATIVE_STREAK,
    ARCHETYPES,
    OUTPUT_PATH
)


def generate_profiles(num_profiles: int, seed: int) -> pd.DataFrame:
    """
    Generate synthetic financial profiles.

    Args:
        num_profiles: Number of profiles to generate
        seed: Random seed for reproducibility

    Returns:
        DataFrame with all profiles in long format (one row per person-month)
    """
    rng = np.random.default_rng(seed)

    # Allocate profiles across archetypes based on weights
    weights = [ARCHETYPES[arch]["weight"] for arch in ARCHETYPES.keys()]
    archetype_names = list(ARCHETYPES.keys())
    profile_archetypes = rng.choice(
        archetype_names,
        size=num_profiles,
        p=weights
    )

    # Generate all profiles
    all_rows = []
    for profile_id in range(1, num_profiles + 1):
        archetype_name = profile_archetypes[profile_id - 1]
        archetype_config = ARCHETYPES[archetype_name]

        profile_rows = _generate_single_profile(
            rng, archetype_config, profile_id
        )
        all_rows.extend(profile_rows)

    # Create DataFrame
    df = pd.DataFrame(all_rows)

    # Apply stress labels
    df = _apply_stress_labels(df)

    return df


def _generate_single_profile(rng, archetype: dict, profile_id: int) -> list:
    """
    Generate one profile's 12 months of financial history.

    Args:
        rng: Numpy random generator
        archetype: Archetype configuration dictionary
        profile_id: Unique profile identifier

    Returns:
        List of dictionaries, one per month
    """
    # Generate annual income
    annual_income = rng.uniform(
        archetype["income_range"][0],
        archetype["income_range"][1]
    )
    monthly_income = annual_income / 12

    # Generate credit score
    credit_score = int(
        np.clip(
            rng.normal(archetype["credit_score_mean"], 40),
            300,
            850
        )
    )

    # Generate base expense ratio
    base_expense_ratio = rng.uniform(
        archetype["expense_ratio"][0],
        archetype["expense_ratio"][1]
    )

    # Calculate base monthly expenses
    base_monthly_expenses = monthly_income * base_expense_ratio

    # Split expenses into categories
    # Essentials: ~60% of total expenses
    # Discretionary: ~20% of total expenses
    # Debt payment: ~20% of total expenses
    base_essentials = base_monthly_expenses * 0.60
    base_discretionary = base_monthly_expenses * 0.20
    debt_payment = base_monthly_expenses * 0.20  # Fixed per profile

    # Generate initial savings
    initial_savings = base_monthly_expenses * rng.uniform(
        archetype["savings_months"][0],
        archetype["savings_months"][1]
    )

    # Generate 12 months of history
    rows = []
    current_savings = initial_savings

    for month in range(1, MONTHS_HISTORY + 1):
        # Apply monthly variance to income
        monthly_income_actual = monthly_income * rng.normal(1.0, MONTHLY_VARIANCE)

        # Apply monthly variance to expenses
        essentials = base_essentials * rng.normal(1.0, MONTHLY_VARIANCE)
        discretionary = base_discretionary * rng.normal(1.0, MONTHLY_VARIANCE)
        # debt_payment stays fixed

        # Calculate totals
        total_expenses = essentials + discretionary + debt_payment
        cash_flow = monthly_income_actual - total_expenses

        # Update savings (cannot go negative)
        current_savings = max(0, current_savings + cash_flow)

        # Record row
        row = {
            "profile_id": profile_id,
            "month": month,
            "income": round(monthly_income_actual, 2),
            "essentials": round(essentials, 2),
            "discretionary": round(discretionary, 2),
            "debt_payment": round(debt_payment, 2),
            "total_expenses": round(total_expenses, 2),
            "savings": round(current_savings, 2),
            "debt": round(debt_payment * 12, 2),  # Annualized
            "credit_score": credit_score
        }
        rows.append(row)

    return rows


def _apply_stress_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply stress labels to profiles based on two conditions.

    Stress conditions (either triggers stress=1):
    1. Final month savings < final month total expenses
    2. 3+ consecutive months with negative cash flow

    Args:
        df: DataFrame with all profiles

    Returns:
        DataFrame with is_stressed column added
    """
    stress_labels = {}

    for profile_id, group in df.groupby("profile_id"):
        # Sort by month to ensure correct ordering
        group = group.sort_values("month")

        # Condition 1: Final month savings < final month total expenses
        final_row = group.iloc[-1]
        condition1 = final_row["savings"] < (final_row["total_expenses"] * STRESS_SAVINGS_THRESHOLD)

        # Condition 2: 3+ consecutive months with negative cash flow
        cash_flows = group["income"] - group["total_expenses"]

        # Find max consecutive negative cash flow months
        max_streak = 0
        current_streak = 0
        for cf in cash_flows:
            if cf < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        condition2 = max_streak >= STRESS_NEGATIVE_STREAK

        # Profile is stressed if either condition is true
        is_stressed = 1 if (condition1 or condition2) else 0
        stress_labels[profile_id] = is_stressed

    # Apply labels to all rows of each profile
    df["is_stressed"] = df["profile_id"].map(stress_labels)

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """
    Save dataset to CSV and print summary.

    Args:
        df: DataFrame to save
        output_path: Path to save CSV file
    """
    # Create parent directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)

    # Print summary
    num_profiles = df["profile_id"].nunique()
    stress_ratio = df.groupby("profile_id")["is_stressed"].first().mean()

    print(f"\nDataset saved to: {output_path}")
    print(f"Total rows: {len(df):,}")
    print(f"Unique profiles: {num_profiles:,}")
    print(f"Stress ratio: {stress_ratio:.2%}")
    print(f"Columns: {list(df.columns)}")

    # Show value ranges
    print(f"\nValue ranges:")
    print(f"  Income: ${df['income'].min():.0f} - ${df['income'].max():.0f}")
    print(f"  Expenses: ${df['total_expenses'].min():.0f} - ${df['total_expenses'].max():.0f}")
    print(f"  Savings: ${df['savings'].min():.0f} - ${df['savings'].max():.0f}")
    print(f"  Credit score: {df['credit_score'].min()} - {df['credit_score'].max()}")


if __name__ == "__main__":
    print("Generating synthetic financial data...")
    print(f"Number of profiles: {NUM_PROFILES:,}")
    print(f"Months of history: {MONTHS_HISTORY}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Archetypes: {', '.join(ARCHETYPES.keys())}")

    # Generate profiles
    df = generate_profiles(NUM_PROFILES, RANDOM_SEED)

    # Save dataset
    save_dataset(df, OUTPUT_PATH)

    # Print class balance details
    print(f"\nClass balance by archetype:")
    # Need to determine archetype per profile based on income
    # This is approximate since we don't store archetype in the data
    profile_stats = df.groupby("profile_id").agg({
        "income": "mean",
        "is_stressed": "first"
    }).reset_index()

    print(f"  Stressed profiles: {profile_stats['is_stressed'].sum():,} ({profile_stats['is_stressed'].mean():.1%})")
    print(f"  Healthy profiles: {(1 - profile_stats['is_stressed']).sum():,} ({(1 - profile_stats['is_stressed'].mean()):.1%})")

    print("\nGeneration complete!")
