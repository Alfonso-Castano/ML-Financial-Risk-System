"""
Feature engineering for ML financial risk assessment.

Transforms long-format financial data (12 monthly rows per profile) into
a single 9-feature vector per profile suitable for MLP model consumption.

Feature set (in order):
    avg_income            - Mean monthly income over 12 months
    avg_expenses          - Mean total monthly expenses over 12 months
    final_savings         - Savings balance at month 12
    expense_ratio         - Average expenses divided by average income
    credit_score          - Credit score (constant per profile)
    savings_months        - Final savings divided by average expenses (buffer in months)
    expense_volatility    - Coefficient of variation of monthly expenses (std/mean)
    net_cash_flow         - Income minus expenses (debt included in expenses)
    savings_trend         - Linear slope of cumulative savings ($/month)
"""

import numpy as np
import pandas as pd
from typing import Dict, List


# Canonical feature order — must match model input expectations
FEATURE_NAMES: List[str] = [
    'avg_income',
    'avg_expenses',
    'final_savings',
    'expense_ratio',
    'credit_score',
    'savings_months',
    'expense_volatility',
    'net_cash_flow',
    'savings_trend',
]


def compute_expense_ratio(
    avg_monthly_income: float,
    avg_monthly_expenses: float,
) -> float:
    """
    Compute the ratio of average expenses to average income.

    Formula: avg_monthly_expenses / avg_monthly_income

    Values near or above 1.0 indicate spending at or beyond income level.
    Returns 0.0 when income is zero or negative to avoid division errors.

    Args:
        avg_monthly_income: Mean monthly income over the observation period.
        avg_monthly_expenses: Mean total monthly expenses over the observation period.

    Returns:
        Expense ratio as a float, or 0.0 if income is non-positive.
    """
    if avg_monthly_income <= 0:
        return 0.0
    return avg_monthly_expenses / avg_monthly_income


def compute_savings_months(
    final_savings: float,
    avg_monthly_expenses: float,
) -> float:
    """
    Compute savings buffer as months of expenses covered.

    Formula: final_savings / avg_monthly_expenses

    Directly interpretable: 3.0 means savings cover 3 months of expenses.
    Returns 0.0 when expenses are zero or negative.

    Args:
        final_savings: Savings balance at the end of the observation window.
        avg_monthly_expenses: Mean total monthly expenses over the observation period.

    Returns:
        Savings buffer in months, or 0.0 if expenses are non-positive.
    """
    if avg_monthly_expenses <= 0:
        return 0.0
    return final_savings / avg_monthly_expenses


def compute_expense_volatility(monthly_expenses: List[float]) -> float:
    """
    Measures expense instability as standard deviation relative to the mean.
    Higher values indicate erratic spending patterns, a risk signal that correlates with
    but does NOT directly encode the label conditions.

    Formula: std(monthly_expenses) / mean(monthly_expenses) — coefficient of variation.
    Returns 0.0 if mean is zero or fewer than 2 data points.

    Args:
        monthly_expenses: List of monthly total expense values in chronological order.

    Returns:
        Coefficient of variation as a float, or 0.0 for degenerate inputs.
    """
    if len(monthly_expenses) < 2:
        return 0.0
    mean_expenses = float(np.mean(monthly_expenses))
    if mean_expenses == 0.0:
        return 0.0
    return float(np.std(monthly_expenses) / mean_expenses)


def compute_savings_trend(cumulative_savings: List[float]) -> float:
    """
    Captures savings trajectory as the linear slope of cumulative savings
    over the observation window. Negative slope signals deteriorating savings
    without directly implementing any label threshold.

    Formula: linear regression slope via np.polyfit — average monthly change
    in savings (dollars per month). Positive = growing savings, negative = declining.
    Returns 0.0 if fewer than 2 data points.

    Args:
        cumulative_savings: List of cumulative savings values in chronological order.

    Returns:
        Linear slope in dollars per month, or 0.0 for degenerate inputs.
    """
    if len(cumulative_savings) < 2:
        return 0.0
    return float(np.polyfit(range(len(cumulative_savings)), cumulative_savings, 1)[0])


def compute_net_cash_flow(
    avg_monthly_income: float,
    avg_monthly_expenses: float,
) -> float:
    """
    Compute net monthly cash flow after expenses.

    Formula: avg_monthly_income - avg_monthly_expenses

    Debt payments are already included in total_expenses (the synthetic generator
    adds debt_payment into total_expenses at generation time), so subtracting
    debt separately would double-count it.

    Args:
        avg_monthly_income: Mean monthly income over the observation period.
        avg_monthly_expenses: Mean total monthly expenses over the observation period.

    Returns:
        Net cash flow as a float. May be negative.
    """
    return avg_monthly_income - avg_monthly_expenses


def compute_cash_flow_volatility(monthly_cash_flows: List[float]) -> float:
    """
    Compute the standard deviation of monthly cash flows.

    Higher volatility indicates less predictable financial behavior.
    NOTE: This function is kept for completeness but is NOT included in the
    final 9-feature set. net_cash_flow was chosen instead per locked decision.

    Args:
        monthly_cash_flows: List of monthly (income - expenses) values.

    Returns:
        Standard deviation as a float, or 0.0 if fewer than 2 data points.
    """
    if len(monthly_cash_flows) < 2:
        return 0.0
    return float(np.std(monthly_cash_flows))


def engineer_features(profile_df: pd.DataFrame) -> Dict[str, float]:
    """
    Transform one profile's 12 monthly rows into a 9-feature dictionary.

    Sorts by month to guarantee correct chronological ordering before
    computing any sequential features (e.g., final_savings from month 12).

    Args:
        profile_df: DataFrame slice containing exactly the rows for one profile_id.
                    Expected columns: month, income, total_expenses, savings,
                    credit_score.

    Returns:
        Dict with exactly 9 keys matching FEATURE_NAMES, all values as float.
    """
    group = profile_df.sort_values('month')

    # Raw aggregates
    avg_income = float(group['income'].mean())
    avg_expenses = float(group['total_expenses'].mean())
    final_savings = float(group.iloc[-1]['savings'])   # month 12 after sorting
    credit_score = float(group.iloc[0]['credit_score'])  # fixed per profile

    # Per-month series for volatility and trend features
    monthly_expenses = group['total_expenses'].tolist()
    savings_series = group['savings'].tolist()

    # Derived features
    expense_ratio = compute_expense_ratio(avg_income, avg_expenses)
    savings_months = compute_savings_months(final_savings, avg_expenses)
    expense_volatility = compute_expense_volatility(monthly_expenses)
    net_cash_flow = compute_net_cash_flow(avg_income, avg_expenses)
    savings_trend = compute_savings_trend(savings_series)

    return {
        'avg_income': avg_income,
        'avg_expenses': avg_expenses,
        'final_savings': final_savings,
        'expense_ratio': expense_ratio,
        'credit_score': credit_score,
        'savings_months': savings_months,
        'expense_volatility': expense_volatility,
        'net_cash_flow': net_cash_flow,
        'savings_trend': savings_trend,
    }


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple:
    """
    Build ML-ready feature matrix from the full long-format CSV.

    Groups the dataframe by profile_id, engineers features for each profile,
    and assembles the results into NumPy arrays ready for model training or
    inference.

    Args:
        df: Full long-format DataFrame with one row per (profile_id, month).
            Expected columns: profile_id, is_stressed, and all columns
            required by engineer_features().

    Returns:
        Tuple of (X, y, profile_ids) where:
            X           - np.ndarray of shape (n_profiles, 9), dtype float32
            y           - np.ndarray of shape (n_profiles,), dtype float32
            profile_ids - list of ints, one per profile in row order
    """
    feature_rows: List[List[float]] = []
    labels: List[float] = []
    profile_ids: List[int] = []

    for profile_id, group in df.groupby('profile_id'):
        features = engineer_features(group)
        # Preserve canonical FEATURE_NAMES order when building the row vector
        feature_rows.append([features[name] for name in FEATURE_NAMES])
        labels.append(float(group.iloc[0]['is_stressed']))
        profile_ids.append(int(profile_id))

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)

    return X, y, profile_ids
