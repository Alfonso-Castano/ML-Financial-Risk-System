"""
Configuration settings for synthetic data generation.

This module contains all constants and configuration parameters
used by the synthetic financial data generator.
"""

# Dataset configuration
NUM_PROFILES = 3000
RANDOM_SEED = 42
MONTHS_HISTORY = 12

# Variance and fluctuation
MONTHLY_VARIANCE = 0.15  # 15% monthly fluctuation in income and expenses

# Stress labeling thresholds
STRESS_SAVINGS_THRESHOLD = 1.0  # Savings must be >= 1 month expenses
STRESS_NEGATIVE_STREAK = 3      # Consecutive negative cash flow months

# Target class balance (approximate, not enforced)
TARGET_STRESS_RATIO = 0.35

# Credit score range
CREDIT_SCORE_RANGE = (300, 850)

# Financial archetypes
# Each archetype represents a recognizable financial profile type
# Adjusted to achieve ~35% stress ratio through realistic financial stress patterns
ARCHETYPES = {
    "struggling": {
        "income_range": (30000, 45000),
        "savings_months": (0, 0.2),         # Almost no savings buffer
        "expense_ratio": (0.95, 1.10),      # Frequently spending more than income
        "credit_score_mean": 620,
        "weight": 0.30                      # Increased to create more stressed profiles
    },
    "getting_by": {
        "income_range": (40000, 55000),
        "savings_months": (0.1, 0.8),       # Very minimal savings buffer
        "expense_ratio": (0.85, 0.98),      # High expense ratio with variance
        "credit_score_mean": 670,
        "weight": 0.25
    },
    "stable": {
        "income_range": (55000, 85000),
        "savings_months": (2, 6),
        "expense_ratio": (0.60, 0.75),      # Moderate expenses
        "credit_score_mean": 720,
        "weight": 0.28                      # Decreased to balance
    },
    "comfortable": {
        "income_range": (85000, 130000),
        "savings_months": (4, 12),
        "expense_ratio": (0.45, 0.60),      # Lower expenses
        "credit_score_mean": 760,
        "weight": 0.17                      # Decreased slightly
    }
}

# Output path
OUTPUT_PATH = "data/synthetic_train.csv"
