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
MONTHLY_VARIANCE = 0.25  # 25% monthly fluctuation — increased from 0.15 to create borderline profiles

# Stress labeling thresholds
STRESS_SAVINGS_THRESHOLD = 1.0  # Savings must be >= 1 month expenses
STRESS_NEGATIVE_STREAK = 3      # Consecutive negative cash flow months

# Target class balance (approximate, not enforced)
TARGET_STRESS_RATIO = 0.35

# Credit score range
CREDIT_SCORE_RANGE = (300, 850)

# Financial archetypes
# Each archetype represents a recognizable financial profile type.
# Income ranges and expense ratios intentionally OVERLAP between archetypes so that
# borderline profiles emerge naturally: a "getting_by" person with a bad run of months
# can look like "struggling", and a "struggling" person with a good run can look like
# "getting_by". Combined with higher MONTHLY_VARIANCE this creates genuine uncertainty
# in the training data, forcing the model to learn probability rather than memorize rules.
ARCHETYPES = {
    "poverty": {
        "income_range": (15000, 30000),     # Very low income, overlaps struggling at top
        "savings_months": (0, 1.0),         # Minimal to no savings buffer
        "expense_ratio": (0.90, 1.20),      # Frequently spending more than earning
        "credit_score_mean": 580,
        "weight": 0.10
    },
    "struggling": {
        "income_range": (28000, 58000),     # Wide range overlapping getting_by
        "savings_months": (0, 3.0),         # Range: no savings to 3 months buffer
        "expense_ratio": (0.85, 1.12),      # High ratio overlapping getting_by
        "credit_score_mean": 630,
        "weight": 0.20
    },
    "getting_by": {
        "income_range": (35000, 72000),     # Overlaps both struggling and stable
        "savings_months": (0, 3.5),         # Range: no savings to 3.5 months buffer
        "expense_ratio": (0.72, 1.02),      # Wide range overlapping both neighbors
        "credit_score_mean": 665,
        "weight": 0.19
    },
    "stable": {
        "income_range": (48000, 92000),     # Lower end overlaps getting_by
        "savings_months": (0.1, 8),         # Range: almost no savings to 8 months
        "expense_ratio": (0.55, 0.88),      # Wide range overlapping getting_by at top
        "credit_score_mean": 715,
        "weight": 0.20
    },
    "comfortable": {
        "income_range": (65000, 130000),    # Lower end overlaps stable
        "savings_months": (0.5, 12),        # Range: half month to 12 months savings
        "expense_ratio": (0.40, 0.70),      # Overlaps stable at the top end
        "credit_score_mean": 755,
        "weight": 0.16
    },
    "high_burn": {
        "income_range": (75000, 150000),    # High earners with wide spending range
        "savings_months": (1, 10),          # Wide buffer range — creates genuine ambiguity
        "expense_ratio": (0.70, 1.05),      # Wide range centered on moderate-to-high
        "credit_score_mean": 700,
        "weight": 0.15
    }
}

# Output path
OUTPUT_PATH = "data/synthetic_train.csv"

# Training hyperparameters
NUM_EPOCHS = 75
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 10

# Data split ratios (by profile, not by row)
TEST_SIZE = 0.15
VAL_SIZE = 0.176  # 0.15 / 0.85 to get 15% of total as validation

# Model paths
MODEL_PATH = "models/latest_model.pth"
METRICS_PATH = "models/metrics.json"
SCALER_PATH = "models/scaler_stats.json"
DATA_PATH = "data/synthetic_train.csv"

# Model architecture
INPUT_SIZE = 9  # Number of engineered features
