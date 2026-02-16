# Phase 1: Foundation & Synthetic Data - Research

**Researched:** 2026-02-16
**Domain:** Synthetic financial data generation with Python
**Confidence:** HIGH

## Summary

Phase 1 requires generating 3,000 synthetic financial profiles with monthly history and stress labels. The standard Python stack for this task is NumPy for numerical distributions, Pandas for data manipulation, and pathlib for directory management. No additional libraries like Faker are needed since we're generating purely numerical financial data.

The key technical challenge is creating realistic correlations between income, expenses, savings, and debt while maintaining temporal coherence across 12 months. Income follows a log-normal distribution (right-skewed like real salaries), credit scores cluster around 670-750, and debt-to-income ratios should stay under 50% for most profiles. Monthly variability can be achieved with small random perturbations (10-20%) around baseline values.

**Primary recommendation:** Use `np.random.default_rng(seed)` for reproducible generation, log-normal distribution for income, normal distribution for monthly variance, and long format CSV (one row per person-month) for maximum ML flexibility.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Profile attributes:**
- Minimal field set (5-6 core financial fields): income, savings (single balance), debt, credit score
- Expenses split into 2-3 categories: essentials (rent/food), discretionary (entertainment), debt payments
- Purely financial data — no demographics (age, employment type, etc.)
- No monthly savings rate — just total savings as a snapshot

**Dataset scale & balance:**
- Default 3,000 profiles (configurable via `NUM_PROFILES` in settings.py)
- Optional random seed with a default value in settings.py for reproducibility
- Class balance (% stressed vs healthy): Claude's discretion

**Data realism:**
- Income drawn from realistic salary bands ($30k-$50k, $50k-$80k, $80k-$120k+)
- Light archetypes: 3-4 internal profile types (low-income, middle, high-income) that set base ranges for field generation
- Field correlations: Claude's discretion on how correlated fields should be
- Edge cases near stress threshold: Claude's discretion

**Temporal structure:**
- 12 months of monthly history per profile
- Monthly values vary ~10-20% around the base (income and expenses fluctuate naturally)
- Generator internally simulates month-by-month data, then the stress rule (savings < 1 month expenses OR 3+ consecutive months negative cash flow) is applied to the full history

### Claude's Discretion

- CSV structure (one row per person-month vs wide format vs derived features)
- Class balance ratio (stressed vs healthy)
- Degree of field correlation between income/expenses/savings
- Whether to include borderline edge cases near the stress threshold
- Exact archetype definitions and their parameter ranges

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope

</user_constraints>

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NumPy | 2.x | Random number generation, distributions | Industry standard for numerical computing, `np.random.default_rng()` is the modern best practice for reproducible randomness |
| Pandas | 2.x/3.x | DataFrame manipulation, CSV export | De facto standard for tabular data in Python, `to_csv()` handles all export needs |
| pathlib | stdlib | Directory creation | Built-in, object-oriented path handling is modern Python standard |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| typing | stdlib | Type hints for configuration | Improves IDE support and code clarity |
| dataclasses | stdlib | Configuration structure | Clean way to define settings without external dependencies |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| NumPy distributions | Faker library | Faker generates text data (names, addresses) which we don't need. NumPy handles numerical distributions better for financial data |
| Pathlib | os.makedirs | Pathlib is more Pythonic and readable. No reason to use legacy `os` module |
| Simple settings.py | Pydantic Settings | Pydantic adds validation and environment variable support but introduces dependency. For a learning project with simple config, native Python is sufficient |

**Installation:**
```bash
pip install numpy pandas
```

## Architecture Patterns

### Recommended Project Structure

Based on locked architecture from CLAUDE.md:
```
backend/
├── config/
│   └── settings.py          # Configuration constants (NUM_PROFILES, SEED, salary bands)
├── data/
│   └── synthetic_generator.py  # Main generation logic
└── main.py                   # Entry point (future)

data/
└── synthetic_train.csv       # Output file

.planning/
├── PROJECT.md
├── ROADMAP.md
└── phases/
    └── 01-foundation-synthetic-data/
```

**Why this structure:** Separates configuration (settings) from execution logic (generator), keeps data outputs in dedicated directory, aligns with locked architecture.

### Pattern 1: Configuration Module

**What:** Single `settings.py` file with uppercase constants for all configurable parameters

**When to use:** Simple projects where environment variables aren't needed, configuration is code-level only

**Example:**
```python
# backend/config/settings.py
# Source: https://docs.python-guide.org/writing/structure/ (Hitchhiker's Guide)

# Dataset parameters
NUM_PROFILES = 3000
RANDOM_SEED = 42
MONTHS_HISTORY = 12

# Salary bands (annual income)
SALARY_BANDS = {
    "low": (30_000, 50_000),
    "middle": (50_000, 80_000),
    "high": (80_000, 120_000),
}

# Credit score ranges (FICO)
CREDIT_SCORE_RANGE = (300, 850)
CREDIT_SCORE_MEAN = 717  # US average

# Financial ratios
MONTHLY_VARIANCE = 0.15  # ±15% fluctuation
```

### Pattern 2: Modern Random Number Generation

**What:** Use `np.random.default_rng(seed)` to create isolated Generator instances

**When to use:** Always. This is the recommended approach as of NumPy 1.17+

**Example:**
```python
# Source: https://numpy.org/doc/stable/reference/random/generator.html (Official NumPy docs)
import numpy as np
from backend.config.settings import RANDOM_SEED

# Create generator instance (isolated, reproducible)
rng = np.random.default_rng(seed=RANDOM_SEED)

# Generate income with log-normal distribution (realistic right-skew)
mean_log = np.log(60_000)  # Median income
sigma_log = 0.5
income = rng.lognormal(mean=mean_log, sigma=sigma_log, size=1000)

# Generate monthly variance (normal distribution around 1.0)
monthly_multiplier = rng.normal(loc=1.0, scale=0.15, size=12)

# Generate credit scores (normal distribution)
credit_scores = rng.normal(loc=717, scale=70, size=1000).clip(300, 850)
```

**Why this matters:**
- Avoids global state (unlike deprecated `np.random.seed()`)
- Enables parallel generation with `rng.spawn()`
- Reproducible when seeded, random when not

### Pattern 3: Long Format CSV for Time Series

**What:** One row per person-month rather than wide format (person with 12 columns for each month)

**When to use:** ML training data, time series analysis, maximum flexibility

**Example structure:**
```
profile_id,month,income,savings,expenses_essential,expenses_discretionary,debt_payment,credit_score,is_stressed
1,1,65000,8000,3500,800,1200,720,0
1,2,63000,7800,3600,750,1200,720,0
1,3,67000,8200,3400,900,1200,722,0
...
2,1,42000,1500,2800,400,800,650,1
```

**Why long format:**
- ML models consume it directly without reshaping
- Easy to filter by month or profile
- Simple to add derived features later
- Natural for time-series models

**Source:** https://towardsdatascience.com/long-and-wide-formats-in-data-explained-e48d7c9a06cb/

### Pattern 4: Archetype-Based Generation

**What:** Define 3-4 financial archetypes with realistic parameter ranges, then sample profiles from each

**When to use:** Creating diverse but realistic synthetic data

**Example:**
```python
ARCHETYPES = {
    "struggling": {
        "income_range": (30_000, 45_000),
        "savings_months": (0, 0.5),  # 0-0.5 months of expenses
        "expense_ratio": (0.85, 0.95),  # 85-95% of income
        "credit_score_mean": 620,
        "weight": 0.25,  # 25% of profiles
    },
    "stable": {
        "income_range": (50_000, 80_000),
        "savings_months": (2, 6),
        "expense_ratio": (0.65, 0.80),
        "credit_score_mean": 720,
        "weight": 0.50,
    },
    "comfortable": {
        "income_range": (80_000, 120_000),
        "savings_months": (6, 12),
        "expense_ratio": (0.50, 0.70),
        "credit_score_mean": 760,
        "weight": 0.25,
    },
}

# Generate N profiles by randomly selecting archetypes
archetype_counts = {
    name: int(NUM_PROFILES * arch["weight"])
    for name, arch in ARCHETYPES.items()
}
```

### Anti-Patterns to Avoid

- **Using deprecated `np.random.seed()` with global functions**: Creates global state, not thread-safe, harder to test. Use `default_rng()` instead.
- **Wide format CSV with 12 columns per metric**: Makes ML preprocessing harder. Use long format.
- **Generating each field independently**: Creates unrealistic data (e.g., high income with very low savings). Generate correlated fields.
- **Integer-only financial values**: Real data has cents. Use floats and round when displaying.
- **Forgetting to clip distributions**: Credit scores/debt can generate invalid values. Always clip to realistic ranges.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Random number generation | Custom PRNG or unseeded random | `np.random.default_rng(seed)` | NumPy's generators are cryptographically strong, reproducible, and battle-tested. Custom PRNGs introduce bugs and security issues. |
| CSV writing with edge cases | Manual file writing with string formatting | `pd.DataFrame.to_csv()` | Pandas handles quoting, escaping, encoding, compression, missing values automatically. CSV edge cases are deceptively complex. |
| Directory creation | Manual checking with if/else | `Path.mkdir(parents=True, exist_ok=True)` | Handles race conditions, permission errors, and platform differences. One-liner vs dozens of error-prone checks. |
| Income distribution modeling | Uniform or normal distribution | Log-normal distribution (`rng.lognormal()`) | Real income is right-skewed (many low earners, few high earners). Log-normal captures this automatically. Source: https://medium.com/data-bistrot/log-normal-distribution-with-python-7b8e384e939e |

**Key insight:** Synthetic data generation looks simple but has many subtle pitfalls. Field correlations, realistic distributions, temporal coherence, and edge cases near boundaries make hand-rolled solutions error-prone. Lean on NumPy's battle-tested distributions and Pandas' robust CSV handling.

## Common Pitfalls

### Pitfall 1: Unrealistic Field Independence

**What goes wrong:** Generating income, savings, expenses, debt independently creates impossible profiles (e.g., $30k income with $500k savings).

**Why it happens:** It's easier to generate each field separately with `rng.uniform()`.

**How to avoid:** Generate income first, then derive dependent fields with constraints:
```python
income = rng.lognormal(mean_log, sigma_log)
monthly_income = income / 12

# Expenses depend on income
expense_ratio = rng.uniform(0.6, 0.9)  # 60-90% of income
total_expenses = monthly_income * expense_ratio

# Savings constrain is relative to expenses
savings = rng.uniform(0, 6) * total_expenses  # 0-6 months expenses

# Debt payment is part of expenses
debt_payment = total_expenses * rng.uniform(0.1, 0.3)
```

**Warning signs:** Extremely high income with high stress labels, or vice versa. Negative cash flow for 12 straight months but high savings.

### Pitfall 2: Class Imbalance Ignored

**What goes wrong:** Generating profiles randomly without controlling stress/healthy ratio leads to severe imbalance (e.g., 95% healthy, 5% stressed), making ML training ineffective.

**Why it happens:** Real financial data is imbalanced toward healthy profiles, so naive generation mirrors this.

**How to avoid:**
1. Target 30-40% stressed for learning (per research, no single optimal ratio exists)
2. Design archetypes with different stress likelihoods
3. Verify balance after generation, regenerate if needed

**Warning signs:** Model predicts "always healthy" with 95% accuracy because it never learned stressed patterns.

**Source:** https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets

### Pitfall 3: Monthly Variance Creates Impossible Values

**What goes wrong:** Applying ±15% variance to savings can create negative savings (`8000 * 0.85 = 6800` is fine, but `500 * 0.85 = 425` becomes `500 * 1.15 = 575` one month and `-200` the next if done incorrectly).

**Why it happens:** Multiplicative variance doesn't respect lower bounds (like $0 for savings).

**How to avoid:**
```python
# Generate monthly multipliers
monthly_mult = rng.normal(loc=1.0, scale=0.15, size=12)

# Apply to income/expenses (can vary freely)
monthly_income = base_income * monthly_mult

# Apply to savings but clip to 0 minimum
monthly_savings = (base_savings * monthly_mult).clip(min=0)
```

**Warning signs:** Negative savings, negative income, expenses exceeding income by 300%.

### Pitfall 4: Stress Labeling Logic Errors

**What goes wrong:** Implementing "savings < 1 month expenses OR 3+ consecutive negative cash flow" with off-by-one errors, wrong comparisons, or incorrect streak counting.

**Why it happens:** The logic involves temporal dependencies (consecutive months) and multiple conditions.

**How to avoid:**
```python
def calculate_stress_label(monthly_data):
    """
    Stress if: savings < 1 month expenses OR 3+ consecutive months negative cash flow
    """
    # Condition 1: Check final savings against final expenses
    final_savings = monthly_data["savings"].iloc[-1]
    final_expenses = monthly_data["total_expenses"].iloc[-1]
    low_savings = final_savings < final_expenses

    # Condition 2: Count consecutive negative cash flow months
    cash_flow = monthly_data["income"] - monthly_data["total_expenses"]

    max_negative_streak = 0
    current_streak = 0
    for cf in cash_flow:
        if cf < 0:
            current_streak += 1
            max_negative_streak = max(max_negative_streak, current_streak)
        else:
            current_streak = 0

    negative_streak = max_negative_streak >= 3

    return 1 if (low_savings or negative_streak) else 0
```

**Warning signs:** All profiles labeled 0 or 1, labels don't match manual inspection, unexpected class balance.

### Pitfall 5: Forgetting to Create Output Directories

**What goes wrong:** `df.to_csv("data/synthetic_train.csv")` throws `FileNotFoundError` because `data/` directory doesn't exist.

**Why it happens:** CSV writing doesn't auto-create parent directories.

**How to avoid:**
```python
from pathlib import Path

output_path = Path("data/synthetic_train.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)
```

**Warning signs:** Script crashes on first run in fresh environment.

**Source:** https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html

## Code Examples

Verified patterns from official sources:

### Setting Up Reproducible Generation

```python
# Source: https://numpy.org/doc/stable/reference/random/generator.html
import numpy as np
from backend.config.settings import RANDOM_SEED

# Create isolated, reproducible generator
rng = np.random.default_rng(seed=RANDOM_SEED)

# Will produce same results every run with same seed
values = rng.random(size=10)
```

### Generating Log-Normal Income Distribution

```python
# Source: https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.lognormal.html
# Real income is right-skewed (many low, few high earners)

# For $60k median income with realistic spread
median_income = 60_000
sigma = 0.5  # Controls spread

# Log-normal parameters
mu = np.log(median_income)
annual_income = rng.lognormal(mean=mu, sigma=sigma, size=n_profiles)

# Clip to realistic bounds
annual_income = annual_income.clip(min=20_000, max=200_000)
```

### Generating Credit Scores

```python
# Source: https://www.fico.com/blogs/average-u-s-fico-score-717-more-consumers-face-financial-headwinds
# US average is 717, distribution is roughly normal

mean_score = 717
std_dev = 70  # Roughly spans 620-814 (±1.5 std dev covers ~86%)

credit_scores = rng.normal(loc=mean_score, scale=std_dev, size=n_profiles)
credit_scores = credit_scores.clip(min=300, max=850).astype(int)
```

### Creating Monthly Variance

```python
# Source: Random walk pattern for time series
# Each month varies ±15% from base values

base_income = 60_000
months = 12

# Generate multipliers (centered at 1.0, ±15% std dev)
monthly_multipliers = rng.normal(loc=1.0, scale=0.15, size=months)

# Apply to income (clip to prevent extreme values)
monthly_income = (base_income / 12 * monthly_multipliers).clip(min=base_income/12 * 0.7)
```

### Exporting to CSV (Long Format)

```python
# Source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
from pathlib import Path
import pandas as pd

# Create output directory
output_path = Path("data/synthetic_train.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Export without index, UTF-8 encoding
df.to_csv(output_path, index=False, encoding="utf-8")

# Verify
print(f"Generated {len(df)} rows")
print(f"Unique profiles: {df['profile_id'].nunique()}")
print(f"Stress ratio: {df['is_stressed'].mean():.2%}")
```

### Calculating Stress Labels

```python
def label_stress(profile_df):
    """
    Apply stress rule: savings < 1 month expenses OR 3+ consecutive negative months

    Args:
        profile_df: DataFrame for single profile with monthly data

    Returns:
        1 if stressed, 0 if healthy
    """
    # Final month savings vs expenses
    final_savings = profile_df["savings"].iloc[-1]
    final_expenses = profile_df["total_expenses"].iloc[-1]

    # Condition 1: Low savings
    low_savings = final_savings < final_expenses

    # Condition 2: Negative cash flow streak
    cash_flow = profile_df["income"] - profile_df["total_expenses"]

    max_streak = 0
    current_streak = 0
    for flow in cash_flow:
        if flow < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    negative_streak = max_streak >= 3

    return 1 if (low_savings or negative_streak) else 0
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `np.random.seed()` + global functions | `np.random.default_rng(seed)` | NumPy 1.17 (2019) | Isolated generators, thread-safe, better for testing |
| Faker for all synthetic data | NumPy distributions for numerical, Faker for text | Ongoing | Better realism for financial data with statistical distributions |
| Wide format CSV (1 row = 1 profile) | Long format CSV (1 row = 1 person-month) | Depends on use case | Long format preferred for ML time series |
| Pydantic Settings for all config | Simple settings.py for learning projects | Project-dependent | Avoid over-engineering for simple cases |
| Manual correlation modeling | Domain-informed archetypes | Best practice | Easier to reason about and maintain |

**Deprecated/outdated:**
- `np.random.seed()` and `np.random.randint()`: Use `default_rng()` instead
- `RandomState` class: Legacy interface, use Generator
- Ignoring class imbalance in synthetic data: Modern best practice is to control balance explicitly

## Open Questions

1. **Optimal class balance for this project**
   - What we know: No universal optimal ratio exists, ranges from 20-40% minority class
   - What's unclear: What ratio works best for our specific stress definition and model architecture
   - Recommendation: Start with 35% stressed, evaluate model performance, adjust if needed

2. **Degree of field correlation**
   - What we know: Real financial data has strong correlations (income→expenses, debt→credit score)
   - What's unclear: Exact correlation coefficients for synthetic realism
   - Recommendation: Use archetype-based generation with logical constraints rather than trying to match exact correlation matrices

3. **Edge cases near threshold**
   - What we know: Boundary data helps ML models learn decision boundaries
   - What's unclear: How many edge cases to include without overfitting
   - Recommendation: Include 10-15% of profiles within ±10% of stress threshold (savings = 0.9-1.1 months expenses)

4. **CSV format choice**
   - What we know: Long format is more flexible, wide format is more human-readable
   - What's unclear: Whether future phases need monthly granularity
   - Recommendation: Use long format for maximum flexibility, can aggregate later if needed

## Sources

### Primary (HIGH confidence)

- **NumPy Official Docs**: Random Generator API - https://numpy.org/doc/stable/reference/random/generator.html
- **Pandas Official Docs**: DataFrame.to_csv() - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html
- **Python Pathlib Docs**: Path.mkdir() - https://docs.python.org/3/library/pathlib.html
- **Google ML Crash Course**: Class Imbalanced Datasets - https://developers.google.com/machine-learning/crash-course/overfitting/imbalanced-datasets

### Secondary (MEDIUM confidence)

- **FICO Official Blog**: Average U.S. FICO Score at 717 - https://www.fico.com/blogs/average-u-s-fico-score-717-more-consumers-face-financial-headwinds
- **Wells Fargo**: Debt-to-Income Ratio Guidelines - https://www.wellsfargo.com/goals-credit/smarter-credit/credit-101/debt-to-income-ratio/understanding-dti/
- **Experian**: Average Credit Score in US - https://www.experian.com/blogs/ask-experian/what-is-the-average-credit-score-in-the-u-s/
- **Towards Data Science**: Long vs Wide Formats in Data - https://towardsdatascience.com/long-and-wide-formats-in-data-explained-e48d7c9a06cb/
- **Medium**: Log-Normal Distribution with Python - https://medium.com/data-bistrot/log-normal-distribution-with-python-7b8e384e939e
- **Built In**: NumPy Random Seed Best Practices - https://builtin.com/data-science/numpy-random-seed
- **DataCamp**: Creating Synthetic Data with Python Faker - https://www.datacamp.com/tutorial/creating-synthetic-data-with-python-faker-tutorial
- **Machine Learning Mastery**: Synthetic Dataset Generation with Faker - https://machinelearningmastery.com/synthetic-dataset-generation-with-faker/
- **Hitchhiker's Guide to Python**: Structuring Your Project - https://docs.python-guide.org/writing/structure/

### Tertiary (LOW confidence)

- None - All key findings verified with official sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - NumPy and Pandas are official documented solutions
- Architecture: HIGH - Follows locked architecture from CLAUDE.md, verified patterns from official docs
- Pitfalls: MEDIUM-HIGH - Based on common ML/data science patterns and official best practices
- Financial parameters: HIGH - Credit scores and DTI ratios from official financial institutions

**Research date:** 2026-02-16
**Valid until:** ~90 days (April 2026) - NumPy/Pandas are stable, financial guidelines don't change rapidly
