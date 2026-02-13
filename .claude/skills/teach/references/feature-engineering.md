# Feature Engineering Reference

## What Is Feature Engineering?

**Feature engineering**: Transform raw data into features that better represent patterns for machine learning models.

**Goal**: Help model learn more effectively by providing meaningful representations.

**Example**: Raw data vs features
- Raw: income=$5000, rent=$2000, car=$500, food=$800, savings=$1000
- Features: debt_ratio=0.66, liquidity_ratio=0.28, monthly_deficit=$-300

**Why needed**: Models learn from numbers - better numbers lead to better predictions.

## Pure Functions

### What Are Pure Functions?

**Pure function**: Function that:
1. **Same input → same output** (deterministic)
2. **No side effects** (doesn't modify external state)

**Example of pure function**:
```python
def compute_debt_ratio(monthly_income: float, monthly_debt: float) -> float:
    """
    Compute debt-to-income ratio.

    Args:
        monthly_income: Monthly income in dollars
        monthly_debt: Monthly debt payments in dollars

    Returns:
        Ratio of debt to income (0 to 1+)
    """
    if monthly_income <= 0:
        return float('inf')  # Cannot compute ratio
    return monthly_debt / monthly_income
```

**Why pure**:
- Same inputs always return same output
- Doesn't modify `monthly_income` or `monthly_debt`
- Doesn't read from database or files
- Doesn't print or log
- No randomness

### Why Pure Functions Matter

**Testability**: Easy to write unit tests
```python
assert compute_debt_ratio(5000, 2500) == 0.5
```

**Reproducibility**: Same features during training and inference
- Critical for ML - features must be computed identically

**Parallelization**: Can compute features in parallel without race conditions

**Debugging**: No hidden dependencies - input determines output

### Non-Pure Function (Avoid)

```python
# BAD: Not pure - reads from database
def compute_debt_ratio_bad(user_id: int) -> float:
    user = database.get_user(user_id)  # Side effect: database read
    return user.debt / user.income
```

**Problems**:
- Output depends on database state (non-deterministic)
- Can't compute features offline
- Hard to test (need database)
- Can't parallelize safely

## Financial Stress Indicators

### What Defines Financial Stress?

**In this project**: User is considered "stressed" if:
1. **Savings < 1 month expenses**, OR
2. **3+ consecutive months of negative cash flow**

### Why These Indicators?

**Indicator 1: Low Savings Buffer**
- **Logic**: Without 1-month buffer, single unexpected expense (car repair, medical bill) causes crisis
- **Threshold**: savings < monthly_expenses
- **Example**: Expenses=$3000/mo, Savings=$2500 → STRESS

**Indicator 2: Sustained Negative Cash Flow**
- **Logic**: One bad month is recoverable, but 3+ months indicates structural problem
- **Threshold**: 3+ consecutive months with income < expenses
- **Example**: Last 4 months all had deficits → STRESS

### Computing Stress Labels

```python
def is_financially_stressed(
    monthly_expenses: float,
    savings: float,
    monthly_cash_flows: List[float]  # Last N months
) -> bool:
    """
    Determine if user is financially stressed.

    Args:
        monthly_expenses: Average monthly expenses
        savings: Current savings balance
        monthly_cash_flows: List of monthly cash flows (income - expenses)

    Returns:
        True if stressed, False if stable
    """
    # Check indicator 1: Low savings
    if savings < monthly_expenses:
        return True

    # Check indicator 2: Sustained negative cash flow
    consecutive_negative = 0
    for cash_flow in monthly_cash_flows:
        if cash_flow < 0:
            consecutive_negative += 1
            if consecutive_negative >= 3:
                return True
        else:
            consecutive_negative = 0  # Reset counter

    return False
```

## Engineered Features

### Debt-to-Income Ratio

**Formula**: `debt_ratio = monthly_debt_payments / monthly_income`

**Interpretation**:
- 0.0 = No debt
- 0.3 = 30% of income goes to debt (reasonable)
- 0.5 = 50% of income goes to debt (high)
- 1.0+ = Debt payments exceed income (unsustainable)

**Why useful**: Captures debt burden relative to income
- Someone earning $10k with $5k debt is different from someone earning $3k with $1.5k debt
- Same ratio (0.5) but different absolute amounts

**Implementation**:
```python
def compute_debt_ratio(monthly_income: float, monthly_debt: float) -> float:
    """Compute debt-to-income ratio."""
    if monthly_income <= 0:
        return float('inf')  # Edge case: no income
    return monthly_debt / monthly_income
```

**Industry standard**: Lenders often use 43% as max acceptable ratio for loans.

### Liquidity Ratio

**Formula**: `liquidity = savings / monthly_expenses`

**Interpretation**: How many months of expenses can be covered by savings?
- 1.0 = 1 month buffer
- 3.0 = 3 months buffer (emergency fund goal)
- 6.0 = 6 months buffer (very safe)
- 0.5 = Half-month buffer (risky)

**Why useful**: Measures financial resilience
- High liquidity → can weather unexpected expenses
- Low liquidity → vulnerable to shocks

**Implementation**:
```python
def compute_liquidity_ratio(savings: float, monthly_expenses: float) -> float:
    """Compute savings-to-expenses ratio (months of buffer)."""
    if monthly_expenses <= 0:
        return float('inf')  # Edge case: no expenses
    return savings / monthly_expenses
```

**Financial advice**: 3-6 months liquidity is recommended emergency fund.

### Cash Flow Volatility

**Formula**: `volatility = std_dev(monthly_cash_flows)`

**Interpretation**: How stable is monthly cash flow?
- Low volatility → Predictable finances (salary)
- High volatility → Unpredictable finances (freelance, commission)

**Why useful**: High volatility increases risk
- Even if average cash flow is positive, high variance means some months are very negative
- Model should learn that volatility is a risk factor

**Implementation**:
```python
import numpy as np

def compute_cash_flow_volatility(monthly_cash_flows: List[float]) -> float:
    """
    Compute standard deviation of monthly cash flows.

    Args:
        monthly_cash_flows: List of monthly cash flows (income - expenses)

    Returns:
        Standard deviation (measure of volatility)
    """
    if len(monthly_cash_flows) < 2:
        return 0.0  # Need at least 2 months for std dev
    return float(np.std(monthly_cash_flows))
```

**Example**:
- Stable: [+$1000, +$1100, +$900, +$1050] → Low volatility
- Volatile: [+$3000, -$500, +$500, +$2000] → High volatility (same average!)

### Consecutive Negative Months

**Formula**: `max_consecutive_negatives = max(streak of negative cash flows)`

**Interpretation**: Longest run of months with deficits
- 0 = Always positive (ideal)
- 1 = One isolated negative month (recoverable)
- 3+ = Sustained problems (financial stress indicator)

**Why useful**: Pattern of sustained deficits predicts future stress

**Implementation**:
```python
def compute_consecutive_negative_months(monthly_cash_flows: List[float]) -> int:
    """
    Compute maximum consecutive months with negative cash flow.

    Args:
        monthly_cash_flows: List of monthly cash flows (income - expenses)

    Returns:
        Maximum streak of consecutive negative months
    """
    max_streak = 0
    current_streak = 0

    for cash_flow in monthly_cash_flows:
        if cash_flow < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0  # Reset streak

    return max_streak
```

**Edge case**: What if list is empty? Return 0 (no negative months).

## Feature Normalization

### What Is Normalization?

**Normalization**: Scale features to similar ranges (typically 0-1 or -1 to 1).

**Why needed**: Neural networks train better when features are on similar scales
- Feature 1: income ($20,000 - $200,000)
- Feature 2: liquidity ratio (0 - 10)
- Without normalization: Income dominates gradient updates

### Min-Max Normalization

**Formula**: `normalized = (value - min) / (max - min)`

**Result**: Values scaled to [0, 1] range

**Example**:
```python
def normalize_minmax(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize value to [0, 1] range.

    Args:
        value: Value to normalize
        min_val: Minimum value in training data
        max_val: Maximum value in training data

    Returns:
        Normalized value between 0 and 1
    """
    if max_val == min_val:
        return 0.0  # Edge case: all values the same
    return (value - min_val) / (max_val - min_val)
```

**Use case**: When features have known bounds
- Income: normalize by $20k - $200k range
- Debt ratio: already 0-1, no normalization needed

### Standard Normalization (Z-Score)

**Formula**: `normalized = (value - mean) / std_dev`

**Result**: Values centered around 0, std dev of 1

**Example**:
```python
def normalize_standard(value: float, mean: float, std_dev: float) -> float:
    """
    Normalize value using z-score.

    Args:
        value: Value to normalize
        mean: Mean from training data
        std_dev: Standard deviation from training data

    Returns:
        Z-score normalized value
    """
    if std_dev == 0:
        return 0.0  # Edge case: no variation
    return (value - mean) / std_dev
```

**Use case**: When features don't have known bounds
- Cash flow volatility: no fixed upper bound
- Consecutive negative months: depends on data

### Critical Rule: Use Training Statistics

**During training**: Compute mean, std, min, max from training data

**During inference**: Use **same** statistics from training
- Don't recompute mean/std from new data!
- Ensures features are on same scale as training

**Example workflow**:
```python
# Training phase
income_mean = train_data['income'].mean()
income_std = train_data['income'].std()

# Save statistics
stats = {'income_mean': income_mean, 'income_std': income_std}
with open('feature_stats.json', 'w') as f:
    json.dump(stats, f)

# Inference phase
with open('feature_stats.json', 'r') as f:
    stats = json.load(f)

# Use training statistics, not current data statistics!
normalized_income = (new_income - stats['income_mean']) / stats['income_std']
```

## Feature Importance

### What Is Feature Importance?

**Feature importance**: Measure of how much each feature contributes to predictions.

**Why useful**:
- Understand which financial factors drive predictions
- Identify if model relies on spurious correlations
- Explain predictions to users

### Methods

**1. Coefficient magnitude** (linear models):
- Larger absolute coefficient = more important

**2. Permutation importance** (any model):
- Shuffle one feature, measure performance drop
- Larger drop = more important feature

**3. SHAP values** (any model):
- Game theory approach
- Shows feature contribution to individual predictions

### Expected Importance in This Project

**Most important features** (hypothesis):
1. **Liquidity ratio** (savings buffer)
2. **Consecutive negative months** (sustained deficits)
3. **Debt-to-income ratio** (debt burden)
4. **Cash flow volatility** (financial instability)

**Why**: These directly relate to stress indicators.

### Using Feature Importance

**Model debugging**:
- If "age" is most important → something wrong (age shouldn't predict stress)
- If "income" is most important → model just predicting "rich = safe"

**Feature engineering**:
- Low-importance features can be removed (simplify model)
- High-importance features suggest creating related features

**Explainability**:
- Tell user: "Your prediction is high risk because liquidity ratio is 0.5 (should be > 1)"

## Complete Feature Engineering Pipeline

### Full Implementation

```python
import numpy as np
from typing import Dict, List

def compute_all_features(
    monthly_income: float,
    monthly_expenses: float,
    monthly_debt: float,
    savings: float,
    monthly_cash_flows: List[float]
) -> Dict[str, float]:
    """
    Compute all engineered features for ML model.

    All functions are pure - same inputs produce same outputs.

    Args:
        monthly_income: Average monthly income
        monthly_expenses: Average monthly expenses
        monthly_debt: Monthly debt payments
        savings: Current savings balance
        monthly_cash_flows: Recent monthly cash flows (income - expenses)

    Returns:
        Dictionary mapping feature names to values
    """
    features = {}

    # Ratio features
    features['debt_ratio'] = compute_debt_ratio(monthly_income, monthly_debt)
    features['liquidity_ratio'] = compute_liquidity_ratio(savings, monthly_expenses)

    # Cash flow features
    features['cash_flow_volatility'] = compute_cash_flow_volatility(monthly_cash_flows)
    features['consecutive_negative_months'] = compute_consecutive_negative_months(monthly_cash_flows)

    # Raw features (may need normalization)
    features['monthly_income'] = monthly_income
    features['monthly_expenses'] = monthly_expenses
    features['savings'] = savings

    return features

# Helper functions (defined earlier)
def compute_debt_ratio(monthly_income: float, monthly_debt: float) -> float:
    if monthly_income <= 0:
        return float('inf')
    return monthly_debt / monthly_income

def compute_liquidity_ratio(savings: float, monthly_expenses: float) -> float:
    if monthly_expenses <= 0:
        return float('inf')
    return savings / monthly_expenses

def compute_cash_flow_volatility(monthly_cash_flows: List[float]) -> float:
    if len(monthly_cash_flows) < 2:
        return 0.0
    return float(np.std(monthly_cash_flows))

def compute_consecutive_negative_months(monthly_cash_flows: List[float]) -> int:
    max_streak = 0
    current_streak = 0
    for cash_flow in monthly_cash_flows:
        if cash_flow < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak
```

### Usage in Training

```python
# For each user in training data
features_list = []
labels_list = []

for user in training_data:
    # Compute features (pure functions)
    features = compute_all_features(
        user['monthly_income'],
        user['monthly_expenses'],
        user['monthly_debt'],
        user['savings'],
        user['monthly_cash_flows']
    )
    features_list.append(list(features.values()))

    # Compute label (pure function)
    label = is_financially_stressed(
        user['monthly_expenses'],
        user['savings'],
        user['monthly_cash_flows']
    )
    labels_list.append(int(label))

# Convert to arrays
X_train = np.array(features_list)
y_train = np.array(labels_list)

# Train model
model.fit(X_train, y_train)
```

### Usage in Inference

```python
# For new user
user_data = get_user_data()

# Compute features (same pure functions!)
features = compute_all_features(
    user_data['monthly_income'],
    user_data['monthly_expenses'],
    user_data['monthly_debt'],
    user_data['savings'],
    user_data['monthly_cash_flows']
)

# Convert to array (same order as training!)
features_array = np.array(list(features.values())).reshape(1, -1)

# Predict
probability = model.predict(features_array)[0]
```

## Key Takeaways

1. **Feature engineering**: Transform raw data into meaningful patterns
2. **Pure functions**: Same input → same output, no side effects
3. **Debt ratio**: monthly_debt / monthly_income (debt burden)
4. **Liquidity ratio**: savings / monthly_expenses (buffer size)
5. **Cash flow volatility**: std_dev(monthly_cash_flows) (financial stability)
6. **Consecutive negatives**: Longest deficit streak (sustained problems)
7. **Normalization**: Scale features to similar ranges for neural networks
8. **Training statistics**: Compute from training data, reuse during inference
9. **Feature importance**: Identify which factors drive predictions
10. **Reproducibility**: Pure functions ensure training and inference match exactly
