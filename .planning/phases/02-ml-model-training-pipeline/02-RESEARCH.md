# Phase 2: ML Model & Training Pipeline - Research

**Researched:** 2026-02-20
**Domain:** PyTorch MLP training on aggregated time-series financial data
**Confidence:** HIGH

---

## Summary

Phase 2 builds a PyTorch MLP (9 → 128 → 64 → 1) that learns to predict financial stress from 12-month financial histories. The dataset is long-format (one row per person-month), so the first and most important design decision is how to transform it into one-sample-per-profile format that the model can consume. The answer is feature engineering: aggregate each profile's 12 rows into 9 scalar features (5 raw + 4 engineered).

The ML stack is already partially installed: PyTorch 2.10.0 (CPU), scikit-learn 1.8.0, NumPy 2.3.4, and Pandas 2.3.3 are all present. **Matplotlib is NOT installed** and must be added to `requirements.txt` and installed before evaluate.py can generate training plots. The class imbalance (36.2% positive) is mild enough that standard BCELoss with Sigmoid works without special handling, though a small pos_weight boost would help recall if needed.

The split must be by `profile_id`, not by row. Splitting by row would leak a single person's data across train/val/test, inflating performance. Stratified split by label preserves the 36.2% stress ratio across all three splits. All architecture details are locked: no alternatives need consideration.

**Primary recommendation:** Aggregate 12 monthly rows per profile into 9 features using pure functions in `feature_engineering.py`, split 3000 profiles by person ID with stratification, train with BCELoss + Adam, save best-val-loss model to `models/latest_model.pth`, and generate plots with matplotlib after installing it.

---

## Data Schema (Verified from `data/synthetic_train.csv`)

**Shape:** 36,000 rows × 11 columns (3,000 profiles × 12 months each)

**Columns:**

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `profile_id` | int | Person identifier | 1 to 3000 |
| `month` | int | Month number | 1 to 12 |
| `income` | float | Monthly income | Varies ±15% per month |
| `essentials` | float | Essential expenses | Varies ±15% per month |
| `discretionary` | float | Discretionary spending | Varies ±15% per month |
| `debt_payment` | float | Monthly debt payment | Fixed per profile |
| `total_expenses` | float | Sum of all expense categories | Varies per month |
| `savings` | float | Cumulative savings balance | Running total, cannot go negative |
| `debt` | float | Annualized debt (debt_payment × 12) | Fixed per profile |
| `credit_score` | int | Credit score | Fixed per profile, 499–850 |
| `is_stressed` | int | Label (0 or 1) | Same for all 12 rows of a profile |

**Label distribution:**
- Stressed (1): 1,086 profiles — 36.2%
- Healthy (0): 1,914 profiles — 63.8%
- This is mild imbalance, not severe. Standard BCE loss handles it.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0 (installed) | Model definition, training, tensors | The dominant deep learning framework for research and mid-scale production |
| scikit-learn | 1.8.0 (installed) | StandardScaler, train_test_split, metrics | Industry standard for ML utilities, metrics, and preprocessing |
| NumPy | 2.3.4 (installed) | Array operations in feature engineering | Core numerical computing |
| Pandas | 2.3.3 (installed) | Reading CSV, groupby aggregations | Standard tabular data library |
| matplotlib | **NOT INSTALLED** | Training plots (loss curves, confusion matrix, ROC) | Standard plotting for Python ML |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `json` (stdlib) | — | Save metrics.json, scaler stats | Always, no external dep needed |
| `pathlib` (stdlib) | — | Create `models/` directory before saving | Always use over `os.path` |
| `typing` (stdlib) | — | Type hints for feature functions | All pure functions should be typed |

### Installation Required

```bash
pip install matplotlib
```

And update `requirements.txt`:
```
numpy>=2.0
pandas>=2.0
torch>=2.0
scikit-learn>=1.0
matplotlib>=3.0
```

### Alternatives Considered

| Instead of | Could Use | Why Not for This Project |
|------------|-----------|--------------------------|
| BCELoss + Sigmoid | BCEWithLogitsLoss (no Sigmoid in model) | BCELoss is more educational — shows Sigmoid explicitly in architecture. Imbalance is only 36%:64%, not severe enough to require BCEWithLogitsLoss's `pos_weight` |
| StandardScaler (sklearn) | Manual z-score normalization | StandardScaler is simpler, already installed, handles fit/transform correctly |
| train_test_split (sklearn) | Manual index shuffling | One function call, supports stratify parameter |
| matplotlib | seaborn, plotly | matplotlib is already the standard for static training plots, no additional dependency |

---

## Feature Engineering Design

### The Core Problem: Long Format to Profile-Level

The CSV has 12 rows per profile. The model needs ONE feature vector per profile. Feature engineering solves this by aggregating the 12 monthly rows into scalar summaries.

**Rule:** Feature engineering happens in `backend/data/feature_engineering.py` as pure functions. The `dataset.py` calls these functions to build the feature matrix.

### The 9 Features (Input Size = 9)

**5 Raw/Aggregated Features:**

| Feature | Formula | Why Include |
|---------|---------|-------------|
| `avg_income` | `mean(income)` over 12 months | Base income level |
| `avg_expenses` | `mean(total_expenses)` over 12 months | Base expense level |
| `final_savings` | `savings` at `month=12` | Current financial cushion |
| `debt_payment` | `debt_payment` (fixed per profile) | Fixed monthly obligation |
| `credit_score` | `credit_score` (fixed per profile) | Financial health signal |

**4 Engineered Features:**

| Feature | Formula | Correlation with Stress | Why Include |
|---------|---------|------------------------|-------------|
| `debt_ratio` | `debt_payment / avg_income` | **+0.773** (highest) | Captures debt burden normalized by income |
| `liquidity_ratio` | `final_savings / avg_expenses` | **-0.651** | Months of expenses covered by savings (directly maps to stress condition 1) |
| `cf_volatility` | `std(income - total_expenses)` over 12 months | **-0.515** | Measures income/expense stability |
| `consec_negative_months` | Max consecutive months where `income < total_expenses` | **+0.769** (second highest) | Directly captures stress condition 2 |

**Correlation findings** (verified from data): `debt_ratio` (0.773) and `consec_negative_months` (0.769) are the two strongest predictors — they directly correspond to the two stress labeling rules. This is expected and validates the feature design.

### Pure Function Signatures

```python
# backend/data/feature_engineering.py

def compute_debt_ratio(avg_monthly_income: float, monthly_debt_payment: float) -> float:
    """debt_payment / avg_income. Returns 0.0 if income <= 0."""

def compute_liquidity_ratio(final_savings: float, avg_monthly_expenses: float) -> float:
    """final_savings / avg_expenses. Returns 0.0 if expenses <= 0."""

def compute_cash_flow_volatility(monthly_cash_flows: list[float]) -> float:
    """std(income - total_expenses per month). Returns 0.0 if < 2 data points."""

def compute_consecutive_negative_months(monthly_cash_flows: list[float]) -> int:
    """Max consecutive months where cash_flow < 0."""

def engineer_features(profile_df: pd.DataFrame) -> dict[str, float]:
    """
    Takes one profile's 12 monthly rows.
    Returns dict with all 9 features.
    Pure function — same input always produces same output.
    """
```

### Edge Case Handling

- `avg_income == 0` in `debt_ratio`: Return `0.0` (not `inf`) — `inf` breaks neural network gradients
- `avg_expenses == 0` in `liquidity_ratio`: Return `0.0` — same reason
- `len(cash_flows) < 2` in volatility: Return `0.0`
- `savings` should always be `month=12` row specifically (not average) — it represents current balance

---

## Dataset & DataLoader Design

### Split Must Be By Profile ID (Not By Row)

**Why this matters:** If you split by row, the same person's data (12 rows) will appear in both train and test. The model would effectively see test data during training (data leakage), producing optimistically inflated metrics.

**Correct approach:**
1. Get unique profile IDs and their labels
2. Split profile IDs with stratification
3. Filter full dataframe by profile ID for each split

```python
# Verified working (from data analysis):
from sklearn.model_selection import train_test_split

profile_labels = df.groupby('profile_id')['is_stressed'].first()
profile_ids = profile_labels.index.values
labels = profile_labels.values

# Step 1: Carve out test set
ids_temp, ids_test, y_temp, y_test = train_test_split(
    profile_ids, labels, test_size=0.15, random_state=42, stratify=labels
)

# Step 2: Split remaining into train/val
# 0.176 = 0.15 / 0.85 (to get 15% of total as val)
ids_train, ids_val, y_train, y_val = train_test_split(
    ids_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
```

**Verified split sizes:**
- Train: 2,101 profiles — 36.2% stressed
- Val: 449 profiles — 36.3% stressed
- Test: 450 profiles — 36.2% stressed

Stratification works correctly — stress ratio is consistent across splits.

### PyTorch Dataset Class

The `FinancialDataset` class in `backend/ml/dataset.py` should:
1. Accept a list/array of profile IDs and the full dataframe
2. For each profile: call `engineer_features()` from `feature_engineering.py` to get 9 features
3. Apply `StandardScaler` (fitted on train set only)
4. Return `(features_tensor, label_tensor)` from `__getitem__`

```python
class FinancialDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
```

Scaling should happen **before** creating the Dataset (in the training script), not inside the Dataset. This keeps the Dataset simple and the scaling logic visible in `train.py`.

### DataLoader Configuration

```python
# Windows note: num_workers > 0 requires __main__ guard (multiprocessing spawn)
# Use num_workers=0 for simplicity and Windows compatibility
DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
```

**Batch size 32:** Standard starting point for tabular data. Larger batches (64, 128) are fine too but 32 is most educational.

**Shuffle=True on train, False on val/test.**

---

## Model Architecture (Locked)

```python
# backend/ml/model.py
class FinancialRiskModel(nn.Module):
    def __init__(self, input_size: int = 9):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
```

**input_size=9** (derived from the 9 features above). Pass this as a constructor argument so evaluate.py and predictor.py can instantiate the model without hardcoding.

---

## Training Loop Design

### Loss Function

**Use: `nn.BCELoss()`**

Rationale: The model ends with `nn.Sigmoid()` which outputs probabilities (0–1). BCELoss expects probabilities. This is the mathematically correct pairing.

Note: `nn.BCEWithLogitsLoss()` would require removing `nn.Sigmoid()` from the model. That's numerically more stable but less educational (hides what Sigmoid does). Since the architecture is locked with Sigmoid, BCELoss is correct.

**Class imbalance:** 36.2% positive is not severe. Model will learn both classes adequately. If recall < 0.7 after training, consider lowering decision threshold from 0.5 to 0.4 rather than changing the loss function.

### Optimizer

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

Adam with lr=0.001 is the standard starting point. No scheduler needed for this scale.

### Training Loop Structure (Educational clarity)

```python
def train_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        optimizer.zero_grad()           # 1. Clear gradients
        outputs = model(features).squeeze()  # 2. Forward pass
        loss = criterion(outputs, labels)    # 3. Compute loss
        loss.backward()                      # 4. Backprop
        optimizer.step()                     # 5. Update weights
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)
```

Separate `train_epoch` and `validate_epoch` functions keep the loop readable and educational. Each function has one responsibility.

### Recommended Hyperparameters

| Param | Value | Reason |
|-------|-------|--------|
| `num_epochs` | 50 | Sufficient for 3000 profiles, fast training on CPU |
| `batch_size` | 32 | Standard tabular baseline |
| `learning_rate` | 0.001 | Adam standard starting point |
| `dropout` | 0.3 | Locked in architecture |
| `early_stopping_patience` | 10 epochs | Save best model, stop if val_loss doesn't improve |

### Model Checkpointing

Save model after each epoch if validation loss improves:

```python
# Save best model (by validation loss)
torch.save(model.state_dict(), 'models/latest_model.pth')
```

In PyTorch 2.10, `torch.load` with `weights_only=True` is the default (verified). Loading should use:

```python
model = FinancialRiskModel(input_size=9)
model.load_state_dict(torch.load('models/latest_model.pth', weights_only=True))
model.eval()
```

---

## Evaluation Design

### Metrics to Compute

```python
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    accuracy_score, roc_auc_score, confusion_matrix, roc_curve
)
```

All verified working in scikit-learn 1.8.0.

**Compute on test set only, after training is complete.**

```python
# Get predictions from trained model
model.eval()
with torch.no_grad():
    probs = model(X_test_tensor).squeeze().numpy()  # Probabilities (0-1)
preds = (probs >= 0.5).astype(int)  # Binary predictions

metrics = {
    "recall": float(recall_score(y_test, preds)),
    "precision": float(precision_score(y_test, preds)),
    "f1": float(f1_score(y_test, preds)),
    "accuracy": float(accuracy_score(y_test, preds)),
    "roc_auc": float(roc_auc_score(y_test, probs)),
    "threshold": 0.5
}
```

**Success criterion: recall > 0.7.** Given strong feature-label correlations (debt_ratio: 0.773, consec_negative: 0.769) this is achievable.

### metrics.json Structure

```json
{
  "recall": 0.82,
  "precision": 0.75,
  "f1": 0.78,
  "accuracy": 0.84,
  "roc_auc": 0.91,
  "threshold": 0.5,
  "train_profiles": 2101,
  "val_profiles": 449,
  "test_profiles": 450,
  "num_epochs_trained": 42,
  "input_size": 9
}
```

Save with: `json.dump(metrics, f, indent=2)`

### Visualization: Three Plots

**1. Loss Curves (`models/loss_curves.png`)**

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('models/loss_curves.png', dpi=100)
plt.close()
```

**2. Confusion Matrix (`models/confusion_matrix.png`)**

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Healthy', 'Stressed'])
ax.set_yticklabels(['Healthy', 'Stressed'])
# Add text annotations
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
plt.title('Confusion Matrix')
plt.colorbar(im)
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=100)
plt.close()
```

**3. ROC Curve (`models/roc_curve.png`)**

```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('models/roc_curve.png', dpi=100)
plt.close()
```

**Always call `plt.close()` after saving** to release memory. Use `plt.tight_layout()` before saving to prevent label clipping.

---

## Architecture Patterns

### Recommended File Responsibilities

```
backend/data/feature_engineering.py
    - Pure functions: compute_debt_ratio, compute_liquidity_ratio,
      compute_cash_flow_volatility, compute_consecutive_negative_months
    - engineer_features(profile_df) -> dict: orchestrates the 9-feature pipeline

backend/ml/dataset.py
    - FinancialDataset(Dataset): accepts pre-scaled numpy arrays, converts to tensors
    - build_datasets(csv_path) -> (train_ds, val_ds, test_ds, scaler, feature_names)
      This function reads CSV, engineers features, splits, scales, creates datasets

backend/ml/model.py
    - FinancialRiskModel(nn.Module): locked MLP architecture
    - No training logic here — just the model definition

backend/ml/train.py
    - main() entry point: loads data, creates model, runs training loop
    - Saves best model to models/latest_model.pth
    - Saves train_losses, val_losses lists for plotting
    - Calls evaluate.py at the end

backend/ml/evaluate.py
    - compute_metrics(model, X_test, y_test) -> dict: computes all 5 metrics
    - save_metrics(metrics, path) -> None: writes metrics.json
    - plot_loss_curves(train_losses, val_losses, path)
    - plot_confusion_matrix(y_true, y_pred, path)
    - plot_roc_curve(y_true, y_probs, path)
```

### Anti-Patterns to Avoid

- **Splitting by row instead of profile_id:** Causes data leakage. A profile's 12 months must all go to the same split.
- **Fitting StandardScaler on val/test data:** Fit on train only. Transform val and test using train statistics.
- **Using `inf` in features:** `debt_payment / 0` returns `inf` which breaks gradient computation. Always return `0.0` for zero-denominator cases.
- **Forgetting `model.train()` / `model.eval()`:** Dropout behaves differently in each mode. Always set correctly.
- **Forgetting `.squeeze()` on model output:** Model returns shape `(batch, 1)`. BCELoss expects `(batch,)`. Call `.squeeze()`.
- **Saving final model instead of best-val-loss model:** The final epoch model may be overfit. Track best val loss and save that checkpoint.
- **Using `num_workers > 0` on Windows without `__main__` guard:** Will hang or crash. Use `num_workers=0`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Train/val/test split | Manual shuffle/slice | `sklearn.train_test_split` with `stratify=` | Handles stratification automatically, single function call |
| Feature normalization | Manual mean/std subtraction | `sklearn.StandardScaler` | Handles fit/transform pattern correctly, serialize statistics via `mean_` and `scale_` attributes |
| Classification metrics | Manual TP/FP/FN counting | `sklearn.metrics.*` | Handles edge cases (zero division), consistent API |
| ROC curve computation | Manual threshold sweeping | `sklearn.metrics.roc_curve` | Correctly handles all thresholds |
| Confusion matrix | Manual label comparison | `sklearn.metrics.confusion_matrix` | Returns numpy array, consistent format |

---

## Common Pitfalls

### Pitfall 1: Splitting by Row Instead of Profile
**What goes wrong:** Model sees test person's data during training, recall appears higher than reality.
**Why it happens:** `train_test_split(df)` splits rows, not profiles.
**How to avoid:** Extract unique profile IDs first, split those, then filter rows.
**Warning signs:** Val/test accuracy suspiciously close to 100%.

### Pitfall 2: Forgetting `.squeeze()` on Model Output
**What goes wrong:** `BCELoss` throws shape mismatch error — "Expected target size (32,), got (32, 1)".
**Why it happens:** `nn.Linear(64, 1)` outputs shape `(batch_size, 1)`, not `(batch_size,)`.
**How to avoid:** `outputs = model(features).squeeze()` in training loop.
**Warning signs:** RuntimeError about tensor shape mismatch.

### Pitfall 3: StandardScaler Fitted on Wrong Data
**What goes wrong:** Features normalized with val/test statistics — subtle but wrong.
**Why it happens:** Fitting scaler before splitting, or refitting on val/test.
**How to avoid:** `scaler.fit_transform(X_train)`, then `scaler.transform(X_val)`, `scaler.transform(X_test)`.
**Warning signs:** No error — just slightly misleading metrics.

### Pitfall 4: matplotlib Not Installed
**What goes wrong:** `ImportError: No module named 'matplotlib'` at the start of evaluate.py.
**Why it happens:** Matplotlib is NOT in the current environment (verified).
**How to avoid:** Install before running: `pip install matplotlib`. Add to `requirements.txt`.
**Warning signs:** First import of matplotlib fails immediately.

### Pitfall 5: Saving Metrics with Non-Serializable Types
**What goes wrong:** `TypeError: Object of type float32 is not JSON serializable`.
**Why it happens:** sklearn metrics return numpy float32/int64, not Python floats.
**How to avoid:** Wrap every metric: `"recall": float(recall_score(...))`.
**Warning signs:** json.dump raises TypeError.

### Pitfall 6: Model Not in eval() Mode During Inference
**What goes wrong:** Dropout randomly drops neurons during evaluation, producing different results on each call.
**Why it happens:** `model.eval()` forgotten after training.
**How to avoid:** Always call `model.eval()` before computing test metrics. Pair with `torch.no_grad()`.

### Pitfall 7: `final_savings` from Wrong Row
**What goes wrong:** If you use `savings.mean()` instead of `savings` at `month=12`, you miss the fact that stress condition 1 checks the *final* balance.
**Why it happens:** Intuition says "aggregate with mean".
**How to avoid:** `final_savings = group[group['month'] == 12]['savings'].iloc[0]` or `group.sort_values('month').iloc[-1]['savings']`.

---

## Code Examples

### Feature Engineering (Verified Pattern)

```python
# backend/data/feature_engineering.py
import numpy as np
import pandas as pd
from typing import Dict, List

def compute_debt_ratio(avg_monthly_income: float, monthly_debt_payment: float) -> float:
    if avg_monthly_income <= 0:
        return 0.0
    return monthly_debt_payment / avg_monthly_income

def compute_liquidity_ratio(final_savings: float, avg_monthly_expenses: float) -> float:
    if avg_monthly_expenses <= 0:
        return 0.0
    return final_savings / avg_monthly_expenses

def compute_cash_flow_volatility(monthly_cash_flows: List[float]) -> float:
    if len(monthly_cash_flows) < 2:
        return 0.0
    return float(np.std(monthly_cash_flows))

def compute_consecutive_negative_months(monthly_cash_flows: List[float]) -> int:
    max_streak = 0
    current_streak = 0
    for cf in monthly_cash_flows:
        if cf < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def engineer_features(profile_df: pd.DataFrame) -> Dict[str, float]:
    """
    Transform one profile's 12 monthly rows into a single feature dict.
    Pure function — no side effects.
    """
    group = profile_df.sort_values('month')
    cash_flows = (group['income'] - group['total_expenses']).tolist()

    avg_income = float(group['income'].mean())
    avg_expenses = float(group['total_expenses'].mean())
    final_savings = float(group.iloc[-1]['savings'])
    debt_payment = float(group.iloc[0]['debt_payment'])
    credit_score = float(group.iloc[0]['credit_score'])

    return {
        'avg_income': avg_income,
        'avg_expenses': avg_expenses,
        'final_savings': final_savings,
        'debt_payment': debt_payment,
        'credit_score': credit_score,
        'debt_ratio': compute_debt_ratio(avg_income, debt_payment),
        'liquidity_ratio': compute_liquidity_ratio(final_savings, avg_expenses),
        'cf_volatility': compute_cash_flow_volatility(cash_flows),
        'consec_negative_months': compute_consecutive_negative_months(cash_flows),
    }
```

### Building Feature Matrix (Verified Pattern)

```python
# In dataset.py or train.py
def build_feature_matrix(df: pd.DataFrame) -> tuple:
    """
    Aggregate long-format dataframe into profile-level feature matrix.
    Returns X (np.ndarray shape [n_profiles, 9]), y (np.ndarray shape [n_profiles])
    """
    rows = []
    labels = []
    profile_ids = []

    for pid, group in df.groupby('profile_id'):
        features = engineer_features(group)
        rows.append(list(features.values()))
        labels.append(int(group['is_stressed'].iloc[0]))
        profile_ids.append(pid)

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)
```

### Train/Val/Test Split by Profile (Verified Working)

```python
from sklearn.model_selection import train_test_split

# profile_ids: array of unique IDs
# labels: stress label per profile

ids_temp, ids_test, y_temp, y_test = train_test_split(
    profile_ids, labels, test_size=0.15, random_state=42, stratify=labels
)
ids_train, ids_val, y_train, y_val = train_test_split(
    ids_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)
# Result: 2101 train, 449 val, 450 test — all ~36.2% stressed
```

### Training Loop (Educational Pattern)

```python
# From pytorch-patterns.md + project-specific adjustments
best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'models/latest_model.pth')

    print(f"Epoch {epoch+1}/{num_epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
```

---

## Open Questions

1. **Should scaler statistics be saved alongside the model?**
   - What we know: Phase 3 (API) needs to normalize incoming API requests with the same statistics used during training. If scaler statistics are not saved, Phase 3 cannot work correctly.
   - Recommendation: Save scaler statistics to `models/metrics.json` alongside the evaluation metrics, or a separate `models/scaler_stats.json`. The planner should decide which file.

2. **What should `input_size` be hardcoded to vs configurable?**
   - What we know: 9 features are determined by the feature engineering design.
   - Recommendation: Pass `input_size=9` as default argument to `FinancialRiskModel.__init__`. Document it clearly. Don't add it to `settings.py` — it's derived from feature count, not an independent setting.

3. **Should `train.py` be runnable as a script directly?**
   - What we know: The `__init__.py` files already exist in `backend/ml/`. The project structure uses module imports (e.g., `from backend.config.settings import ...`).
   - Recommendation: Yes, `python -m backend.ml.train` from project root should work. Use `if __name__ == '__main__':` guard (also required for Windows multiprocessing safety).

---

## State of the Art

| Old Approach | Current Approach | Impact for This Project |
|--------------|------------------|------------------------|
| `model.state_dict()` loaded with `weights_only=False` | `weights_only=True` (default in PyTorch 2.6+) | Use `weights_only=True` when loading — it's the secure default in our version (2.10) |
| Wide-format aggregation (separate column per month) | Long-format with groupby aggregation | We have long-format — groupby per profile_id to aggregate |
| `np.random.seed()` global seeding | `torch.manual_seed(42)` for reproducibility | Set `torch.manual_seed(42)` at start of training for reproducible results |
| Manual metric computation | sklearn.metrics | sklearn handles all edge cases (zero division, etc.) |

---

## Sources

### Primary (HIGH confidence — verified from running code in this environment)

- **data/synthetic_train.csv** — Schema, label distribution, profile structure confirmed by direct inspection
- **PyTorch 2.10.0 installed** — Version confirmed via `torch.__version__`
- **scikit-learn 1.8.0 installed** — All metrics API verified working
- **Feature correlations** — Computed directly from CSV: debt_ratio (0.773), consec_negative (0.769), liquidity_ratio (-0.651) are strongest predictors
- **Split sizes** — 2101/449/450 confirmed working with 0.15/0.176 test_size params
- **`.claude/skills/teach/references/pytorch-patterns.md`** — Project-specific PyTorch reference with validated patterns
- **`.claude/skills/teach/references/feature-engineering.md`** — Project-specific feature engineering reference
- **`.claude/skills/teach/references/ml-concepts.md`** — Project-specific ML concepts reference

### Secondary (MEDIUM confidence)

- **PyTorch official docs** — nn.BCELoss, nn.Module, DataLoader patterns (consistent with installed version)
- **scikit-learn official docs** — StandardScaler fit/transform pattern, train_test_split stratify parameter

### Tertiary (LOW confidence — none in this research)

---

## Metadata

**Confidence breakdown:**
- Data schema: HIGH — read directly from CSV
- Feature engineering: HIGH — computed correlations and stats from actual data
- Split design (by profile): HIGH — data leakage reasoning is definitive; verified split sizes
- Standard stack: HIGH — all packages verified installed and API tested
- matplotlib missing: HIGH — confirmed `ModuleNotFoundError`
- Loss function choice (BCELoss): HIGH — mathematically required by locked Sigmoid output
- Hyperparameters (epochs=50, lr=0.001): MEDIUM — standard starting points, may need tuning
- Recall > 0.7 achievability: MEDIUM — feature correlations are strong, but actual training result unknown until executed
- Scaler stats persistence: MEDIUM — Phase 3 will need this, but exact storage location is open

**Research date:** 2026-02-20
**Valid until:** 2026-05-20 (PyTorch and sklearn are stable; matplotlib API is stable)
