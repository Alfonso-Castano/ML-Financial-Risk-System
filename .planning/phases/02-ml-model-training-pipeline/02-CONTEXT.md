# Phase 2: ML Model & Training Pipeline - Context

**Gathered:** 2026-02-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a PyTorch MLP that learns to classify financial stress from derived features, trains on synthetic_train.csv, and produces a saved checkpoint with evaluation artifacts. API integration and inference are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Feature engineering
- Three derived features: **debt ratio**, **liquidity ratio**, **net cash flow**
- Debt ratio = `debt_payments / (income - expenses)` (relative to free cash flow, not gross income)
- Liquidity ratio = `savings / monthly_expenses`
- Net cash flow = `income - expenses - debt_payments`
- **Normalization: z-score standardization** (mean 0, std 1) applied before model input
- Features computed per row (single-month snapshot) — Claude decides on time aggregation

### Training configuration
- **75 epochs**
- **Batch size: 64**
- Optimizer: Claude's discretion (standard for this type of MLP)
- Class weighting: Claude's discretion based on recall > 0.7 target

### Training feedback
- **Progress bar (tqdm)** during training — no per-epoch log spam
- **Final summary printed to console** after training: recall and key loss values
- **Save best validation model** (lowest val loss) as `models/latest_model.pth` — not the final epoch

### Training CLI
- Claude's discretion on command-line arguments — keep appropriate for a learning project

### Evaluation & artifacts
- **metrics.json contains:** accuracy, precision, recall, F1, AUC-ROC
- **Plots:** save silently to `models/` and print file paths when done (no interactive display)
- **Data split:** Claude's discretion — fixed seed for reproducibility
- **evaluate.py is a standalone script** (`python evaluate.py`) — runs independently on a saved model, not called by train.py

### Claude's Discretion
- Optimizer choice (Adam is standard)
- Whether to apply class weighting (likely yes, given recall > 0.7 success criterion)
- Time aggregation vs single-row features (single-row is simpler and appropriate)
- Random seed value for reproducibility
- CLI argument design for train.py

</decisions>

<specifics>
## Specific Ideas

No specific references mentioned — open to standard PyTorch training patterns.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 02-ml-model-training-pipeline*
*Context gathered: 2026-02-20*
