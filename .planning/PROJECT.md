# ML Financial Risk System

## What This Is

A **learning-focused ML system** that predicts financial stress probability within 3-6 months. This is NOT a production SaaS - it's designed to help the user learn how to integrate ML models into applications with Claude.

**Goal**: Learn how to build apps with Claude and integrate PyTorch models into simple web applications.

## Core Value

Demonstrate clean ML system architecture with proper separation of concerns while keeping it simple and beginner-friendly.

## Problem Formulation

**Binary Classification Problem:**
- Input: Personal financial data (income, expenses, savings, debt)
- Output:
  - Risk score (0-100)
  - Probability of financial stress
  - Risk classification (Low/Moderate/High)
  - Feature-based insights

**Target Variable:**
- `1` (stress) = Savings < 1 month expenses OR 3+ consecutive months negative cash flow
- `0` (stable) = Neither condition true

## Architecture (6 Layers - Locked)

```
Frontend (vanilla HTML/CSS/JS)
    ↓
API (FastAPI - thin routes only)
    ↓
ml/predictor.py (orchestration lives here)
    ↓
Feature Engineering (pure functions)
    ↓
ML Model (PyTorch MLP: input → 128 → 64 → output)
    ↓
Synthetic Data Generator
    ↓
Files (models/latest_model.pth, models/metrics.json)
```

## Directory Structure

```
backend/
├── main.py
├── api/
│   ├── routes.py          # Thin FastAPI routes
│   └── schemas.py         # Pydantic models
├── ml/
│   ├── model.py           # MLP definition
│   ├── predictor.py       # ALL orchestration here
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Metrics
│   └── dataset.py         # PyTorch Dataset
├── data/
│   ├── synthetic_generator.py
│   ├── feature_engineering.py
│   └── preprocessing.py
└── config/
    └── settings.py

frontend/
├── index.html
├── app.js
└── styles.css

models/
├── latest_model.pth
└── metrics.json

.planning/
├── PROJECT.md
├── ROADMAP.md
├── STATE.md
└── REQUIREMENTS.md

Dockerfile
.env
requirements.txt
```

## Hard Constraints (Permanently Locked)

**NO:**
- ❌ Service layer (no services/, inference_service.py, training_service.py)
- ❌ Database (no SQLite, no persistence layer)
- ❌ React/Vue/any framework (vanilla JS only)
- ❌ docker-compose (single Dockerfile only)
- ❌ REQ-ID bureaucracy
- ❌ Production infrastructure

**YES:**
- ✓ Orchestration in ml/predictor.py
- ✓ Vanilla HTML/CSS/JS frontend
- ✓ Single Dockerfile
- ✓ Simple, beginner-friendly

## Model Specification (Locked)

**Architecture:**
```
Input Layer (n features)
    ↓
Hidden Layer 1 (128 neurons, ReLU, Dropout)
    ↓
Hidden Layer 2 (64 neurons, ReLU, Dropout)
    ↓
Output Layer (1 neuron, Sigmoid)
```

**Training:**
- Train on synthetic financial data ($20k-$200k income range)
- 70/15/15 train/val/test split
- Binary cross-entropy loss
- Adam optimizer
- Early stopping on validation loss

**Evaluation Priority:**
1. Recall (catch all at-risk users - minimize false negatives)
2. F1 Score
3. Precision
4. Accuracy
5. ROC-AUC

## Data Strategy

**Phase 1 - Training:**
- Generate synthetic financial profiles
- Income: $20k-$200k (realistic range)
- Features: expenses, savings, debt, cash flow patterns
- Apply stress labeling: savings < 1 month OR 3+ months negative cash flow

**Phase 2 - Inference:**
- User uploads financial data (CSV or web form)
- Compute same engineered features
- Run inference with trained model
- Return risk assessment

## Key Decisions

| Decision | Rationale | Status |
|----------|-----------|--------|
| No service layer | Keep simple - orchestration in ml/predictor.py | ✓ Locked |
| No database | Only save model.pth and metrics.json | ✓ Locked |
| Vanilla JS frontend | No build tools - beginner-friendly | ✓ Locked |
| Synthetic training data | No real financial data needed | ✓ Locked |
| Recall-prioritized | Better to warn stable users than miss at-risk | ✓ Locked |
| Single Dockerfile | Simple deployment, no compose complexity | ✓ Locked |

---

*Last updated: 2026-02-13 after project initialization*
