# Roadmap

## Overview

**5 phases** to build a learning-focused ML financial risk system.

Each phase is atomic and demonstrates a key integration point.

---

## Phase 1: Foundation & Synthetic Data

**Goal**: Set up project structure and generate training data.

**Deliverables:**
- Directory structure (backend/, frontend/, models/, .planning/)
- backend/config/settings.py (basic configuration)
- backend/data/synthetic_generator.py (generates financial profiles)
- data/synthetic_train.csv (labeled training data)

**Success Criteria:**
- [ ] Synthetic generator creates N profiles with realistic income/expense/savings
- [ ] Stress labels applied correctly (savings < 1 month OR 3+ months negative)
- [ ] CSV saved and inspected manually

**Requirements Covered:**
- Generate synthetic profiles
- Apply stress labeling
- Save dataset

---

## Phase 2: ML Model & Training Pipeline

**Goal**: Build PyTorch model and train it on synthetic data.

**Deliverables:**
- backend/ml/model.py (MLP: input → 128 → 64 → output)
- backend/ml/dataset.py (PyTorch Dataset/DataLoader)
- backend/ml/train.py (training loop)
- backend/ml/evaluate.py (metrics computation)
- backend/data/feature_engineering.py (debt ratio, liquidity, cash flow features)
- models/latest_model.pth (trained model checkpoint)
- models/metrics.json (evaluation results)

**Success Criteria:**
- [ ] Model trains without errors
- [ ] Validation loss decreases
- [ ] Test recall > 0.7 (catching at-risk users)
- [ ] Metrics saved to JSON

**Requirements Covered:**
- Define MLP architecture
- Implement Dataset/DataLoader
- Training loop with 70/15/15 split
- Compute features (pure functions)
- Evaluate with recall priority
- Save model and metrics

---

## Phase 3: API Layer & Orchestration

**Goal**: Create FastAPI endpoints and orchestration logic.

**Deliverables:**
- backend/main.py (FastAPI app)
- backend/api/routes.py (POST /predict endpoint)
- backend/api/schemas.py (Pydantic request/response models)
- backend/ml/predictor.py (orchestration: load model, compute features, predict)

**Success Criteria:**
- [ ] POST /predict accepts financial data and returns risk assessment
- [ ] Routes are thin (delegate to predictor.py)
- [ ] Predictor loads model, computes features, runs inference
- [ ] Response includes: risk_score, probability, classification, insights

**Requirements Covered:**
- FastAPI endpoint
- Pydantic schemas
- Routes are thin
- ml/predictor.py orchestration

---

## Phase 4: Frontend Dashboard

**Goal**: Build simple vanilla JS dashboard.

**Deliverables:**
- frontend/index.html (dashboard UI)
- frontend/app.js (fetch calls, DOM updates)
- frontend/styles.css (clean styling)

**Success Criteria:**
- [ ] User can input financial data (form or CSV upload)
- [ ] Dashboard calls POST /predict
- [ ] Risk score, probability, classification displayed
- [ ] No frameworks, no build tools

**Requirements Covered:**
- HTML dashboard
- Vanilla JavaScript
- CSS styling
- No frameworks

---

## Phase 5: Deployment & Documentation

**Goal**: Containerize and document the system.

**Deliverables:**
- Dockerfile (single container for backend + frontend)
- .env.example (environment variable template)
- requirements.txt (Python dependencies)
- README.md (how to run, architecture overview)

**Success Criteria:**
- [ ] docker build succeeds
- [ ] docker run launches app
- [ ] App accessible at localhost
- [ ] README explains how to use

**Requirements Covered:**
- Single Dockerfile
- Environment variables
- Reproducible execution

---

## Phase Summary

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| 1 | Data | synthetic_train.csv |
| 2 | Model | latest_model.pth |
| 3 | API | POST /predict |
| 4 | Frontend | index.html dashboard |
| 5 | Deploy | Dockerfile |

---

*Roadmap created: 2026-02-13*
