**1. Project Objective**

The goal of this project is to build a modular, ML-powered application that evaluates personal financial resilience and predicts the probability of financial stress within a 3–6 month horizon.

The system will:
- Accept structured financial data
- Perform feature engineering
- Use a PyTorch-based neural network to predict financial stress probability
- Output:
    - Risk Score (0–100)
    - Probability of Financial Stress
    - Risk Classification (Low / Moderate / High)
    - Feature-based insights
- Provide a clean dashboard interface
- Run locally
- Be fully containerized with Docker

This project must demonstrate clean ML engineering practices, modular architecture, and clear separation of concerns.

**2. Problem Formulation**

This is a supervised binary classification problem.

We define:
𝑃(Financial Stress in next 3–6 months)

Target variable:
1 = Financial stress event likely
0 = Financially stable
Financial stress is programmatically defined based on structural financial instability conditions (e.g., liquidity thresholds, debt ratios, sustained negative cash flow).
The system must allow stress labeling logic to be configurable and isolated from model logic.

**3. Data Strategy**
**Phase 1 – Synthetic Training Data**

The model will be trained using fully synthetic financial data generated via controlled statistical distributions.
Synthetic profiles will include:
- Income patterns
- Expense behavior
- Savings rates
- Debt ratios
- Liquidity metrics
- Volatility indicators
Stress labels will be generated using rule-based financial instability logic.
Synthetic generation must be modular and configurable.

**Phase 2 – Real-World Inference**

Users can upload structured financial data (e.g., CSV).
The system will:
- Compute engineered features
- Run inference using trained model
- Output risk metrics
The initial version does not require retraining on user data.

4. System Architecture

The system must follow a layered SaaS-style architecture.

Frontend Dashboard
        ↓
FastAPI Backend (API Layer)
        ↓
Service Layer (Orchestration)
        ↓
Feature Engineering Layer
        ↓
ML Model Layer (PyTorch)
        ↓
Persistence Layer (SQLite)

Architectural Principles:
- Training logic must be separated from inference logic.
- Feature engineering must be independent of API routes.
- API routes must not contain business logic.
- The ML model must be isolated from preprocessing logic.
- Configuration must not be hard-coded.

The system must be modular and extensible.

**5. Backend Structure (Expected)**
backend/
│
├── main.py
├── api/
│   ├── routes.py
│   └── schemas.py
│
├── services/
│   ├── inference_service.py
│   ├── training_service.py
│   └── recommendation_engine.py
│
├── ml/
│   ├── model.py
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── data/
│   ├── synthetic_generator.py
│   ├── feature_engineering.py
│   └── preprocessing.py
│
└── config/
    └── settings.py

This structure may be refined but must preserve clean separation of responsibilities.

6. ML Requirements

The ML component must:
- Use PyTorch
- Implement a custom neural network (MLP)
- Use explicit training loop
- Use Dataset + DataLoader abstractions
- Perform train/validation/test split
- Compute evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Save and load trained models
- Use configurable hyperparameters

Prohibited shortcuts:
- No sklearn classifier wrappers
- No notebook-only pipelines
- No mixing preprocessing inside model forward pass

7. Frontend Requirements

The frontend should:
- Provide a clean, modern dashboard
- Display:
    - Risk Score
    - Stress Probability
    - Risk Category
    - Key financial metrics
- Allow financial data upload
- Be visually structured but not overly complex
- Not dominate project complexity

Frontend implementation can be minimal but must feel product-like.

8. Docker Requirements

The system must:
- Run locally
- Be containerized
- Have a proper Dockerfile
- Use environment variables correctly
- Support mounted data directories
- Allow reproducible execution

The architecture should reflect production-style organization even if deployment is local.

9. Workflows Within WAT Framework

The following high-level workflows are expected:
1. System Architecture Planning Workflow
2. Synthetic Data Generation Workflow
3. ML Model Training Workflow
4. Model Evaluation Workflow
5. Inference & API Integration Workflow
6. Frontend Integration Workflow
7. Dockerization Workflow
8. Testing & Refactoring Workflow

Claude should:
- Suggest refinements to these workflows
- Identify missing workflows
- Propose improvements in modularity
- Suggest better abstractions if necessary

10. Non-Goals

This project is NOT:
- A stock prediction system
- A real banking system
- A production financial compliance tool
- A UI-heavy frontend project
- A Kaggle competition notebook

It is an ML engineering system demonstrating clean architecture and applied financial modeling.