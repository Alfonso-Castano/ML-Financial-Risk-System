# Architecture Reference

## The 6-Layer Architecture

This project uses a simple 6-layer stack designed specifically for learning ML integration:

```
Frontend (vanilla HTML/CSS/JS)
    ↓
API (FastAPI - thin routes only)
    ↓
ml/predictor.py (orchestration lives here)
    ↓
Feature Engineering (pure functions)
    ↓
ML Model (PyTorch MLP)
    ↓
Files (models/latest_model.pth, models/metrics.json)
```

### Why This Architecture?

**Learning Focus**: Each layer demonstrates a key ML integration concept without production complexity.

## Design Decisions Explained

### Why No Service Layer?

**Decision**: Orchestration logic lives directly in `ml/predictor.py`, not in a separate `services/` directory.

**Rationale**:
- **Simplicity**: Beginner-friendly - one clear orchestration file instead of abstract service layer
- **Learning**: Directly shows how to coordinate: model loading → feature computation → inference → response
- **Avoids over-engineering**: Service layers are for large systems with many business rules - overkill for learning project

**What `ml/predictor.py` does**:
- Load trained model from disk
- Compute engineered features from input data
- Run model inference
- Format prediction results (risk score, probability, classification)

**Trade-off**: In production systems, a service layer provides reusability and testability. We sacrifice that for clarity.

### Why No Database?

**Decision**: No SQLite, no PostgreSQL, no ORM - just save/load files.

**What we save**:
- `models/latest_model.pth` - Trained PyTorch model checkpoint
- `models/metrics.json` - Evaluation metrics (recall, precision, F1, etc.)
- `data/synthetic_train.csv` - Generated training data

**Rationale**:
- **Learning focus**: Database adds infrastructure complexity that doesn't teach ML concepts
- **Stateless API**: Each prediction is independent - no need to store history
- **Simplicity**: File I/O is sufficient for model serving

**Trade-off**: Production systems need databases for prediction history, user profiles, A/B testing. We skip that to focus on ML.

### Why No Frontend Frameworks?

**Decision**: Vanilla HTML, CSS, JavaScript only - no React, Vue, Angular.

**Rationale**:
- **Beginner-friendly**: No build tools, no JSX, no component lifecycle - just `fetch()` calls
- **Focus on backend**: Frontend is supporting tool to test API, not the learning goal
- **Zero setup**: Open `index.html` in browser - works immediately

**What the frontend does**:
- Collect user input (income, expenses, savings, debt)
- Call `POST /predict` endpoint
- Display results (risk score, probability, classification, insights)

**Trade-off**: Production dashboards need state management, routing, components. We skip that for simplicity.

### Why Single Dockerfile?

**Decision**: One `Dockerfile` serving both backend and frontend, not `docker-compose.yml` with multiple services.

**Rationale**:
- **Simplicity**: One container, one command to run
- **Learning focus**: Understand containerization basics without orchestration complexity
- **Sufficient**: Frontend is static files served by FastAPI - no separate web server needed

**Trade-off**: Microservices architectures need docker-compose/Kubernetes. We skip that for learning.

## Separation of Concerns Principles

### Layer Responsibilities

**Frontend Layer**:
- **Responsibility**: User interface and data collection
- **Should NOT**: Compute features, train models, orchestrate logic
- **Communication**: Sends raw data to API via HTTP

**API Layer (`api/routes.py`)**:
- **Responsibility**: HTTP endpoint definitions and request/response validation
- **Should NOT**: Contain business logic, feature computation, or model code
- **Communication**: Thin wrapper - delegates immediately to `ml/predictor.py`

**Orchestration Layer (`ml/predictor.py`)**:
- **Responsibility**: Coordinate the ML prediction workflow
- **Tasks**:
  1. Load trained model
  2. Call feature engineering functions
  3. Run model inference
  4. Format response with insights
- **Should NOT**: Define routes, implement features, define model architecture

**Feature Engineering Layer (`data/feature_engineering.py`)**:
- **Responsibility**: Transform raw financial data into model-ready features
- **Pure functions**: No side effects, deterministic
- **Examples**: `compute_debt_ratio(income, debt)`, `compute_liquidity(savings, expenses)`
- **Should NOT**: Load models, call APIs, or orchestrate workflows

**ML Model Layer (`ml/model.py`)**:
- **Responsibility**: Define neural network architecture
- **PyTorch specifics**: `nn.Module` subclass with `forward()` method
- **Should NOT**: Load data, compute features, or handle HTTP requests

**File Layer**:
- **Responsibility**: Persistent storage
- **Files**: `.pth` model checkpoints, `.json` metrics, `.csv` datasets
- **Should NOT**: Contain logic - just storage

### Why This Matters

**Maintainability**: Change one layer without touching others (e.g., swap frontend without changing API)

**Testability**: Test feature functions independently of model, test model independently of API

**Clarity**: Each file has one clear purpose - easier to understand and debug

**Extensibility**: Add new features or models without rewriting orchestration

## How This Demonstrates ML Integration

### The Flow (User Request → Prediction)

1. **User input**: Financial data entered in HTML form
2. **Frontend**: Sends `POST /predict` with JSON payload
3. **API route**: Validates with Pydantic schema, calls `predictor.predict(data)`
4. **Orchestration**:
   - Loads model from `models/latest_model.pth`
   - Calls `feature_engineering.compute_features(data)`
   - Runs `model(features)` to get probability
   - Formats response with risk score and insights
5. **API returns**: JSON response with risk assessment
6. **Frontend displays**: Shows risk score, classification, probability

### Key Integration Points

**Offline Training → Online Inference**: Model trained once (`ml/train.py`), served many times (`ml/predictor.py`)

**Features Must Match**: Training features must exactly match inference features (same order, same normalization)

**Checkpoint Loading**: `.pth` file contains model weights - must match architecture in `ml/model.py`

**Stateless Serving**: Each prediction is independent - scalable and simple

## Architecture Comparison

### What We Have (Learning-Focused)
```
frontend/ → backend/api/ → backend/ml/predictor.py → backend/data/feature_engineering.py → backend/ml/model.py → models/
```
- 6 layers, clear separation
- No abstractions, direct orchestration
- File-based persistence

### What Production Might Have (NOT our scope)
```
frontend/ → API Gateway → Load Balancer → Service Layer → Feature Store → ML Platform → Database/Cache → Monitoring
```
- 10+ layers with many abstractions
- Microservices, message queues, model registries
- Database persistence, caching, A/B testing, monitoring

### Why We Don't Build Production Version

**Learning goal**: Understand **how** ML integrates into applications, not how to build scalable infrastructure

**Time**: Production architecture takes months - learning architecture takes days

**Focus**: Master the fundamentals before optimizing for scale

## Common Questions

**Q: Why not use scikit-learn?**
A: PyTorch requires explicit training loops, Dataset/DataLoader patterns - teaches more about how ML actually works.

**Q: Why not add authentication?**
A: Adds infrastructure complexity without teaching ML concepts. Out of scope for v1.

**Q: Why not store predictions in a database?**
A: Stateless serving is simpler and sufficient for learning. Prediction history is a product feature, not ML core.

**Q: Is this production-ready?**
A: No. This is intentionally simplified for learning. Production needs error handling, monitoring, scaling, security, etc.

**Q: Can I add features later?**
A: Yes! The separation of concerns makes it easy to extend. Just follow the existing patterns.
