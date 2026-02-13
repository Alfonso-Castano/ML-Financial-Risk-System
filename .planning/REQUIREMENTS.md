# Requirements

## v1 Requirements

### Data Generation
- [ ] Generate synthetic financial profiles with configurable income range ($20k-$200k)
- [ ] Apply stress labeling: savings < 1 month expenses OR 3+ months negative cash flow
- [ ] Save synthetic dataset as CSV

### Feature Engineering
- [ ] Compute debt-to-income ratio
- [ ] Compute liquidity metrics (savings-to-expenses)
- [ ] Compute cash flow features (volatility, consecutive negative months)
- [ ] Feature functions are pure (no side effects)

### ML Model
- [ ] Define 3-layer MLP: input → 128 → 64 → output (Sigmoid)
- [ ] Implement PyTorch Dataset and DataLoader
- [ ] Training loop with train/val/test split (70/15/15)
- [ ] Evaluate with metrics: Recall, F1, Precision, Accuracy, ROC-AUC
- [ ] Save trained model to models/latest_model.pth
- [ ] Save metrics to models/metrics.json

### API
- [ ] FastAPI endpoint: POST /predict (accepts financial data, returns risk assessment)
- [ ] Pydantic schemas for request/response validation
- [ ] Routes are thin (delegate to ml/predictor.py)

### Orchestration
- [ ] ml/predictor.py handles: load model, compute features, run inference, return results
- [ ] No service layer abstractions

### Frontend
- [ ] Simple HTML dashboard showing risk score, probability, classification
- [ ] Vanilla JavaScript (fetch API, DOM manipulation)
- [ ] CSS styling for clean aesthetic
- [ ] No frameworks, no build tools

### Deployment
- [ ] Single Dockerfile for backend + frontend
- [ ] Environment variables via .env
- [ ] Reproducible local execution

## Out of Scope (v1)

- Database or prediction history storage
- User authentication
- Model retraining on user data
- Mobile responsiveness
- Real-time updates
- Recommendation engine (unless explicitly added later)

---

*Defined: 2026-02-13*
