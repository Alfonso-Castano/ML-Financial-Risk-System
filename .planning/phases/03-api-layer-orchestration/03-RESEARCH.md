# Phase 3: API Layer & Orchestration - Research

**Researched:** 2026-03-02
**Domain:** FastAPI + Pydantic v2 + PyTorch inference orchestration
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

#### Request/Response design
- Input format: monthly time series (not summary stats)
- Accepts 6-12 months of data (flexible length, not fixed at 12)
- Each monthly entry includes: income (required), expenses (required), debt_payment (optional)
- Credit score submitted once (not per-month)
- When debt_payment is omitted: could mean zero debt or unknown — handle gracefully (default to 0, note it in response)
- Response includes: risk_score (0-100), risk_category (low/medium/high), probability (0-1), insights object, computed feature values

#### Prediction insights
- Return both feature contributions AND a plain-language summary
- Show all relevant risk factors (not capped at top-3) — every feature that meaningfully contributes
- Include computed feature values in response (avg_income, debt_ratio, liquidity_ratio, etc.) for frontend display

#### Error handling & edge cases
- Missing monthly fields: fill with defaults (zeros/averages) and proceed — don't reject
- Invalid values (negative income, extreme outliers): reject with 422 and simple error message format
- Error format: simple string messages, e.g. `{"error": "Invalid input: income must be positive"}`
- No structured error objects — keep it straightforward

#### Health check endpoint
- GET /health included
- Reports model loaded status, feature count, and training metrics (recall, AUC) from metrics.json

### Claude's Discretion
- Feature contribution computation method (simple thresholds vs model-derived importance — balance educational value with accuracy)
- Model load timing (startup vs lazy-load)
- Behavior when model files are missing (fail-fast vs degraded mode)
- Whether FastAPI serves frontend static files (aligns with single-Dockerfile architecture)
- Default values strategy for missing monthly fields
- How to handle fewer than 12 months of data (padding/interpolation approach)

### Deferred Ideas (OUT OF SCOPE)
None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 3 bridges the trained PyTorch model to the world. The core task is building four files: `main.py` (FastAPI app), `api/routes.py` (thin route handlers), `api/schemas.py` (Pydantic request/response shapes), and `ml/predictor.py` (orchestration logic). The stack — FastAPI 0.128.0, Pydantic 2.12.3, uvicorn 0.40.0 — is already installed in the project's Python environment. No new packages need to be added to `requirements.txt` beyond `fastapi` and `uvicorn`, and both are already present system-wide.

The central design challenge is the input transformation. The model was trained on 12-month aggregated profiles, but the API accepts 6-12 months of variable-length time-series data. The predictor must normalize this: `engineer_features()` in `feature_engineering.py` already handles the aggregation correctly (it sorts by month, takes averages, computes the final row's savings). Since the function accepts any number of months, passing 6-11 months works as-is — averages over fewer months, last row's savings still valid. The predictor then applies the z-score normalization using the saved `scaler_stats.json` before running model inference.

The feature contributions (insights) are the most design-intensive part. Since the model is an MLP without SHAP or LIME, the recommended approach is domain-threshold analysis — compare each computed feature value against known risk thresholds and derive a plain-language sentence. This is more educational than gradient-based attribution and requires no additional libraries. It is the right choice for this learning project.

**Primary recommendation:** Load model and scaler at startup via FastAPI's lifespan context manager, store in `app.state`, route handlers delegate entirely to `predictor.py`, use Pydantic v2 field validators for input validation (reject negatives with 422, fill missing fields with 0), compute insights via threshold comparison on computed feature values.

---

## What Already Exists (Do Not Rebuild)

This is critical to avoid re-implementing Phase 2 work.

| Artifact | Location | What It Does | How Predictor Uses It |
|----------|----------|--------------|----------------------|
| `FinancialRiskModel` | `backend/ml/model.py` | MLP class (9→128→64→1) | Instantiate, load weights, call in eval mode |
| `engineer_features()` | `backend/data/feature_engineering.py` | One profile's rows → 9-feature dict | Pass constructed DataFrame, get feature dict back |
| `FEATURE_NAMES` | `backend/data/feature_engineering.py` | Canonical feature order list | Use to assemble scaled tensor in correct order |
| `models/latest_model.pth` | `models/` | Trained weights | `torch.load()` + `model.load_state_dict()` |
| `models/scaler_stats.json` | `models/` | mean/scale arrays for 9 features | Z-score normalization: `(X - mean) / scale` |
| `models/metrics.json` | `models/` | recall, precision, f1, accuracy, roc_auc, threshold | Returned by GET /health |
| `backend/config/settings.py` | `backend/config/` | MODEL_PATH, SCALER_PATH, METRICS_PATH, INPUT_SIZE | Import in predictor for path constants |

**Exact scaling formula from train.py (verified):**
```python
X_scaled = (X - mean) / scale  # same as StandardScaler.transform()
```
Where `mean` and `scale` are loaded from `scaler_stats.json` as numpy arrays.

**Model loading pattern from evaluate.py (verified working):**
```python
model = FinancialRiskModel(input_size=settings.INPUT_SIZE)
model.load_state_dict(torch.load(settings.MODEL_PATH, weights_only=True))
model.eval()
```

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| FastAPI | 0.128.0 (installed) | Web framework, route definitions, dependency injection | Modern Python API framework; built-in OpenAPI docs, async support, Pydantic integration |
| Pydantic | 2.12.3 (installed) | Request/response schemas, input validation | FastAPI's native validation layer; automatic 422 on schema violations |
| uvicorn | 0.40.0 (installed) | ASGI server to run FastAPI | Standard uvicorn is the canonical FastAPI server |
| PyTorch | Installed in Phase 2 | Model inference | Already used; `model.eval()` + `torch.no_grad()` for inference |
| NumPy | Installed in Phase 2 | Scaling arithmetic, tensor prep | `np.array()` for feature vector; z-score math |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `contextlib.asynccontextmanager` | stdlib | Lifespan context manager | Required for FastAPI lifespan pattern |
| `fastapi.staticfiles.StaticFiles` | bundled with FastAPI | Serve frontend static files | Mount once in main.py for single-container architecture |
| `fastapi.middleware.cors.CORSMiddleware` | bundled with FastAPI | CORS headers for frontend dev | If frontend served from different origin during development |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Threshold-based insights | SHAP or LIME | SHAP/LIME add dependencies, complexity, and latency. Threshold analysis is simpler, educational, and accurate enough for this domain. |
| Startup lifespan loading | Lazy load on first request | Lazy loading delays first request and complicates error handling. Startup is cleaner and fails fast if files are missing. |
| `torch.no_grad()` | `torch.inference_mode()` | `inference_mode` is marginally faster but behaves identically for this use case. `no_grad()` is more familiar for a learning project. |

**Installation:** No new packages needed. All packages are already installed. Add to `requirements.txt`:
```
fastapi>=0.100
uvicorn>=0.20
```

---

## Architecture Patterns

### File Responsibilities

```
backend/
├── main.py              # FastAPI app creation, lifespan, middleware, router include, static mount
├── api/
│   ├── __init__.py      # (exists)
│   ├── routes.py        # Thin route handlers — validate input, call predictor, return response
│   └── schemas.py       # Pydantic models for request body and response shape
└── ml/
    ├── __init__.py      # (exists)
    ├── model.py         # (exists — do not touch)
    ├── predictor.py     # ALL orchestration: load model, scale features, run inference, build response
    ├── train.py         # (exists — do not touch)
    └── evaluate.py      # (exists — do not touch)
```

**The single rule:** Routes are thin. All business logic lives in `predictor.py`. Routes do: receive request → call predictor → return response.

### Pattern 1: FastAPI Lifespan for Model Loading

**What:** Load the model, scaler, and metrics once at app startup. Store in `app.state`. No global variables.

**When to use:** Any resource that is expensive to load and shared across requests.

**Why startup over lazy-load:** Fail fast — if `models/latest_model.pth` is missing, the app refuses to start rather than silently failing on the first prediction request.

```python
# Source: https://fastapi.tiangolo.com/advanced/events/
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load everything once
    app.state.predictor = Predictor()          # loads model, scaler, metrics
    yield
    # Shutdown: nothing to clean up for file-backed model

app = FastAPI(lifespan=lifespan)

# Access in route handler:
@app.post("/predict")
async def predict(request: Request, body: PredictRequest):
    return request.app.state.predictor.predict(body)
```

**Fail-fast behavior for missing model files:** Raise `RuntimeError` inside `Predictor.__init__` if `models/latest_model.pth` or `models/scaler_stats.json` don't exist. The app will crash on startup with a clear error message rather than serving degraded responses.

### Pattern 2: Pydantic v2 Schema for Time-Series Input

**What:** Nested model with a list of monthly entries. Credit score at top level. All validation in the schema.

```python
# Source: https://fastapi.tiangolo.com/tutorial/body-nested-models/
# Source: https://docs.pydantic.dev/latest/concepts/validators/
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

class MonthlyEntry(BaseModel):
    income: float
    expenses: float
    debt_payment: float = 0.0        # Optional — defaults to 0

    @field_validator('income', 'expenses')
    @classmethod
    def must_be_positive(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

class PredictRequest(BaseModel):
    months: list[MonthlyEntry]
    credit_score: float

    @model_validator(mode='after')
    def validate_month_count(self) -> 'PredictRequest':
        if not (6 <= len(self.months) <= 12):
            raise ValueError("months must contain between 6 and 12 entries")
        return self

class ComputedFeatures(BaseModel):
    avg_income: float
    avg_expenses: float
    final_savings: float
    debt_payment: float
    credit_score: float
    debt_ratio: float
    liquidity_ratio: float
    net_cash_flow: float
    consec_negative_months: float

class InsightsObject(BaseModel):
    risk_factors: list[str]     # plain-language sentences
    summary: str                # one paragraph financial advice tone

class PredictResponse(BaseModel):
    risk_score: float           # 0-100 (probability * 100)
    risk_category: str          # "low" | "medium" | "high"
    probability: float          # raw model output 0-1
    insights: InsightsObject
    computed_features: ComputedFeatures
    debt_payment_defaulted: bool  # True if any month used default 0
```

### Pattern 3: Thin Route Handler

```python
# backend/api/routes.py
from fastapi import APIRouter, Request
from .schemas import PredictRequest, PredictResponse, HealthResponse

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    return request.app.state.predictor.predict(body)

@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    return request.app.state.predictor.health()
```

### Pattern 4: Custom 422 Error Format

The user decided error format is `{"error": "message"}` — simple strings, not structured Pydantic error objects.

```python
# Source: https://fastapi.tiangolo.com/tutorial/handling-errors/
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    # Flatten Pydantic's detailed error list into a single readable string
    messages = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        messages.append(f"{field}: {error['msg']}")
    return JSONResponse(
        status_code=422,
        content={"error": "; ".join(messages)}
    )
```

### Pattern 5: Predictor Orchestration Flow

This is the heart of `ml/predictor.py`. The flow for a single prediction:

```
PredictRequest
    ↓
1. Build DataFrame from months + credit_score
    ↓
2. engineer_features(profile_df) → feature dict  [calls existing feature_engineering.py]
    ↓
3. Assemble feature vector in FEATURE_NAMES order → np.array shape (9,)
    ↓
4. Z-score scale: (X - mean) / scale  [using loaded scaler_stats]
    ↓
5. torch.tensor → model(tensor) → probability float
    ↓
6. Derive risk_score (prob * 100), risk_category (thresholds)
    ↓
7. Compute insights via threshold analysis on raw feature values
    ↓
8. Return PredictResponse
```

### Pattern 6: Serving Static Files (Claude's Discretion — Recommended YES)

FastAPI can serve the vanilla JS frontend directly, which supports the single-Dockerfile architecture and eliminates CORS entirely.

```python
# Source: https://fastapi.tiangolo.com/tutorial/static-files/
from fastapi.staticfiles import StaticFiles

# Mount AFTER API routes so API paths are not shadowed
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

`html=True` makes the StaticFiles mount serve `index.html` for the root path and unknown paths — exactly what a single-page app needs.

### Anti-Patterns to Avoid

- **Global model variable:** Don't do `model = load_model()` at module level in predictor.py. This runs on import — breaks tests, runs twice in some servers. Use `app.state` via lifespan instead.
- **Calling engineer_features with a single-row dict:** The function expects a DataFrame with columns matching the training CSV (`month`, `income`, `total_expenses`, `savings`, `debt_payment`, `credit_score`). The predictor must construct a proper DataFrame from the request.
- **Fat routes:** No business logic in `routes.py`. No torch imports in `routes.py`. The route just calls `predictor.predict()`.
- **Forgetting `model.eval()`:** The model has Dropout layers. Without `model.eval()`, dropout remains active during inference and produces non-deterministic, lower-quality predictions.
- **Forgetting `weights_only=True`** in `torch.load()`: Already established as correct practice in evaluate.py. Maintains consistency and security.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Request body validation | Custom validation logic | Pydantic `@field_validator` | Automatic 422 generation, editor support, schema docs |
| Nested JSON schema | Manual dict parsing | Pydantic `BaseModel` with nested models | Type safety, automatic coercion |
| ASGI server | Custom HTTP server | uvicorn | Handles async correctly, production-tested |
| Startup/shutdown lifecycle | `@app.on_event` (deprecated) | `@asynccontextmanager` lifespan | Modern FastAPI pattern, cleaner resource management |
| JSON serialization | `json.dumps` manually | FastAPI response_model | Handles Pydantic model serialization automatically |

**Key insight:** Pydantic v2 + FastAPI eliminate almost all manual validation and serialization code. The predictor should focus entirely on ML orchestration, not JSON handling.

---

## Common Pitfalls

### Pitfall 1: engineer_features Expects DataFrame Columns from Training CSV

**What goes wrong:** The API receives `income`, `expenses`, `debt_payment` per month. But `engineer_features()` expects a DataFrame with column names from the synthetic training data: `month`, `income`, `total_expenses`, `savings`, `debt_payment`, `credit_score`.

**Why it happens:** The training CSV has `total_expenses` (sum of essentials + discretionary + debt_payment). The API receives `expenses` directly. The column names must match exactly what `engineer_features()` reads.

**How to avoid:** In the predictor, when constructing the DataFrame from the request, map API fields to training column names:

```python
# API field → training column name mapping
rows = []
cumulative_savings = 0.0
for i, month in enumerate(request.months):
    net = month.income - month.expenses - month.debt_payment
    cumulative_savings = max(0, cumulative_savings + net)
    rows.append({
        'month': i + 1,
        'income': month.income,
        'total_expenses': month.expenses,  # NOTE: map expenses → total_expenses
        'savings': cumulative_savings,      # NOTE: must compute running savings
        'debt_payment': month.debt_payment,
        'credit_score': request.credit_score,
    })
profile_df = pd.DataFrame(rows)
```

**Critical detail:** `final_savings` in `engineer_features()` reads `group.iloc[-1]['savings']` — the last row's savings column. The API request doesn't include savings history, so the predictor must compute a running cumulative savings. Use `max(0, cumulative_savings + monthly_net)` to match the synthetic generator's floor-at-zero behavior.

**Warning signs:** `KeyError: 'total_expenses'` or `KeyError: 'savings'` from engineer_features call.

### Pitfall 2: Feature Vector Order Must Match FEATURE_NAMES

**What goes wrong:** `engineer_features()` returns a dict. Dicts preserve insertion order in Python 3.7+, but the dict is assembled in engineer_features using literal order. The scaler's mean/scale arrays were fit in `FEATURE_NAMES` canonical order. If the tensor is assembled in a different order, predictions are meaningless.

**Why it happens:** Easy to accidentally use `list(features.values())` which may differ from scaler order if engineer_features internals change.

**How to avoid:** Always assemble the feature vector explicitly:
```python
from backend.data.feature_engineering import FEATURE_NAMES
feature_vector = np.array([features[name] for name in FEATURE_NAMES], dtype=np.float32)
```

**Warning signs:** Model outputs consistently high or low probabilities regardless of input.

### Pitfall 3: Model Called in Training Mode

**What goes wrong:** Predictions are non-deterministic because Dropout (p=0.3) randomly zeros activations.

**Why it happens:** `model.eval()` must be called after loading weights. If the predictor calls the model without `model.eval()`, the dropout layers remain active.

**How to avoid:** Call `model.eval()` once right after `model.load_state_dict()` in the Predictor constructor. Do not call `model.train()` anywhere in predictor.py.

**Warning signs:** Same input produces different probability values on repeated calls.

### Pitfall 4: torch.no_grad() Missing During Inference

**What goes wrong:** Memory grows with each request as PyTorch builds a computation graph for every forward pass.

**Why it happens:** Gradient tracking is enabled by default. It must be explicitly disabled for inference.

**How to avoid:**
```python
with torch.no_grad():
    tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
    prob = model(tensor).item()
```

**Warning signs:** Memory usage grows over time; evaluation already uses this pattern correctly (see evaluate.py).

### Pitfall 5: Negative `final_savings` in Variable-Length Inputs

**What goes wrong:** The model was trained on profiles where savings have a floor of 0 (the synthetic generator clamps at 0). If the API computes running savings without clamping, `final_savings` can go negative, which the scaler was not trained on. This produces out-of-distribution inputs.

**How to avoid:** Apply `max(0, running_savings)` when computing cumulative savings in the DataFrame construction step (shown in Pitfall 1 code above).

### Pitfall 6: Settings Paths Are Relative, Not Absolute

**What goes wrong:** `settings.MODEL_PATH = "models/latest_model.pth"` — this relative path resolves from wherever `uvicorn` is launched, not from the file's location.

**Why it happens:** The project is designed to run from the project root (`uvicorn backend.main:app`), and all paths assume the CWD is the project root. This is correct behavior — document it, don't change it.

**How to avoid:** Document that the server must be started from the project root. Don't convert to `Path(__file__).parent` paths — this would break the established convention from Phase 2.

---

## Code Examples

All examples are derived from verified sources (official FastAPI docs, official Pydantic docs) combined with the verified existing codebase patterns from Phase 2.

### Complete main.py Pattern

```python
# Source: https://fastapi.tiangolo.com/advanced/events/ + https://fastapi.tiangolo.com/tutorial/static-files/
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.api.routes import router
from backend.ml.predictor import Predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor()   # loads model, scaler, metrics — fails fast if missing
    yield

app = FastAPI(title="Financial Risk API", lifespan=lifespan)

# Custom 422 error format ({"error": "message"})
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    messages = [f"{' -> '.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "; ".join(messages)})

# API routes
app.include_router(router)

# Serve frontend static files last (catches all remaining paths)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
```

### Predictor Class Skeleton

```python
# backend/ml/predictor.py
import json
import numpy as np
import pandas as pd
import torch

from backend.config import settings
from backend.ml.model import FinancialRiskModel
from backend.data.feature_engineering import engineer_features, FEATURE_NAMES

class Predictor:
    def __init__(self):
        # Load model
        self.model = FinancialRiskModel(input_size=settings.INPUT_SIZE)
        self.model.load_state_dict(torch.load(settings.MODEL_PATH, weights_only=True))
        self.model.eval()

        # Load scaler stats
        with open(settings.SCALER_PATH) as f:
            stats = json.load(f)
        self.mean = np.array(stats["mean"], dtype=np.float32)
        self.scale = np.array(stats["scale"], dtype=np.float32)

        # Load training metrics for /health
        with open(settings.METRICS_PATH) as f:
            self.metrics = json.load(f)

    def predict(self, request) -> dict:
        # 1. Build DataFrame
        profile_df = self._build_dataframe(request)
        # 2. Engineer features (reuse existing function)
        features = engineer_features(profile_df)
        # 3. Scale
        feature_vector = np.array([features[n] for n in FEATURE_NAMES], dtype=np.float32)
        scaled = (feature_vector - self.mean) / self.scale
        # 4. Inference
        with torch.no_grad():
            tensor = torch.tensor(scaled).unsqueeze(0)
            prob = float(self.model(tensor).item())
        # 5. Derive outputs
        risk_score = round(prob * 100, 1)
        risk_category = "high" if prob >= 0.65 else ("medium" if prob >= 0.35 else "low")
        # 6. Insights
        insights = self._compute_insights(features, prob)
        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "probability": round(prob, 4),
            "insights": insights,
            "computed_features": features,
            "debt_payment_defaulted": any(m.debt_payment == 0.0 for m in request.months),
        }
```

### Threshold-Based Insight Logic (Recommended Approach)

**Why threshold analysis over model-derived importance:** The MLP has no built-in feature importance. Gradient-based attribution (integrated gradients) requires additional libraries and is harder to explain. Threshold analysis maps directly to the stress definition users understand: savings below 1 month of expenses, 3+ consecutive negative months. This is the educational choice.

```python
def _compute_insights(self, features: dict, prob: float) -> dict:
    risk_factors = []

    # Liquidity: savings < 1 month of expenses is a stress condition
    if features['liquidity_ratio'] < 1.0:
        months = round(features['liquidity_ratio'], 1)
        risk_factors.append(
            f"Low emergency fund: savings cover only {months} months of expenses "
            f"(target: 1+ months)"
        )

    # Consecutive negative months: 3+ is a stress condition
    if features['consec_negative_months'] >= 3:
        risk_factors.append(
            f"Sustained cash flow deficit: {int(features['consec_negative_months'])} "
            f"consecutive months of negative cash flow"
        )

    # Debt ratio: consuming most of free cash flow
    if features['debt_ratio'] > 0.5:
        pct = round(features['debt_ratio'] * 100)
        risk_factors.append(
            f"High debt burden: debt payments consume {pct}% of available cash flow "
            f"after expenses"
        )

    # Net cash flow: negative average means structural deficit
    if features['net_cash_flow'] < 0:
        risk_factors.append(
            f"Negative average cash flow: spending exceeds income by "
            f"${abs(round(features['net_cash_flow']))/month on average"
        )

    # Credit score: below 640 is subprime
    if features['credit_score'] < 640:
        risk_factors.append(
            f"Credit score of {int(features['credit_score'])} indicates elevated credit risk"
        )

    # Summary paragraph
    if not risk_factors:
        summary = (
            "Your finances appear healthy. Maintain your savings buffer and continue "
            "managing expenses within income."
        )
    elif prob >= 0.65:
        summary = (
            f"Multiple risk factors indicate significant financial stress. "
            f"Priority actions: {'and '.join(r.split(':')[0].lower() for r in risk_factors[:2])}."
        )
    else:
        summary = (
            f"Some financial risk indicators present. Monitor your cash flow closely "
            f"and build your emergency fund."
        )

    return {"risk_factors": risk_factors, "summary": summary}
```

### Risk Category Thresholds (Claude's Discretion — Recommended)

| Probability | Category | Rationale |
|------------|----------|-----------|
| >= 0.65 | high | Model is confident — above training threshold of 0.5, with margin |
| 0.35 - 0.65 | medium | Model is uncertain — warrants monitoring |
| < 0.35 | low | Model is confident of no stress |

These thresholds are a recommendation. The model's decision threshold is 0.5 (from metrics.json). The risk_category uses wider bands to communicate uncertainty meaningfully.

---

## Discretion Decisions (Recommended)

These are the "Claude's Discretion" items from CONTEXT.md. Recommendations based on research:

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Feature contribution method | Domain threshold analysis (described above) | No extra libraries, educational, maps to stress definition |
| Model load timing | Startup via lifespan | Fail fast, single load, no per-request overhead |
| Missing model files | Fail-fast (crash on startup) | Better UX than silent failures at prediction time |
| Missing monthly fields | Default to 0 (already encoded in Pydantic schema) | Pydantic `= 0.0` default handles this automatically |
| Frontend static files | YES, mount via StaticFiles | Supports single-Dockerfile architecture, eliminates CORS |
| Fewer than 12 months | Pass as-is to engineer_features | engineer_features already handles N rows correctly; averages over fewer months remain valid |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|-----------------|--------------|--------|
| `@app.on_event("startup")` | `@asynccontextmanager` lifespan | FastAPI 0.95.0 (2023) | Old decorators are deprecated; lifespan is the standard |
| Pydantic v1 `@validator` | Pydantic v2 `@field_validator` + `@model_validator` | Pydantic 2.0 (2023) | v1 validators still work via compatibility shim but v2 style is correct for new code |
| `torch.load(path)` | `torch.load(path, weights_only=True)` | PyTorch 2.0+ | Security improvement; suppress FutureWarning; already used in evaluate.py |

**Deprecated/outdated in this codebase:**
- `@app.on_event("startup")` / `@app.on_event("shutdown")`: These work but FastAPI docs mark them deprecated. Use lifespan.
- `Optional[float]` with `from typing import Optional`: Still valid Python, but Python 3.10+ prefers `float | None`. Either works with Pydantic v2.

---

## Open Questions

1. **Savings reconstruction accuracy for fewer than 12 months**
   - What we know: The predictor must compute running savings from income/expenses because the API doesn't ask users for their savings history
   - What's unclear: If a user submits 6 months, the `final_savings` will be the 6-month cumulative savings, but the model was trained on 12-month profiles. The feature value will be in-distribution only if the user's 6-month savings happens to be in a similar range to a 12-month savings balance.
   - Recommendation: Accept this limitation. Document in the response that results are estimates. The model will still produce meaningful predictions because it trained on the relative patterns (liquidity_ratio, debt_ratio), not absolute savings amounts.

2. **Running savings initial value**
   - What we know: The predictor must construct running savings without knowing the user's starting balance
   - What's unclear: Should cumulative savings start at 0? That assumes the user has no savings before the observation period.
   - Recommendation: Start at 0. This makes `final_savings` represent the net savings accumulated during the observation period, which is still a valid relative measure. Document this assumption.

---

## Sources

### Primary (HIGH confidence)
- FastAPI official docs (events/lifespan): https://fastapi.tiangolo.com/advanced/events/ — lifespan pattern, app.state usage
- FastAPI official docs (static files): https://fastapi.tiangolo.com/tutorial/static-files/ — StaticFiles mount pattern
- FastAPI official docs (error handling): https://fastapi.tiangolo.com/tutorial/handling-errors/ — RequestValidationError override
- FastAPI official docs (nested models): https://fastapi.tiangolo.com/tutorial/body-nested-models/ — list[Model] request body
- Pydantic official docs (validators): https://docs.pydantic.dev/latest/concepts/validators/ — field_validator, model_validator
- `pip show fastapi pydantic uvicorn` — version numbers verified from installed packages (FastAPI 0.128.0, Pydantic 2.12.3, uvicorn 0.40.0)
- Verified codebase: `backend/data/feature_engineering.py`, `backend/ml/model.py`, `backend/ml/evaluate.py`, `models/scaler_stats.json`, `models/metrics.json`

### Secondary (MEDIUM confidence)
- PyTorch forums: `model.eval()` vs `torch.no_grad()` — confirmed orthogonal, both required for inference
- FastAPI CORS docs: https://fastapi.tiangolo.com/tutorial/cors/ — CORSMiddleware pattern confirmed

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified from installed packages; APIs verified from official docs
- Architecture patterns: HIGH — patterns verified from official FastAPI and Pydantic docs
- Feature engineering integration: HIGH — patterns verified from existing codebase (evaluate.py, train.py)
- Insight computation: MEDIUM — threshold values are domain judgment calls, not verified against user testing
- Savings reconstruction: MEDIUM — logical approach but untested against real user data patterns

**Research date:** 2026-03-02
**Valid until:** 2026-09-02 (FastAPI and Pydantic are stable; lifespan API introduced 2023 and unlikely to change)
