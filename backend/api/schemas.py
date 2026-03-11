"""
Pydantic v2 request/response models for the Financial Risk API.

Defines the API contract for two endpoints:
  - POST /predict  — accepts monthly time-series + credit score, returns risk assessment
  - GET  /health   — returns model status and training metrics

All validation (negative values, month count, credit score range) is enforced here
via Pydantic validators so FastAPI automatically returns 422 on invalid input.
"""

from pydantic import BaseModel, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class MonthlyEntry(BaseModel):
    """One month of financial data submitted by the user.

    Users should report total expenses including debt payments.
    """

    income: float
    expenses: float

    @field_validator('income', 'expenses')
    @classmethod
    def income_expenses_must_be_non_negative(cls, v: float, info) -> float:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v


class PredictRequest(BaseModel):
    """Top-level request body for POST /predict."""

    months: list[MonthlyEntry]
    credit_score: float

    @model_validator(mode='after')
    def validate_month_count(self) -> 'PredictRequest':
        if not (6 <= len(self.months) <= 12):
            raise ValueError(
                f"months must contain between 6 and 12 entries, got {len(self.months)}"
            )
        return self

    @field_validator('credit_score')
    @classmethod
    def credit_score_must_be_in_range(cls, v: float) -> float:
        if not (300 <= v <= 850):
            raise ValueError(
                f"credit_score must be between 300 and 850, got {v}"
            )
        return v


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ComputedFeatures(BaseModel):
    """The 9 engineered features computed from the monthly input data.

    Returned in the response so the frontend can display them alongside
    the risk score. Order matches FEATURE_NAMES from feature_engineering.py.
    """

    avg_income: float
    avg_expenses: float
    final_savings: float
    expense_ratio: float
    credit_score: float
    savings_months: float
    expense_volatility: float
    net_cash_flow: float
    savings_trend: float


class InsightsObject(BaseModel):
    """Plain-language risk analysis returned with each prediction."""

    risk_factors: list[str]  # One sentence per triggered risk condition
    summary: str             # One paragraph, financial-advice tone


class PredictResponse(BaseModel):
    """Full response body for POST /predict."""

    risk_score: float           # 0–100 (probability × 100, rounded to 1 decimal)
    risk_category: str          # "low" | "medium" | "high"
    probability: float          # Raw model output in [0, 1]
    insights: InsightsObject
    computed_features: ComputedFeatures


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str         # Always "healthy" when the model loaded successfully
    model_loaded: bool
    feature_count: int  # Number of engineered features (9)
    metrics: dict       # Subset of metrics.json: recall + roc_auc
