"""
API route handlers for the Financial Risk API.

These handlers are intentionally thin — they receive validated request bodies
(Pydantic handles 422 for us), delegate to the Predictor, and return the result.
No business logic, no torch imports, no feature engineering here.

Architecture: routes.py (thin) → predictor.py (orchestration) → model.py (inference)
"""

from fastapi import APIRouter, Request
from backend.api.schemas import PredictRequest, PredictResponse, HealthResponse

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """
    Accept 6-12 months of financial data and return a full risk assessment.

    Delegates entirely to request.app.state.predictor which was loaded at startup.
    Pydantic validates the request body before this function is called.
    """
    result = request.app.state.predictor.predict(body)
    return result


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """
    Return model status and training metrics.

    Quick liveness check that confirms the model was loaded and returns
    the training metrics (recall, roc_auc) for observability.
    """
    result = request.app.state.predictor.health()
    return result
