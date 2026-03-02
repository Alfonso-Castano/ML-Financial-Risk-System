"""
FastAPI application entry point for the Financial Risk API.

Responsibilities:
  1. Lifespan: load Predictor once at startup, store in app.state
  2. Custom 422 handler: return {"error": "message"} instead of Pydantic's verbose format
  3. Router: register /predict and /health from routes.py
  4. Static files: serve frontend/ at "/" for single-Dockerfile architecture

Run from project root:
    uvicorn backend.main:app

Note: Must be run from project root so settings.py paths (models/, data/) resolve correctly.
"""

import contextlib

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.routes import router
from backend.ml.predictor import Predictor


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the Predictor on startup and make it available via app.state.predictor.

    Predictor.__init__ loads model weights, scaler stats, and training metrics.
    If any file is missing, it raises RuntimeError and the server refuses to start.
    This is fail-fast behavior — better to crash on startup than to serve bad predictions.
    """
    app.state.predictor = Predictor()
    yield
    # No cleanup needed — model is in-memory, garbage collected on shutdown


# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial Risk API",
    description="ML-powered financial risk assessment from monthly time-series data",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Custom 422 error handler
# ---------------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc: RequestValidationError):
    """
    Override Pydantic's verbose validation error format with a simple string.

    Pydantic's default 422 response nests errors in a complex structure.
    This handler flattens them into: {"error": "field -> subfield: message; ..."}

    This keeps the API contract simple for the frontend and curl users.
    """
    messages = []
    for error in exc.errors():
        # loc is a tuple like ('body', 'months', 0, 'income')
        # Skip 'body' prefix, join the rest with ' -> '
        loc = error.get("loc", ())
        field_path = " -> ".join(str(part) for part in loc if part != "body")
        msg = error.get("msg", "invalid value")
        if field_path:
            messages.append(f"{field_path}: {msg}")
        else:
            messages.append(msg)

    error_string = "; ".join(messages)
    return JSONResponse(
        status_code=422,
        content={"error": error_string},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Include API routes BEFORE mounting static files so /predict and /health
# take precedence over the static file handler at "/"
app.include_router(router)


# ---------------------------------------------------------------------------
# Static files (frontend)
# ---------------------------------------------------------------------------

# Mount the frontend directory at "/" — serves index.html for the single-page app.
# This must come AFTER include_router so API routes take precedence.
# The frontend/ directory will be populated in Phase 4.
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
