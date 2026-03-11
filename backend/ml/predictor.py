"""
Predictor: orchestration layer between the API and the trained ML model.

This class is the bridge that connects a raw HTTP request to a financial risk
prediction. It owns three responsibilities:

  1. Loading artifacts from disk (model weights, scaler stats, metrics)
  2. Transforming a PredictRequest into a scaled feature tensor
  3. Running inference and assembling the full response dict

Design decisions:
  - No global variables — everything is encapsulated in the Predictor instance
  - Fail-fast on missing files (RuntimeError in __init__)
  - Loaded once at startup via FastAPI lifespan and stored in app.state
  - Insights computed via domain-threshold analysis (no extra libraries needed)

Architecture context:
  Frontend → routes.py (thin) → predictor.py (orchestration) → model.py (inference)
                                                              → feature_engineering.py
"""

import json
import numpy as np
import pandas as pd
import torch

from backend.config import settings
from backend.ml.model import FinancialRiskModel
from backend.data.feature_engineering import engineer_features, FEATURE_NAMES


class Predictor:
    """
    Orchestrates the full inference pipeline for financial risk assessment.

    Instantiate once per application lifetime (via FastAPI lifespan).
    Thread-safe for concurrent reads because all state is set once in __init__
    and model.eval() disables Dropout permanently.
    """

    def __init__(self) -> None:
        """
        Load model weights, scaler stats, and training metrics from disk.

        Raises:
            RuntimeError: If any required model file is missing. This causes
                          FastAPI to refuse startup rather than serve bad predictions.
        """
        # --- Load PyTorch model ---
        try:
            self.model = FinancialRiskModel(input_size=settings.INPUT_SIZE)
            state_dict = torch.load(settings.MODEL_PATH, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()  # Disable Dropout for deterministic inference
        except FileNotFoundError:
            raise RuntimeError(
                f"Model weights not found at '{settings.MODEL_PATH}'. "
                "Run backend/ml/train.py first to generate the model file."
            )

        # --- Load scaler statistics (mean and scale for z-score normalization) ---
        try:
            with open(settings.SCALER_PATH) as f:
                stats = json.load(f)
            self.mean = np.array(stats["mean"], dtype=np.float32)
            self.scale = np.array(stats["scale"], dtype=np.float32)
        except FileNotFoundError:
            raise RuntimeError(
                f"Scaler stats not found at '{settings.SCALER_PATH}'. "
                "Run backend/ml/train.py first to generate the scaler file."
            )

        # --- Load training metrics for /health endpoint ---
        try:
            with open(settings.METRICS_PATH) as f:
                self.metrics = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(
                f"Training metrics not found at '{settings.METRICS_PATH}'. "
                "Run backend/ml/evaluate.py first to generate metrics."
            )

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def predict(self, request) -> dict:
        """
        Run the full inference pipeline for a single prediction request.

        Args:
            request: PredictRequest — validated by Pydantic before this is called.

        Returns:
            dict matching PredictResponse schema fields:
              risk_score, risk_category, probability, insights,
              computed_features, debt_payment_defaulted
        """
        # Step 1: Build a DataFrame in the format engineer_features() expects
        profile_df = self._build_dataframe(request)

        # Step 2: Engineer the 9 features from the monthly data
        features = engineer_features(profile_df)

        # Step 3: Assemble feature vector in canonical FEATURE_NAMES order
        # IMPORTANT: must use FEATURE_NAMES order — the scaler was fit in this order
        feature_vector = np.array(
            [features[name] for name in FEATURE_NAMES],
            dtype=np.float32
        )

        # Step 4: Z-score normalization — same formula used in train.py
        scaled = (feature_vector - self.mean) / self.scale

        # Step 5: Model inference — no_grad() prevents memory leaks from grad tracking
        with torch.no_grad():
            tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            prob = float(self.model(tensor).item())

        # Step 6: Derive human-readable outputs from raw probability
        risk_score = round(prob * 100, 1)
        risk_category = (
            "high" if prob >= 0.65 else
            "medium" if prob >= 0.35 else
            "low"
        )

        # Step 7: Generate plain-language insights via threshold analysis
        insights = self._compute_insights(features, prob)

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "probability": round(prob, 4),
            "insights": insights,
            "computed_features": features,
        }

    def health(self) -> dict:
        """
        Return model status and training metrics for GET /health.

        Returns:
            dict matching HealthResponse schema fields.
        """
        return {
            "status": "healthy",
            "model_loaded": True,
            "feature_count": len(FEATURE_NAMES),
            "metrics": {
                "recall": self.metrics["recall"],
                "roc_auc": self.metrics["roc_auc"],
            },
        }

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_dataframe(self, request) -> pd.DataFrame:
        """
        Convert a PredictRequest into a DataFrame that engineer_features() accepts.

        Column mapping (API field -> training CSV column name):
          month.income       -> income          (same name)
          month.expenses     -> total_expenses  (CRITICAL: different name from API)
          computed running   -> savings         (cumulative, floored at 0)
          request.credit_score -> credit_score  (broadcast to every row)
          i + 1              -> month           (1-indexed chronological index)

        Users report total expenses including debt payments. This matches training
        data where total_expenses = essentials + discretionary + debt_payment.

        Monthly variance injection:
          Training data applies MONTHLY_VARIANCE (25%) noise to essentials (60% of
          expenses) and discretionary (20%) independently, while debt_payment (20%)
          stays fixed. This produces expense_volatility ~0.15 (CV of total_expenses).
          API users typically send constant values, producing expense_volatility=0 —
          a value the model has never seen (z-score = -4.55). To close this train/serve
          skew, we inject calibrated noise (15% variance) that reproduces the training
          distribution's effective CV. A fixed seed derived from the input ensures
          deterministic predictions for the same request.

        The running savings is computed month-by-month and clamped at 0 to match
        the synthetic data generator's behavior (savings never go below zero).
        Starting balance is assumed to be 0 — this represents net savings
        accumulated during the observation period.

        Args:
            request: PredictRequest with validated months and credit_score.

        Returns:
            pd.DataFrame with one row per month and the columns engineer_features expects.
        """
        # Fixed seed from input for deterministic predictions
        seed_value = int(abs(hash((
            tuple((m.income, m.expenses) for m in request.months),
            request.credit_score
        )))) % (2**31)
        rng = np.random.default_rng(seed_value)

        rows = []
        cumulative_savings = 0.0

        for i, month in enumerate(request.months):
            # Inject calibrated monthly variance matching training data's effective CV.
            # Training uses 25% noise on essentials (60%) and discretionary (20%)
            # independently, keeping debt (20%) fixed — effective CV on total ~0.15.
            # Using 0.15 variance here reproduces that distribution.
            inference_variance = 0.15
            income = month.income * rng.normal(1.0, inference_variance)
            expenses = month.expenses * rng.normal(1.0, inference_variance)

            # Expenses already include debt — no separate subtraction
            monthly_net = income - expenses
            cumulative_savings = max(0.0, cumulative_savings + monthly_net)

            rows.append({
                'month': i + 1,
                'income': round(income, 2),
                'total_expenses': round(expenses, 2),
                'savings': round(cumulative_savings, 2),
                'credit_score': request.credit_score,
            })

        return pd.DataFrame(rows)

    def _compute_insights(self, features: dict, prob: float) -> dict:
        """
        Generate plain-language risk factors and a summary paragraph.

        Uses domain-threshold analysis — each risk condition has a known threshold
        that directly maps to the stress definition used during training. This
        approach is educational and requires no additional libraries.

        All triggered conditions are included (not capped at a top-N list).

        Args:
            features: Dict with 9 engineered feature values (from engineer_features).
            prob: Raw model output probability in [0, 1].

        Returns:
            dict with 'risk_factors' (list[str]) and 'summary' (str).
        """
        risk_factors = []

        # --- Expense volatility: highly erratic spending is a risk signal ---
        if features['expense_volatility'] > 0.15:
            pct = round(features['expense_volatility'] * 100, 1)
            risk_factors.append(
                f"Erratic spending pattern: monthly expenses vary by {pct}% relative to your average "
                f"(high volatility increases financial instability risk)"
            )

        # --- Savings trend: declining savings trajectory signals deterioration ---
        if features['savings_trend'] < -100:
            monthly_decline = abs(round(features['savings_trend']))
            risk_factors.append(
                f"Declining savings trajectory: savings decreasing by approximately "
                f"${monthly_decline}/month on average"
            )

        # --- Expense ratio: spending at or above income level ---
        if features['expense_ratio'] > 0.90:
            pct = round(features['expense_ratio'] * 100)
            risk_factors.append(
                f"High spending pressure: expenses consume {pct}% of income "
                f"(values near or above 100% indicate financial strain)"
            )

        # --- Savings buffer: less than 1 month of expenses saved ---
        if features['savings_months'] < 1.0:
            months_str = round(features['savings_months'], 1)
            risk_factors.append(
                f"Thin savings buffer: savings cover only {months_str} months of expenses "
                f"(below the 1-month minimum recommended buffer)"
            )

        # --- Net cash flow: structural deficit when average is negative ---
        if features['net_cash_flow'] < 0:
            monthly_deficit = abs(round(features['net_cash_flow']))
            risk_factors.append(
                f"Negative average cash flow: spending exceeds income by "
                f"${monthly_deficit}/month on average"
            )

        # --- Credit score: below 640 is considered subprime ---
        if features['credit_score'] < 640:
            score = int(features['credit_score'])
            risk_factors.append(
                f"Credit score of {score} indicates elevated credit risk"
            )

        # --- Summary paragraph (tone: financial advisor, not technical) ---
        if not risk_factors:
            summary = (
                "Your finances appear healthy. Maintain your savings buffer and continue "
                "managing expenses within income."
            )
        elif prob >= 0.65:
            # High risk: name the top two factors as priority actions
            top_labels = [f.split(':')[0].lower() for f in risk_factors[:2]]
            priority_str = " and ".join(top_labels)
            summary = (
                f"Multiple risk factors indicate significant financial stress. "
                f"Priority actions: address {priority_str} to reduce financial vulnerability."
            )
        else:
            # Medium risk: monitoring advice
            summary = (
                "Some financial risk indicators are present. Monitor your cash flow closely "
                "and build your emergency fund to reduce financial vulnerability."
            )

        return {
            "risk_factors": risk_factors,
            "summary": summary,
        }
