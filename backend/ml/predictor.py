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

        # Step 8: Flag if any month used the default 0.0 for debt_payment
        # This catches both explicit zero submissions and omitted debt_payment fields
        debt_payment_defaulted = any(m.debt_payment == 0.0 for m in request.months)

        return {
            "risk_score": risk_score,
            "risk_category": risk_category,
            "probability": round(prob, 4),
            "insights": insights,
            "computed_features": features,
            "debt_payment_defaulted": debt_payment_defaulted,
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

        Column mapping (API field → training CSV column name):
          month.income       → income          (same name)
          month.expenses     → total_expenses  (CRITICAL: different name from API)
          computed running   → savings         (cumulative, floored at 0)
          month.debt_payment → debt_payment    (same name)
          request.credit_score → credit_score  (broadcast to every row)
          i + 1              → month           (1-indexed chronological index)

        The running savings is computed month-by-month and clamped at 0 to match
        the synthetic data generator's behavior (savings never go below zero).
        Starting balance is assumed to be 0 — this represents net savings
        accumulated during the observation period.

        Args:
            request: PredictRequest with validated months and credit_score.

        Returns:
            pd.DataFrame with one row per month and the columns engineer_features expects.
        """
        rows = []
        cumulative_savings = 0.0

        for i, month in enumerate(request.months):
            monthly_net = month.income - month.expenses - month.debt_payment
            cumulative_savings = max(0.0, cumulative_savings + monthly_net)

            rows.append({
                'month': i + 1,
                'income': month.income,
                'total_expenses': month.expenses,   # API: expenses → training: total_expenses
                'savings': cumulative_savings,
                'debt_payment': month.debt_payment,
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

        # --- Liquidity: savings < 1 month of expenses is a stress condition ---
        if features['liquidity_ratio'] < 1.0:
            months_covered = round(features['liquidity_ratio'], 1)
            risk_factors.append(
                f"Low emergency fund: savings cover only {months_covered} months of expenses "
                f"(recommended: 1+ months)"
            )

        # --- Consecutive negative cash flow months: 3+ is a stress condition ---
        if features['consec_negative_months'] >= 3:
            streak = int(features['consec_negative_months'])
            risk_factors.append(
                f"Sustained cash flow deficit: {streak} consecutive months of negative cash flow"
            )

        # --- Debt ratio: consuming the majority of free cash flow ---
        if features['debt_ratio'] > 0.5:
            pct = round(features['debt_ratio'] * 100)
            risk_factors.append(
                f"High debt burden: debt payments consume {pct}% of available cash flow "
                f"after expenses"
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
