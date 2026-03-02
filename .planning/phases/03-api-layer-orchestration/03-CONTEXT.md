# Phase 3: API Layer & Orchestration - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Create FastAPI backend with POST /predict endpoint and GET /health endpoint. The API accepts monthly financial time series data, runs it through the feature engineering pipeline and trained model, and returns a risk assessment with score, category, probability, insights, and computed features. Frontend consumption and deployment are separate phases.

</domain>

<decisions>
## Implementation Decisions

### Request/Response design
- Input format: monthly time series (not summary stats)
- Accepts 6-12 months of data (flexible length, not fixed at 12)
- Each monthly entry includes: income (required), expenses (required), debt_payment (optional)
- Credit score submitted once (not per-month)
- When debt_payment is omitted: could mean zero debt or unknown — handle gracefully (default to 0, note it in response)
- Response includes: risk_score (0-100), risk_category (low/medium/high), probability (0-1), insights object, computed feature values

### Prediction insights
- Return both feature contributions AND a plain-language summary
- Show all relevant risk factors (not capped at top-3) — every feature that meaningfully contributes
- Include computed feature values in response (avg_income, debt_ratio, liquidity_ratio, etc.) for frontend display

### Error handling & edge cases
- Missing monthly fields: fill with defaults (zeros/averages) and proceed — don't reject
- Invalid values (negative income, extreme outliers): reject with 422 and simple error message format
- Error format: simple string messages, e.g. {"error": "Invalid input: income must be positive"}
- No structured error objects — keep it straightforward

### Health check endpoint
- GET /health included
- Reports model loaded status, feature count, and training metrics (recall, AUC) from metrics.json

### Claude's Discretion
- Feature contribution computation method (simple thresholds vs model-derived importance — balance educational value with accuracy)
- Model load timing (startup vs lazy-load)
- Behavior when model files are missing (fail-fast vs degraded mode)
- Whether FastAPI serves frontend static files (aligns with single-Dockerfile architecture)
- Default values strategy for missing monthly fields
- How to handle fewer than 12 months of data (padding/interpolation approach)

</decisions>

<specifics>
## Specific Ideas

- Debt payment being optional is a deliberate UX choice — real users may not know their exact debt, and zero debt is a valid scenario
- The 6-12 month flexibility means the API needs to normalize variable-length input before feature engineering
- Plain-language summaries should read like financial advice, not technical output

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-api-layer-orchestration*
*Context gathered: 2026-03-02*
