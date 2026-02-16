# Phase 1: Foundation & Synthetic Data - Context

**Gathered:** 2026-02-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Set up project structure and generate synthetic financial profiles with stress labels for ML training. The generator creates realistic financial data with monthly history, applies stress labeling rules, and outputs a CSV for model training in Phase 2.

</domain>

<decisions>
## Implementation Decisions

### Profile attributes
- Minimal field set (5-6 core financial fields): income, savings (single balance), debt, credit score
- Expenses split into 2-3 categories: essentials (rent/food), discretionary (entertainment), debt payments
- Purely financial data — no demographics (age, employment type, etc.)
- No monthly savings rate — just total savings as a snapshot

### Dataset scale & balance
- Default 3,000 profiles (configurable via `NUM_PROFILES` in settings.py)
- Optional random seed with a default value in settings.py for reproducibility
- Class balance (% stressed vs healthy): Claude's discretion

### Data realism
- Income drawn from realistic salary bands ($30k-$50k, $50k-$80k, $80k-$120k+)
- Light archetypes: 3-4 internal profile types (low-income, middle, high-income) that set base ranges for field generation
- Field correlations: Claude's discretion on how correlated fields should be
- Edge cases near stress threshold: Claude's discretion

### Temporal structure
- 12 months of monthly history per profile
- Monthly values vary ~10-20% around the base (income and expenses fluctuate naturally)
- Generator internally simulates month-by-month data, then the stress rule (savings < 1 month expenses OR 3+ consecutive months negative cash flow) is applied to the full history

### Claude's Discretion
- CSV structure (one row per person-month vs wide format vs derived features)
- Class balance ratio (stressed vs healthy)
- Degree of field correlation between income/expenses/savings
- Whether to include borderline edge cases near the stress threshold
- Exact archetype definitions and their parameter ranges

</decisions>

<specifics>
## Specific Ideas

- Archetypes should feel like recognizable financial situations (e.g., "someone just getting by" vs "comfortable earner" vs "high-income but high-debt")
- Monthly variability makes data feel less synthetic — people don't earn/spend identically each month
- Settings.py should be the single source for generator configuration

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-foundation-synthetic-data*
*Context gathered: 2026-02-16*
