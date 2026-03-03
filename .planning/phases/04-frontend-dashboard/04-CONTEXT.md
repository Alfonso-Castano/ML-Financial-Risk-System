# Phase 4: Frontend Dashboard - Context

**Gathered:** 2026-03-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a vanilla JS dashboard (index.html, app.js, styles.css) that lets users input financial data and see risk predictions from POST /predict. No frameworks, no build tools. Served via FastAPI static files mount.

</domain>

<decisions>
## Implementation Decisions

### Design Workflow
- **frontend-design skill drives design first** — invoke it to establish the design system (fonts, colors, layout, spatial composition) before executor writes functional code
- **frontend-developer agent is dropped** — incompatible with vanilla JS constraint; the skill alone handles design quality
- Design system output feeds into styles.css and index.html structure

### Aesthetic Direction
- **Anthropic/Claude-inspired aesthetic** — mimic Anthropic's visual language: warm earth tones (tan/sand backgrounds), clean sans-serif typography, distinctive orange accents, generous whitespace, understated elegance
- Match the feel of claude.ai and Anthropic's marketing pages
- CSS variables for the color system and typography scale

### Input Form Design
- **Dual input: manual form + CSV upload**
- Manual form uses a **spreadsheet-style grid** — all 12 months visible at once, rows are months, columns are income/expenses/debt_payment
- Additional fields outside the grid: credit_score
- **Empty form with placeholder hints** (e.g., "4500") — no pre-filled data
- CSV upload as alternative input method
- **CSV format: Claude's Discretion** — pick format based on what the API expects

### Page Structure
- **Tabbed sections** — multiple tabs to organize content
- **Tab selection: Claude's Discretion** — pick tabs that showcase the ML pipeline without scope creep (e.g., "Predict" + "About the Model")
- **Side-by-side layout** on the Predict tab — form on left, results on right (desktop). Stacked on mobile
- **Desktop-first, basic mobile** — optimize for desktop, stack vertically on mobile as functional fallback

### Results Display
- **Risk gauge / meter** — semicircular or circular gauge visualization for the risk probability, colored green-to-red
- **Expandable insight cards** — each insight from the API is a card that expands to show detail (e.g., "Debt Payment Risk" with explanation)
- **Show computed features** — display intermediate ML features (debt_ratio, liquidity_ratio, net_cash_flow, etc.) so users see what the model used. Educational value
- Results appear in the right panel of the side-by-side layout

### States & Feedback
- **Loading: skeleton/shimmer placeholders** — gray placeholder shapes with shimmer animation where results will appear
- **Validation: inline + summary banner** — red text under each invalid field AND a summary banner above the form listing all errors
- **API errors: inline in results area** — error state with retry button replaces the results panel
- **Empty/initial state: placeholder with call-to-action** — results area shows a friendly message ("Enter financial data and click Predict...") with an icon before first submission

### Claude's Discretion
- CSV upload format (based on API contract)
- Tab selection and "About the Model" content
- Exact font choices within Anthropic's aesthetic language
- Gauge animation details and timing
- Expandable card interaction mechanics
- Skeleton shimmer implementation
- Grid column widths and month row styling
- Mobile breakpoint behavior specifics

</decisions>

<specifics>
## Specific Ideas

- "Take direct inspiration from Anthropic and Claude Code, mimic the colors, fonts, symbols, and overall aesthetic"
- The frontend-design skill should be invoked first to establish the full design system before any functional code is written
- Computed features display adds educational value — users see the ML pipeline in action (raw input -> features -> prediction)
- Spreadsheet grid for 12 months should feel natural, not overwhelming — good defaults and placeholder hints are key

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-frontend-dashboard*
*Context gathered: 2026-03-02*
