# Phase 4: Frontend Dashboard - Research

**Researched:** 2026-03-04
**Domain:** Vanilla JS Dashboard / CSS Design System / SVG Gauge / FastAPI Static Files
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Design Workflow:**
- frontend-design skill drives design first — invoke it to establish the design system (fonts, colors, layout, spatial composition) before executor writes functional code
- frontend-developer agent is dropped — incompatible with vanilla JS constraint; the skill alone handles design quality
- Design system output feeds into styles.css and index.html structure

**Aesthetic Direction:**
- Anthropic/Claude-inspired aesthetic — mimic Anthropic's visual language: warm earth tones (tan/sand backgrounds), clean sans-serif typography, distinctive orange accents, generous whitespace, understated elegance
- Match the feel of claude.ai and Anthropic's marketing pages
- CSS variables for the color system and typography scale

**Input Form Design:**
- Dual input: manual form + CSV upload
- Manual form uses a spreadsheet-style grid — all 12 months visible at once, rows are months, columns are income/expenses/debt_payment
- Additional fields outside the grid: credit_score
- Empty form with placeholder hints (e.g., "4500") — no pre-filled data
- CSV upload as alternative input method
- CSV format: Claude's Discretion — pick format based on what the API expects

**Page Structure:**
- Tabbed sections — multiple tabs to organize content
- Tab selection: Claude's Discretion — pick tabs that showcase the ML pipeline without scope creep
- Side-by-side layout on the Predict tab — form on left, results on right (desktop). Stacked on mobile
- Desktop-first, basic mobile — optimize for desktop, stack vertically on mobile as functional fallback

**Results Display:**
- Risk gauge / meter — semicircular or circular gauge visualization for the risk probability, colored green-to-red
- Expandable insight cards — each insight from the API is a card that expands to show detail
- Show computed features — display intermediate ML features so users see what the model used
- Results appear in the right panel of the side-by-side layout

**States & Feedback:**
- Loading: skeleton/shimmer placeholders — gray placeholder shapes with shimmer animation
- Validation: inline + summary banner — red text under each invalid field AND a summary banner above the form
- API errors: inline in results area — error state with retry button replaces the results panel
- Empty/initial state: placeholder with call-to-action — results area shows a friendly message before first submission

### Claude's Discretion

- CSV upload format (based on API contract)
- Tab selection and "About the Model" content
- Exact font choices within Anthropic's aesthetic language
- Gauge animation details and timing
- Expandable card interaction mechanics
- Skeleton shimmer implementation
- Grid column widths and month row styling
- Mobile breakpoint behavior specifics

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 4 builds a three-file vanilla JS dashboard (index.html, app.js, styles.css) that integrates directly with the existing POST /predict and GET /health FastAPI endpoints. The backend already mounts `frontend/` at "/" via StaticFiles — so no server changes are required; the files just need to exist. The design execution follows a two-step workflow: the frontend-design skill establishes the full visual design system first, then functional JS is written against that design.

The technical domain is well-understood vanilla web platform. No build tools, no bundlers, no npm. The most complex pieces are (1) the SVG gauge visualization, which can be built purely with SVG stroke-dasharray/dashoffset manipulation in ~40 lines of code or via the zero-dependency svg-gauge library, and (2) the shimmer skeleton loader, which is a pure CSS keyframe animation on a gradient background. Everything else — tabs, expandable cards, fetch calls, CSV parsing with FileReader — follows standard patterns that have been stable for years.

The Anthropic color palette is documented: background `#eeece2` (warm sand), body text `#3d3929` (dark brown), primary accent `#da7756` (terra cotta orange), button accent `#bd5d3a` (darker orange). Typography: Anthropic uses the commercial Styrene font for headings. The closest free equivalent for a web dashboard is Plus Jakarta Sans (Google Fonts), which shares the geometric sans-serif character without the Inter-style genericness. A serif body font (DM Serif Display or Lora) can optionally pair for the "About" tab to echo Claude's editorial feel.

**Primary recommendation:** Write the three files in order — styles.css (design system + all states), index.html (structure), app.js (fetch + DOM). Invoke the frontend-design skill before writing any code to get the exact design system spec.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| Vanilla HTML/CSS/JS | N/A (browser built-in) | All UI | Locked constraint; no build tools |
| Fetch API | Browser built-in | POST /predict, GET /health | Standard async HTTP in modern browsers |
| FileReader API | Browser built-in | CSV file reading | No library needed for simple CSV |
| CSS Custom Properties | Browser built-in | Design token system | Native, zero-cost theming |
| SVG stroke-dasharray | Browser built-in | Gauge visualization | Pure SVG math, zero dependencies |

### Supporting (Claude's Discretion — Recommend Based on Research)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| svg-gauge (naikus) | 1.x (npm/CDN) | Animated SVG gauge | If hand-rolled gauge needs animation smoothness; CDN avoids npm |
| Google Fonts (Plus Jakarta Sans) | CDN | Anthropic-adjacent typography | Free, closest match to Styrene; load via `<link>` in `<head>` |

**Recommendation on svg-gauge vs hand-rolled:** Build it hand-rolled. The math is 5 lines. svg-gauge adds a CDN dependency for marginal benefit. The hand-rolled SVG approach is more educational and aligns with the project's learning goal.

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Hand-rolled SVG gauge | svg-gauge library | Library adds CDN dependency; hand-roll is ~40 lines and educational |
| Plus Jakarta Sans | DM Sans, Outfit | DM Sans is slightly more neutral; Plus Jakarta Sans has more personality |
| CSS skeleton shimmer | JS-based loading spinner | Shimmer is pure CSS, no JS needed, better UX signal |
| novalidate + custom JS validation | HTML5 native validation | Native validation bubbles are not styleable; custom gives inline + banner pattern |

**Installation:** None. No npm. All via `<link>` and `<script>` tags, or built-in browser APIs.

---

## Architecture Patterns

### Recommended File Structure

```
frontend/
├── index.html    # Structure: tabs, grid form, results panel, modals
├── app.js        # All behavior: fetch, DOM, validation, CSV parsing, gauge
└── styles.css    # All visual: CSS vars, layout, states, animations
```

No subdirectories. Three files. This is the locked structure.

### Pattern 1: CSS Variable Design Token System

**What:** All colors, spacing, typography, and radii defined as CSS custom properties on `:root`. Components reference variables, never raw values.

**When to use:** Always — the design system starts here before any component code.

**Example (verified from Anthropic brand colors research):**
```css
/* Source: Anthropic brand identity, verified 2026-03-04 */
:root {
  /* Color system - Anthropic/Claude palette */
  --color-bg:         #eeece2;  /* warm sand background */
  --color-surface:    #f5f3eb;  /* slightly lighter card surface */
  --color-text:       #3d3929;  /* warm dark brown body */
  --color-text-muted: #6b6654;  /* muted secondary text */
  --color-accent:     #da7756;  /* terra cotta primary accent */
  --color-accent-dark:#bd5d3a;  /* button hover / darker accent */
  --color-border:     #d9d6c8;  /* subtle warm border */
  --color-error:      #c0392b;  /* validation error red */

  /* Risk colors (gauge green-to-red) */
  --color-risk-low:    #4caf50;
  --color-risk-medium: #ff9800;
  --color-risk-high:   #e53935;

  /* Typography */
  --font-sans: 'Plus Jakarta Sans', ui-sans-serif, system-ui, sans-serif;
  --font-mono: ui-monospace, 'Cascadia Code', monospace;

  /* Scale */
  --text-xs:   0.75rem;
  --text-sm:   0.875rem;
  --text-base: 1rem;
  --text-lg:   1.125rem;
  --text-xl:   1.25rem;
  --text-2xl:  1.5rem;
  --text-3xl:  1.875rem;

  /* Spacing */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-6: 1.5rem;
  --space-8: 2rem;
  --space-12: 3rem;

  /* Radius */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-lg: 12px;
}
```

### Pattern 2: Tab Switching (Vanilla JS)

**What:** Tab navigation using `data-tab` attributes and CSS class toggling. No library.

**When to use:** Multiple content sections that need to be shown/hidden.

**Recommended tabs (Claude's Discretion decision):**
- **"Predict"** — the main form + results panel
- **"About the Model"** — explains the ML pipeline, the 9 features, risk thresholds. Educational value. Fulfills the "showcase the ML pipeline" goal without adding any new API calls.

**Example:**
```html
<nav class="tab-nav" role="tablist">
  <button class="tab-btn active" data-tab="predict" role="tab">Predict</button>
  <button class="tab-btn" data-tab="about" role="tab">About the Model</button>
</nav>
<section id="tab-predict" class="tab-panel active">...</section>
<section id="tab-about" class="tab-panel">...</section>
```

```javascript
// Source: standard vanilla JS tab pattern, verified from MDN/CSS-Tricks
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});
```

### Pattern 3: SVG Semicircular Gauge (Hand-Rolled)

**What:** SVG circle with stroke-dasharray/dashoffset math to draw a progress arc. Rotated 135 degrees to make a semicircle (270-degree arc). Color transitions green (low) → orange (medium) → red (high) based on risk value.

**When to use:** Risk probability display. Animates on each new prediction.

**Example (verified from SVG spec and fullstack.com research):**
```html
<!-- Source: SVG stroke-dasharray technique, verified from MDN SVG docs -->
<svg class="gauge" viewBox="0 0 100 100" width="200" height="200">
  <circle class="gauge-bg" cx="50" cy="50" r="40"
    stroke-dasharray="188.5 251.3"
    stroke-dashoffset="-31.4"
    fill="none" stroke-width="10" />
  <circle class="gauge-fill" cx="50" cy="50" r="40"
    stroke-dasharray="0 251.3"
    fill="none" stroke-width="10"
    stroke-linecap="round" />
  <text class="gauge-label" x="50" y="58" text-anchor="middle">--</text>
</svg>
```

```javascript
// Circumference of r=40 circle: 2π×40 ≈ 251.3
// Arc is 270° = 75% of circumference = 188.5
// Rotation offset to start at bottom-left: rotate(-225deg) on the SVG
const CIRCUMFERENCE = 251.3;
const ARC = 188.5;  // 270° arc

function setGauge(percent) {
  const filled = (percent / 100) * ARC;
  const fillEl = document.querySelector('.gauge-fill');
  fillEl.style.strokeDasharray = `${filled} ${CIRCUMFERENCE}`;
  fillEl.style.stroke = percent >= 65 ? 'var(--color-risk-high)'
                       : percent >= 35 ? 'var(--color-risk-medium)'
                       : 'var(--color-risk-low)';
  document.querySelector('.gauge-label').textContent = `${percent}`;
}
```

```css
.gauge { transform: rotate(-225deg); }
.gauge-bg { stroke: var(--color-border); }
.gauge-fill { transition: stroke-dasharray 0.8s ease-out, stroke 0.4s ease; }
```

### Pattern 4: CSS Shimmer Skeleton

**What:** Gray placeholder rectangles with a sweeping shimmer animation using CSS gradient + keyframes. Applied while waiting for API response.

**When to use:** Replace the results panel content with skeleton shapes during fetch.

**Example (verified from CSS-Tricks and codewithbilal.medium.com research):**
```css
/* Source: Pure CSS shimmer technique, verified from multiple sources 2026-03-04 */
.skeleton {
  background: linear-gradient(
    100deg,
    var(--color-border) 0%,
    var(--color-surface) 50%,
    var(--color-border) 100%
  );
  background-size: 300% 100%;
  animation: shimmer 1.5s infinite linear;
  border-radius: var(--radius-sm);
}

@keyframes shimmer {
  0%   { background-position: 200% center; }
  100% { background-position: -100% center; }
}

/* Skeleton shapes */
.skeleton-gauge   { width: 200px; height: 120px; border-radius: 50%; margin: 0 auto; }
.skeleton-text-lg { height: 1.5rem; width: 60%; margin-bottom: var(--space-2); }
.skeleton-text-sm { height: 1rem; width: 80%; margin-bottom: var(--space-2); }
.skeleton-card    { height: 3rem; width: 100%; margin-bottom: var(--space-2); }
```

### Pattern 5: Expandable Insight Cards

**What:** Cards that toggle an expanded body section using CSS max-height transition. Each card shows the risk factor title, toggles open to reveal explanation text.

**When to use:** Displaying the `insights.risk_factors` array from the API response.

**Example:**
```html
<div class="insight-card" data-expanded="false">
  <button class="insight-header">
    <span class="insight-title">Low Emergency Fund</span>
    <span class="insight-chevron">▾</span>
  </button>
  <div class="insight-body">
    <p>Savings cover only 0.8 months of expenses (recommended: 1+ months)</p>
  </div>
</div>
```

```css
.insight-body {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease-out, padding 0.2s ease;
}
.insight-card[data-expanded="true"] .insight-body {
  max-height: 200px;
  padding: var(--space-3) var(--space-4);
}
.insight-card[data-expanded="true"] .insight-chevron { transform: rotate(180deg); }
.insight-chevron { transition: transform 0.2s ease; display: inline-block; }
```

```javascript
document.querySelectorAll('.insight-header').forEach(btn => {
  btn.addEventListener('click', () => {
    const card = btn.closest('.insight-card');
    const expanded = card.dataset.expanded === 'true';
    card.dataset.expanded = String(!expanded);
  });
});
```

### Pattern 6: Spreadsheet Grid Form

**What:** HTML `<table>` with one row per month (rows 1-12), columns for Month label, Income, Expenses, Debt Payment. All inputs visible at once. Uses `<input type="number" placeholder="4500">`.

**When to use:** The manual data entry form for the 12-month financial grid.

**Grid layout decision:** Use a real `<table>` element — not CSS grid. Tables have correct semantics for tabular data, inherit column widths naturally, and support `<th>` headers. This gives proper tab-order behavior for form navigation.

**Important:** The API accepts 6-12 months. The form always shows 12 rows. Empty rows at the end are excluded when building the JSON payload. Rows with any filled field count as "submitted." Contiguous filled rows from month 1 define the active range.

**Example structure:**
```html
<table class="month-grid" id="monthGrid">
  <thead>
    <tr>
      <th>Month</th>
      <th>Income ($)</th>
      <th>Expenses ($)</th>
      <th>Debt Payment ($)</th>
    </tr>
  </thead>
  <tbody>
    <!-- Rows 1-12 generated by JS or static HTML -->
    <tr data-month="1">
      <td class="month-label">Month 1</td>
      <td><input type="number" class="grid-input income" placeholder="4500" min="0" step="0.01"></td>
      <td><input type="number" class="grid-input expenses" placeholder="3200" min="0" step="0.01"></td>
      <td><input type="number" class="grid-input debt" placeholder="500" min="0" step="0.01"></td>
    </tr>
    <!-- ... rows 2-12 ... -->
  </tbody>
</table>
<div class="credit-score-row">
  <label for="creditScore">Credit Score</label>
  <input type="number" id="creditScore" placeholder="720" min="300" max="850" step="1">
</div>
```

### Pattern 7: CSV Upload Format and Parsing

**What:** FileReader API reads the file as text, custom split logic parses rows. No library.

**CSV format decision (based on API contract):**
The API expects `months[]` (income, expenses, debt_payment per month) and `credit_score`. The natural CSV format is:

```csv
month,income,expenses,debt_payment,credit_score
1,4500,3200,500,720
2,4800,3100,500,720
...
```

`credit_score` is repeated on every row (the parser uses the value from row 1, or the last non-empty value). Alternative flat format with `credit_score` as a separate first row is also viable but adds parsing complexity. Row-per-month is simpler.

**Example parsing:**
```javascript
// Source: FileReader API, MDN Web Docs — browser built-in
function parseCSV(text) {
  const lines = text.trim().split('\n').filter(l => l.trim());
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const rows = lines.slice(1).map(line => {
    const vals = line.split(',').map(v => v.trim());
    return Object.fromEntries(headers.map((h, i) => [h, vals[i]]));
  });
  return rows;
}

function csvRowsToPayload(rows) {
  const creditScore = parseFloat(rows[0]?.credit_score ?? 720);
  const months = rows.map(r => ({
    income: parseFloat(r.income || 0),
    expenses: parseFloat(r.expenses || 0),
    debt_payment: parseFloat(r.debt_payment || 0),
  }));
  return { months, credit_score: creditScore };
}
```

### Pattern 8: Fetch + Error Handling

**What:** POST to /predict with JSON body, check `response.ok`, handle 422 validation errors (which return `{"error": "message"}`) and network failures separately.

**No CORS issues:** Frontend is served from the same FastAPI origin — `StaticFiles` mounts at "/", API routes registered before that. Same-origin requests have no CORS restriction.

**Example (verified from MDN Fetch API docs):**
```javascript
// Source: MDN Fetch API documentation, verified 2026-03-04
async function submitPrediction(payload) {
  showLoading();
  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      // 422 returns {"error": "field: message"} per backend design
      showApiError(data.error || `HTTP ${res.status}`);
      return;
    }
    renderResults(data);
  } catch (err) {
    // Network failure, server unreachable
    showApiError('Could not reach the server. Is the backend running?');
  } finally {
    hideLoading();
  }
}
```

### Pattern 9: Inline Validation + Summary Banner

**What:** Use `novalidate` on the form to suppress native browser bubbles. On submit, iterate all grid inputs, validate ranges, collect errors. Render red helper text under invalid fields and a summary banner at the top of the form.

**Validation rules (from schemas.py):**
- `income` >= 0 (required, non-negative)
- `expenses` >= 0 (required, non-negative)
- `debt_payment` >= 0 (optional, defaults to 0, non-negative)
- `credit_score` between 300 and 850 (required)
- Month count: 6-12 months with filled income+expenses (debt_payment can be empty)
- Rows must be contiguous from month 1 (no gaps allowed — simple UX choice)

**Error display pattern:**
```javascript
function validateForm() {
  const errors = [];
  clearValidationUI();

  const activeRows = getActiveRows(); // rows with income+expenses filled
  if (activeRows.length < 6) {
    errors.push('Enter at least 6 months of data (income and expenses required)');
  }
  if (activeRows.length > 12) {
    errors.push('Maximum 12 months allowed');
  }

  const creditScore = parseFloat(document.getElementById('creditScore').value);
  if (isNaN(creditScore) || creditScore < 300 || creditScore > 850) {
    markFieldError('creditScore', 'Must be between 300 and 850');
    errors.push('Credit score must be between 300 and 850');
  }

  // Per-row validation
  activeRows.forEach((row, i) => {
    const income = parseFloat(row.querySelector('.income').value);
    if (isNaN(income) || income < 0) {
      markFieldError(row.querySelector('.income'), `Month ${i+1}: income must be ≥ 0`);
      errors.push(`Month ${i+1}: income must be a non-negative number`);
    }
    // ... similar for expenses
  });

  if (errors.length > 0) {
    showValidationBanner(errors);
    return false;
  }
  return true;
}
```

### Pattern 10: FastAPI Static Files — No Changes Needed

**What:** The backend already has `app.mount("/", StaticFiles(directory="frontend", html=True))` in `backend/main.py`. No server modifications needed for Phase 4.

**Behavior confirmed:**
- `GET /` → serves `frontend/index.html`
- `GET /styles.css` → serves `frontend/styles.css`
- `GET /app.js` → serves `frontend/app.js`
- `POST /predict` → API route (registered first, takes precedence)
- `GET /health` → API route (registered first, takes precedence)

**No CORS configuration needed** because frontend and API share the same origin.

### Anti-Patterns to Avoid

- **Fetching relative paths with `../`:** All fetch calls use absolute paths like `/predict` and `/health` — the backend mount handles routing.
- **Using `innerHTML` with raw API data:** Risk of XSS if API ever returns unexpected content. Use `textContent` for all user-visible string values from the API.
- **Putting business logic in index.html:** All behavior goes in app.js. HTML is structure only.
- **Hardcoding colors:** Everything goes through CSS variables. The design system must be changeable from one place.
- **Using `querySelector` repeatedly for the same element:** Cache DOM references at initialization time; don't query on every event.
- **Blocking the main thread with CSV parsing:** FileReader's `onload` callback is already async. Keep parsing synchronous but fast — simple split() is fine for this use case.
- **Showing all 12 months as "required":** Months 7-12 are optional. Only months 1-6 should have visual emphasis as minimum requirement. Placeholder text handles the hint.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Shimmer animation | JS-driven loading state | Pure CSS @keyframes on gradient | CSS shimmer is 10 lines, zero JS, smooth 60fps |
| Gauge animation | requestAnimationFrame counter | CSS transition on stroke-dasharray | Browser handles interpolation natively |
| CSV parsing | Full RFC 4180 parser | Simple split('\n') + split(',') | API input is our own CSV format — no quoted commas expected |
| Font loading | Self-hosted fonts | Google Fonts CDN `<link>` | Zero setup, already cached for many users |
| Tab state | URL hash routing | data-* attributes + classList | Single page, no back/forward navigation needed |

**Key insight:** Every "smart" abstraction here adds complexity without adding value. The browser already does the hard parts (font caching, CSS transitions, async file reading). Let it.

---

## Common Pitfalls

### Pitfall 1: SVG Gauge Coordinate System

**What goes wrong:** The gauge arc starts at the wrong position because SVG 0° points right (3 o'clock), not up.

**Why it happens:** SVG coordinate system has 0° at 3 o'clock and rotates clockwise. A semicircle starting at bottom-left requires a -225deg rotation on the SVG element.

**How to avoid:** Apply `transform: rotate(-225deg)` to the SVG element itself in CSS. The gauge-bg circle should show the full 270° arc; the gauge-fill starts at 0 and fills up.

**Warning signs:** Gauge appears to fill from the wrong starting position or points downward.

### Pitfall 2: stroke-dasharray Circumference Math

**What goes wrong:** Gauge fill percentage is wrong because circumference formula uses the wrong radius.

**Why it happens:** `stroke-width` eats into the circle area. The correct radius for circumference calculation is the center of the stroke path, which equals the `r` attribute value (not `r + stroke-width/2`).

**How to avoid:** Circumference = 2π × r (where r matches the SVG `r` attribute). For r=40: 2 × 3.14159 × 40 = 251.33. The 270° arc = 251.33 × 0.75 = 188.5.

**Warning signs:** Gauge shows 50% when it should show 100%, or arc doesn't close at the right position.

### Pitfall 3: CSS max-height Transition on Expandable Cards

**What goes wrong:** The collapse animation doesn't work (card snaps shut instead of sliding).

**Why it happens:** `max-height: 0 → auto` doesn't animate. `auto` is not a numeric value CSS can interpolate.

**How to avoid:** Use a fixed `max-height` pixel value large enough to contain the content (e.g., `max-height: 300px`) for the expanded state. The animation eases from 0 to that value.

**Warning signs:** Opening animates but closing snaps, or neither animates.

### Pitfall 4: Empty Row Handling in the Month Grid

**What goes wrong:** Sending months with income=0 and expenses=0 for unfilled rows causes the API to accept them as valid data, skewing the prediction.

**Why it happens:** `<input type="number">` with empty value returns `""` — parseFloat("") returns NaN, but if you default to 0, you're sending zeroed-out months.

**How to avoid:** An "active row" is defined as having a non-empty income AND non-empty expenses value. Parse with `parseFloat(input.value)` and check `isNaN()` — NaN means the field was empty. Only include rows where both income and expenses are non-NaN. Stop at the first empty row (no gaps).

**Warning signs:** API accepts 12 months when user only filled 3, returning unexpectedly low or high risk.

### Pitfall 5: FastAPI StaticFiles html=True Redirect Behavior

**What goes wrong:** Navigating to `http://localhost:8000` shows a redirect or 307, not the page.

**Why it happens:** StaticFiles with html=True redirects `/foo` to `/foo/` (adds trailing slash) but serves `index.html` for `/` (root). This is correct behavior and doesn't affect the dashboard — users just access `http://localhost:8000/`.

**How to avoid:** No action needed. Just document that the frontend URL is `http://localhost:8000/` (with trailing slash or without — both work for root).

**Warning signs:** None — this is expected behavior, not a bug.

### Pitfall 6: Number Input Step Attribute and Validation

**What goes wrong:** Browser marks income/expenses fields as invalid because `step` doesn't match entered value (e.g., step=1 but user enters 4500.50).

**Why it happens:** `<input type="number" step="1">` rejects decimal values with native validation.

**How to avoid:** Use `step="0.01"` for income/expenses/debt_payment (financial values). Use `step="1"` for credit_score (integer 300-850).

**Warning signs:** Browser shows "Please enter a valid value" on fields with decimal amounts.

### Pitfall 7: textContent vs innerHTML for API Data

**What goes wrong:** XSS risk if insight text from API contains characters like `<` or `>`.

**Why it happens:** `innerHTML` interprets HTML entities. API text could contain `<` in financial comparisons (e.g., "savings < 1 month").

**How to avoid:** Always use `element.textContent = apiString` for inserting strings from the API response. Never use `innerHTML` for variable data.

**Warning signs:** Insight text displays raw HTML tags or renders unexpected elements.

---

## Code Examples

Verified patterns from official/MDN sources:

### Reading a CSV File with FileReader

```javascript
// Source: FileReader API — MDN Web Docs (browser built-in)
document.getElementById('csvUpload').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (evt) => {
    const text = evt.target.result;
    const payload = csvTextToPayload(text);
    if (payload) populateGridFromPayload(payload);
  };
  reader.readAsText(file);
});
```

### Building the POST /predict Payload

```javascript
// Assembles the payload from the month grid form
function buildPayload() {
  const rows = Array.from(document.querySelectorAll('#monthGrid tbody tr'));
  const months = [];
  for (const row of rows) {
    const income = parseFloat(row.querySelector('.income').value);
    const expenses = parseFloat(row.querySelector('.expenses').value);
    const debt = parseFloat(row.querySelector('.debt').value);
    if (isNaN(income) || isNaN(expenses)) break; // stop at first empty row
    months.push({
      income,
      expenses,
      debt_payment: isNaN(debt) ? 0.0 : debt,
    });
  }
  const creditScore = parseFloat(document.getElementById('creditScore').value);
  return { months, credit_score: creditScore };
}
```

### Rendering Computed Features Grid

```javascript
// Formats the 9 engineered features returned in the API response
const FEATURE_LABELS = {
  avg_income: 'Avg Monthly Income',
  avg_expenses: 'Avg Monthly Expenses',
  final_savings: 'Final Savings',
  debt_payment: 'Monthly Debt Payment',
  credit_score: 'Credit Score',
  debt_ratio: 'Debt Ratio',
  liquidity_ratio: 'Liquidity Ratio',
  net_cash_flow: 'Net Cash Flow',
  consec_negative_months: 'Consecutive Negative Months',
};

function renderFeatures(features) {
  const container = document.getElementById('featuresGrid');
  container.innerHTML = ''; // safe — no user data in feature keys
  for (const [key, label] of Object.entries(FEATURE_LABELS)) {
    const val = features[key];
    const formatted = ['debt_ratio', 'liquidity_ratio'].includes(key)
      ? val.toFixed(2)
      : key === 'consec_negative_months'
      ? Math.round(val).toString()
      : `$${Math.round(val).toLocaleString()}`;
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.innerHTML = `<span class="feature-label"></span><span class="feature-value"></span>`;
    item.querySelector('.feature-label').textContent = label;
    item.querySelector('.feature-value').textContent = formatted;
    container.appendChild(item);
  }
}
```

### Skeleton → Results State Transition

```javascript
// Show skeleton state while loading
function showLoading() {
  document.getElementById('resultsPanel').classList.add('is-loading');
  document.getElementById('resultsPanel').classList.remove('is-empty', 'is-error', 'has-results');
}

// Switch to results state
function showResults(data) {
  const panel = document.getElementById('resultsPanel');
  panel.classList.remove('is-loading');
  panel.classList.add('has-results');
  renderGauge(data.risk_score);
  renderInsights(data.insights);
  renderFeatures(data.computed_features);
}
```

---

## API Contract Reference

The following is the exact API contract Phase 4 integrates against (from schemas.py):

### POST /predict

**Request body:**
```json
{
  "months": [
    { "income": 4500, "expenses": 3200, "debt_payment": 500 },
    ...
  ],
  "credit_score": 720
}
```

- `months`: array, 6-12 entries required
- `months[].income`: float, >= 0
- `months[].expenses`: float, >= 0
- `months[].debt_payment`: float, >= 0, defaults to 0.0 if omitted
- `credit_score`: float, 300-850

**Response body:**
```json
{
  "risk_score": 73.4,
  "risk_category": "high",
  "probability": 0.7341,
  "insights": {
    "risk_factors": ["Low emergency fund: ...", "High debt burden: ..."],
    "summary": "Multiple risk factors indicate..."
  },
  "computed_features": {
    "avg_income": 4500.0,
    "avg_expenses": 3200.0,
    "final_savings": 800.0,
    "debt_payment": 500.0,
    "credit_score": 720.0,
    "debt_ratio": 0.38,
    "liquidity_ratio": 0.25,
    "net_cash_flow": -200.0,
    "consec_negative_months": 4.0
  },
  "debt_payment_defaulted": false
}
```

**Risk categories:** low (< 35), medium (35-64), high (>= 65)

**Error response (422):**
```json
{ "error": "months -> 0 -> income: Value error, income must be non-negative" }
```

### GET /health

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feature_count": 9,
  "metrics": { "recall": 0.94, "roc_auc": 0.98 }
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| HSL colors | OKLCH colors | CSS Color Level 4 (now widely supported) | Better perceptual uniformity; Anthropic uses oklch(0.70 0.14 45) for their accent. Hex equivalents work fine for this project's scope. |
| Canvas-based gauges | SVG stroke-dasharray | ~2015 onwards | SVG scales perfectly, accessible, CSS-animatable |
| XMLHttpRequest | fetch() + async/await | ~2018, universal now | Cleaner promise-based API, no polyfills needed |
| inline `onload=` | addEventListener | Always preferred | Separation of concerns, testable |
| px-based media queries | Container queries (new) | 2023, >90% support | For this project, viewport px breakpoints at 768px are sufficient |

**Deprecated/outdated:**
- jQuery `$.ajax()`: Not used. Fetch API is universal.
- Form submission with page reload: Not used. All submission is via fetch, DOM updated in place.
- `document.write()`: Never use. DOM manipulation only.

---

## Open Questions

1. **Minimum month requirement UX: where is month 6 visually indicated?**
   - What we know: API requires 6-12 months. Grid shows 12 rows.
   - What's unclear: Should rows 7-12 be visually dimmed or labeled "optional"?
   - Recommendation: Add a subtle visual separator or "Optional" label after row 6, or dim rows 7-12 with reduced opacity. The frontend-design skill should make this call during design phase.

2. **debt_payment_defaulted warning: how prominent?**
   - What we know: API returns `debt_payment_defaulted: true` when any month has debt_payment == 0.0.
   - What's unclear: The user may have legitimately no debt. A warning could be misleading.
   - Recommendation: Show a small informational note (not an error): "Months with no debt payment recorded — if you have debt, enter it above." Only show when debt_payment_defaulted is true.

3. **About the Model tab content: static HTML or fetched from /health?**
   - What we know: GET /health returns `metrics: { recall, roc_auc }`, `feature_count: 9`.
   - What's unclear: Should the About tab fetch live metrics or use hardcoded copy?
   - Recommendation: Fetch /health on page load, populate the model metrics section dynamically. Hardcode the architectural explanation (6 layers, MLP architecture). This adds minimal JS and demonstrates the /health endpoint.

4. **CSV template download: needed?**
   - What we know: Users can upload CSV. They need to know the format.
   - What's unclear: Should there be a "Download CSV Template" button?
   - Recommendation: Yes — a simple JS function that generates and downloads a blank template CSV with correct headers. Pure JS `Blob` + `URL.createObjectURL` + `<a>.click()`. Trivial to implement, high UX value.

---

## Sources

### Primary (HIGH confidence)
- Anthropic brand identity page (beginswithai.com verified 2026-03-04) — hex colors #eeece2, #3d3929, #da7756, #bd5d3a
- MDN Web Docs — Fetch API, FileReader API, stroke-dasharray SVG attribute (browser built-ins, stable)
- FastAPI StaticFiles docs (fastapi.tiangolo.com) — html=True behavior confirmed
- backend/api/schemas.py (project source) — exact API contract, validation rules
- backend/ml/predictor.py (project source) — risk thresholds, 9 feature names

### Secondary (MEDIUM confidence)
- Anthropic typography research (type.today Styrene article + companyfonts.com) — Styrene + Tiempos confirmed as Anthropic's commercial fonts; Plus Jakarta Sans as free alternative
- fullstack.com SVG gauge tutorial — stroke-dasharray technique verified against MDN SVG spec
- codewithbilal.medium.com / CSS-Tricks — shimmer skeleton CSS pattern verified against multiple sources
- naikus/svg-gauge GitHub — library API documented (considered and rejected in favor of hand-rolled)

### Tertiary (LOW confidence)
- shadcn.io Claude theme — oklch(0.70 0.14 45) as accent color; corroborated by brand research but not official Anthropic source

---

## Metadata

**Confidence breakdown:**
- API contract: HIGH — verified directly from project source files
- Anthropic colors: HIGH — verified from brand identity sources with hex codes
- Standard stack (SVG gauge, FileReader, fetch): HIGH — all browser built-ins, MDN documented
- Typography (Plus Jakarta Sans recommendation): MEDIUM — Anthropic uses commercial Styrene; free alternative is a research recommendation
- Shimmer CSS pattern: HIGH — stable CSS technique, multiple authoritative sources
- svg-gauge library: MEDIUM — library evaluated from GitHub; recommendation to not use it is based on complexity/benefit analysis

**Research date:** 2026-03-04
**Valid until:** 2026-04-04 (browser APIs stable; Anthropic brand may update)
