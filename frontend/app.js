/* ==========================================================================
   Financial Risk Dashboard — Application Logic
   Architecture: vanilla JS, no frameworks, no build step
   All API data rendered with textContent (never innerHTML) for XSS safety
   ========================================================================== */

'use strict';

// ==========================================================================
// SECTION 1: Constants & Feature Label Map
// ==========================================================================

const CIRCUMFERENCE = 251.3;  // 2 * π * r  where r = 40
const ARC           = 188.5;  // 270° arc portion: (270/360) * CIRCUMFERENCE

// Maps API feature keys → human-readable display labels
const FEATURE_LABELS = {
  avg_income:             'Avg Monthly Income',
  avg_expenses:           'Avg Monthly Expenses',
  final_savings:          'Final Savings',
  debt_payment:           'Monthly Debt Payment',
  credit_score:           'Credit Score',
  debt_ratio:             'Debt Ratio',
  expense_volatility:     'Expense Volatility',
  net_cash_flow:          'Net Cash Flow',
  savings_trend:          'Savings Trend ($/mo)',
};

// Risk color hex values (matching CSS variables --color-risk-*)
const RISK_COLORS = {
  low:    '#4caf50',
  medium: '#ff9800',
  high:   '#e53935',
};


// ==========================================================================
// SECTION 2: DOM References (cached at init time)
// ==========================================================================

let dom = {};  // Populated in initDOMRefs()

function initDOMRefs() {
  dom = {
    // Grid / form
    monthGridBody:      document.querySelector('#monthGrid tbody'),
    creditScore:        document.getElementById('creditScore'),
    predictBtn:         document.getElementById('predictBtn'),
    predictForm:        document.getElementById('predictForm'),
    validationBanner:   document.getElementById('validationBanner'),
    validationBannerList: document.getElementById('validationBannerList'),

    // CSV
    csvUpload:          document.getElementById('csvUpload'),
    csvFilename:        document.getElementById('csvFilename'),
    downloadTemplate:   document.getElementById('downloadTemplate'),

    // Results panel
    resultsPanel:       document.getElementById('resultsPanel'),

    // Gauge
    gaugeFill:          document.querySelector('.gauge-fill'),
    gaugeLabel:         document.querySelector('.gauge-label'),
    riskCategory:       document.getElementById('riskCategory'),
    riskProbability:    document.getElementById('riskProbability'),

    // Debt warning
    debtWarning:        document.getElementById('debtWarning'),

    // Insights
    insightsSummary:    document.getElementById('insightsSummary'),
    insightsContainer:  document.getElementById('insightsContainer'),

    // Computed features
    featuresGrid:       document.getElementById('featuresGrid'),

    // Error state
    errorMessage:       document.getElementById('errorMessage'),
    retryBtn:           document.getElementById('retryBtn'),

    // About tab metrics
    metricRecall:       document.getElementById('metricRecall'),
    metricRocAuc:       document.getElementById('metricRocAuc'),
    metricFeatureCount: document.getElementById('metricFeatureCount'),

    // Tab controls
    tabBtns:            document.querySelectorAll('.tab-btn'),
  };
}


// ==========================================================================
// SECTION 3: Tab Switching
// ==========================================================================

function initTabs() {
  dom.tabBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      // Deactivate all buttons and panels
      dom.tabBtns.forEach(function (b) {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      document.querySelectorAll('.tab-panel').forEach(function (panel) {
        panel.classList.remove('active');
      });

      // Activate clicked button and its panel
      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');
      const panelId = 'tab-' + btn.dataset.tab;
      const panel = document.getElementById(panelId);
      if (panel) {
        panel.classList.add('active');
      }
    });
  });
}


// ==========================================================================
// SECTION 4: Results Panel State Management
// ==========================================================================

/**
 * Switch the results panel to one of four states:
 *   'is-empty' | 'is-loading' | 'has-results' | 'is-error'
 *
 * CSS rules in styles.css show the correct .state-* child based on the
 * class applied to #resultsPanel — this function is the single source of truth.
 */
function showState(state) {
  const panel = dom.resultsPanel;
  panel.classList.remove('is-empty', 'is-loading', 'has-results', 'is-error');
  panel.classList.add(state);
}


// ==========================================================================
// SECTION 5: Form Validation
// ==========================================================================

/**
 * Remove all inline error markup and hide the banner.
 */
function clearValidationUI() {
  // Remove field-error class from all inputs in the grid
  dom.monthGridBody.querySelectorAll('.grid-input').forEach(function (input) {
    input.classList.remove('field-error');
  });

  // Remove field-error from credit score
  dom.creditScore.classList.remove('field-error');

  // Remove any injected .error-text elements
  document.querySelectorAll('.error-text').forEach(function (el) {
    el.remove();
  });

  // Hide banner
  dom.validationBanner.classList.remove('is-visible');
  dom.validationBannerList.innerHTML = '';
}

/**
 * Add the field-error class to an input and insert an .error-text span after it.
 */
function markFieldError(inputEl, message) {
  inputEl.classList.add('field-error');

  // Don't duplicate error texts
  if (inputEl.nextElementSibling && inputEl.nextElementSibling.classList.contains('error-text')) {
    return;
  }

  const span = document.createElement('span');
  span.className = 'error-text is-visible';
  span.textContent = message;
  inputEl.parentNode.insertBefore(span, inputEl.nextSibling);
}

/**
 * Populate and show the validation summary banner.
 */
function showValidationBanner(errors) {
  dom.validationBannerList.innerHTML = '';
  errors.forEach(function (msg) {
    const li = document.createElement('li');
    li.textContent = msg;
    dom.validationBannerList.appendChild(li);
  });
  dom.validationBanner.classList.add('is-visible');
}

/**
 * Get all "active" rows — contiguous rows from Month 1 where both income
 * AND expenses are filled. Stops at the first row where both fields are empty.
 *
 * Returns an array of: { month, income, expenses, debtInput, debtValue }
 */
function getActiveRows() {
  const rows = dom.monthGridBody.querySelectorAll('tr[data-month]');
  const active = [];

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const incomeInput   = row.querySelector('.income');
    const expensesInput = row.querySelector('.expenses');
    const debtInput     = row.querySelector('.debt');

    const incomeRaw   = incomeInput   ? incomeInput.value.trim()   : '';
    const expensesRaw = expensesInput ? expensesInput.value.trim() : '';

    const income   = parseFloat(incomeRaw);
    const expenses = parseFloat(expensesRaw);

    const incomeEmpty   = incomeRaw   === '' || isNaN(income);
    const expensesEmpty = expensesRaw === '' || isNaN(expenses);

    // Stop at first fully empty row (both income AND expenses empty)
    if (incomeEmpty && expensesEmpty) {
      break;
    }

    active.push({
      month:      parseInt(row.dataset.month, 10),
      income:     income,
      expenses:   expenses,
      incomeInput:  incomeInput,
      expensesInput: expensesInput,
      debtInput:  debtInput,
      incomeEmpty,
      expensesEmpty,
    });
  }

  return active;
}

/**
 * Validate the form. Returns { valid, errors, payload }.
 * payload is null when invalid.
 */
function validateForm() {
  const errors = [];
  let valid = true;

  // --- Get all data rows (including gaps) for gap detection ---
  const allRows = dom.monthGridBody.querySelectorAll('tr[data-month]');
  let firstEmptyIdx = -1;
  let hasDataAfterFirstEmpty = false;

  // Also check for gaps: find the first empty row, then see if any row after it has data
  for (let i = 0; i < allRows.length; i++) {
    const row = allRows[i];
    const incomeInput   = row.querySelector('.income');
    const expensesInput = row.querySelector('.expenses');
    const incomeRaw     = incomeInput ? incomeInput.value.trim() : '';
    const expensesRaw   = expensesInput ? expensesInput.value.trim() : '';
    const incomeEmpty   = incomeRaw === '' || isNaN(parseFloat(incomeRaw));
    const expensesEmpty = expensesRaw === '' || isNaN(parseFloat(expensesRaw));

    const rowEmpty = incomeEmpty && expensesEmpty;

    if (rowEmpty && firstEmptyIdx === -1) {
      firstEmptyIdx = i;
    } else if (!rowEmpty && firstEmptyIdx !== -1) {
      hasDataAfterFirstEmpty = true;
    }
  }

  if (hasDataAfterFirstEmpty) {
    errors.push('No gaps allowed — fill months consecutively from Month 1');
    valid = false;
  }

  // --- Active rows ---
  const activeRows = getActiveRows();

  // Minimum 6 months
  if (activeRows.length < 6) {
    errors.push('Enter at least 6 months of data');
    valid = false;
  }

  // Maximum 12 months (schema enforces this, but validate on client too)
  if (activeRows.length > 12) {
    errors.push('Maximum 12 months of data allowed');
    valid = false;
  }

  // Per-row validation: income and expenses must be non-negative, both must be filled
  activeRows.forEach(function (r) {
    if (r.incomeEmpty) {
      markFieldError(r.incomeInput, 'Required');
      if (!errors.find(function (e) { return e.includes('income'); })) {
        errors.push('Income is required for all active months');
      }
      valid = false;
    } else if (r.income < 0) {
      markFieldError(r.incomeInput, 'Must be 0 or greater');
      errors.push('Income must be non-negative (Month ' + r.month + ')');
      valid = false;
    }

    if (r.expensesEmpty) {
      markFieldError(r.expensesInput, 'Required');
      if (!errors.find(function (e) { return e.includes('expenses'); })) {
        errors.push('Expenses is required for all active months');
      }
      valid = false;
    } else if (r.expenses < 0) {
      markFieldError(r.expensesInput, 'Must be 0 or greater');
      errors.push('Expenses must be non-negative (Month ' + r.month + ')');
      valid = false;
    }

    // Debt payment: optional, but if provided must be >= 0
    if (r.debtInput) {
      const debtRaw = r.debtInput.value.trim();
      if (debtRaw !== '') {
        const debt = parseFloat(debtRaw);
        if (isNaN(debt) || debt < 0) {
          markFieldError(r.debtInput, 'Must be 0 or greater');
          errors.push('Debt payment must be non-negative (Month ' + r.month + ')');
          valid = false;
        }
      }
    }
  });

  // Credit score: required, 300-850
  const csRaw = dom.creditScore.value.trim();
  if (csRaw === '') {
    markFieldError(dom.creditScore, 'Required (300–850)');
    errors.push('Credit score is required');
    valid = false;
  } else {
    const cs = parseFloat(csRaw);
    if (isNaN(cs) || cs < 300 || cs > 850) {
      markFieldError(dom.creditScore, 'Must be between 300 and 850');
      errors.push('Credit score must be between 300 and 850');
      valid = false;
    }
  }

  if (!valid) {
    return { valid: false, errors: errors, payload: null };
  }

  // Build and return payload
  const payload = buildPayload(activeRows, parseFloat(csRaw));
  return { valid: true, errors: [], payload: payload };
}


// ==========================================================================
// SECTION 6: Build Payload
// ==========================================================================

/**
 * Construct the API request payload from validated active rows + credit score.
 */
function buildPayload(activeRows, creditScore) {
  const months = activeRows.map(function (r) {
    const debtRaw = r.debtInput ? r.debtInput.value.trim() : '';
    const debt = (debtRaw !== '' && !isNaN(parseFloat(debtRaw)))
      ? parseFloat(debtRaw)
      : 0.0;  // Default to 0 when empty

    return {
      income:       r.income,
      expenses:     r.expenses,
      debt_payment: debt,
    };
  });

  return {
    months:       months,
    credit_score: creditScore,
  };
}


// ==========================================================================
// SECTION 7: API Integration — POST /predict
// ==========================================================================

async function submitPrediction(payload) {
  showState('is-loading');

  try {
    const response = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      const msg = (data && data.error) ? data.error : `Server error (${response.status})`;
      showApiError(msg);
      return;
    }

    renderResults(data);

  } catch (err) {
    showApiError('Could not reach the server. Is the backend running?');
  }
}

function initPredictBtn() {
  // Intercept form submission (covers both button click and Enter key)
  dom.predictForm.addEventListener('submit', function (e) {
    e.preventDefault();
    handlePredict();
  });

  // Also wire explicit button click for clarity
  dom.predictBtn.addEventListener('click', function (e) {
    // The form submit handler above will fire first; this is a fallback
    // if the button is outside the form or type isn't submit.
    // Since predictBtn IS type="submit" inside predictForm, form submit fires.
    // This listener is harmless duplication — form submit will call handlePredict.
  });
}

function handlePredict() {
  clearValidationUI();
  const result = validateForm();

  if (!result.valid) {
    showValidationBanner(result.errors);
    return;
  }

  submitPrediction(result.payload);
}


// ==========================================================================
// SECTION 8: SVG Gauge
// ==========================================================================

/**
 * Animate the gauge to a risk score (0-100).
 * CSS transition on stroke-dasharray handles the animation automatically.
 */
function setGauge(riskScore) {
  const score = Math.max(0, Math.min(100, riskScore));
  const filled = (score / 100) * ARC;

  // Update fill arc
  dom.gaugeFill.setAttribute('stroke-dasharray', filled + ' ' + CIRCUMFERENCE);

  // Update color based on thresholds: high >= 65, medium >= 35, low < 35
  let color;
  if (score >= 65) {
    color = RISK_COLORS.high;
  } else if (score >= 35) {
    color = RISK_COLORS.medium;
  } else {
    color = RISK_COLORS.low;
  }
  dom.gaugeFill.setAttribute('stroke', color);

  // Update the score label inside the gauge
  dom.gaugeLabel.textContent = Math.round(score).toString();
}


// ==========================================================================
// SECTION 9: Render Results
// ==========================================================================

/**
 * Render the full API response into the results panel.
 * Expects the PredictResponse shape from schemas.py.
 */
function renderResults(data) {
  showState('has-results');

  // Gauge
  setGauge(data.risk_score);

  // Risk category label — apply color class
  const category = (data.risk_category || '').toLowerCase();
  dom.riskCategory.textContent = data.risk_category || '';
  dom.riskCategory.className   = 'gauge-category ' + category;

  // Probability percentage
  const probPct = (data.probability * 100).toFixed(1) + '%';
  dom.riskProbability.textContent = probPct;

  // Debt payment warning
  if (data.debt_payment_defaulted) {
    dom.debtWarning.classList.add('is-visible');
  } else {
    dom.debtWarning.classList.remove('is-visible');
  }

  // Insights and features
  renderInsights(data.insights);
  renderFeatures(data.computed_features);
}


// ==========================================================================
// SECTION 10: Render Insights (expandable accordion cards)
// ==========================================================================

function renderInsights(insights) {
  // Summary paragraph
  dom.insightsSummary.textContent = insights.summary || '';

  // Clear existing cards
  dom.insightsContainer.innerHTML = '';

  if (!insights.risk_factors || insights.risk_factors.length === 0) {
    const p = document.createElement('p');
    p.className = 'insights-summary';
    p.textContent = 'No specific risk factors identified.';
    dom.insightsContainer.appendChild(p);
    return;
  }

  insights.risk_factors.forEach(function (factorText) {
    // Split on first colon to get title vs body text
    const colonIdx = factorText.indexOf(':');
    let title, body;
    if (colonIdx !== -1) {
      title = factorText.substring(0, colonIdx).trim();
      body  = factorText.substring(colonIdx + 1).trim();
    } else {
      title = factorText;
      body  = '';
    }

    // Card container
    const card = document.createElement('div');
    card.className = 'insight-card';
    card.setAttribute('data-expanded', 'false');

    // Header button
    const header = document.createElement('button');
    header.type = 'button';
    header.className = 'insight-header';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'insight-title';
    titleSpan.textContent = title;

    const chevron = document.createElement('span');
    chevron.className = 'insight-chevron';
    chevron.setAttribute('aria-hidden', 'true');
    chevron.textContent = '▾';

    header.appendChild(titleSpan);
    header.appendChild(chevron);

    // Body
    const bodyDiv = document.createElement('div');
    bodyDiv.className = 'insight-body';

    const bodyP = document.createElement('p');
    bodyP.textContent = body;
    bodyDiv.appendChild(bodyP);

    // Toggle on click
    header.addEventListener('click', function () {
      const expanded = card.getAttribute('data-expanded') === 'true';
      card.setAttribute('data-expanded', expanded ? 'false' : 'true');
    });

    card.appendChild(header);
    card.appendChild(bodyDiv);
    dom.insightsContainer.appendChild(card);
  });
}


// ==========================================================================
// SECTION 11: Render Computed Features Grid
// ==========================================================================

/**
 * Format a feature value for display based on the feature key.
 */
function formatFeatureValue(key, value) {
  if (value === null || value === undefined) return '—';

  switch (key) {
    case 'debt_ratio':
    case 'expense_volatility':
      return value.toFixed(2);

    case 'savings_trend':
      // Dollar per month — can be negative, so show sign
      return (value >= 0 ? '+$' : '-$') + Math.abs(Math.round(value)).toLocaleString() + '/mo';

    case 'credit_score':
      return Math.round(value).toString();

    default:
      // Dollar values: avg_income, avg_expenses, final_savings, debt_payment, net_cash_flow
      return '$' + Math.round(value).toLocaleString();
  }
}

function renderFeatures(features) {
  dom.featuresGrid.innerHTML = '';

  Object.keys(FEATURE_LABELS).forEach(function (key) {
    const label = FEATURE_LABELS[key];
    const value = features[key];

    const item = document.createElement('div');
    item.className = 'feature-item';

    const labelSpan = document.createElement('span');
    labelSpan.className = 'feature-label';
    labelSpan.textContent = label;

    const valueSpan = document.createElement('span');
    valueSpan.className = 'feature-value';
    valueSpan.textContent = formatFeatureValue(key, value);

    item.appendChild(labelSpan);
    item.appendChild(valueSpan);
    dom.featuresGrid.appendChild(item);
  });
}


// ==========================================================================
// SECTION 12: Error State
// ==========================================================================

function showApiError(message) {
  showState('is-error');
  dom.errorMessage.textContent = message;
}

function initRetryBtn() {
  dom.retryBtn.addEventListener('click', function () {
    handlePredict();
  });
}


// ==========================================================================
// SECTION 13: CSV Upload & Template Download
// ==========================================================================

/**
 * Parse CSV text into an array of row objects.
 * Handles basic CSV (no quoted commas).
 * Expected columns: month, income, expenses, debt_payment, credit_score
 */
function parseCSV(text) {
  const lines = text.split(/\r?\n/).filter(function (l) { return l.trim() !== ''; });
  if (lines.length < 2) return { months: [], credit_score: null };

  const headers = lines[0].split(',').map(function (h) { return h.trim().toLowerCase(); });

  const monthIdx       = headers.indexOf('month');
  const incomeIdx      = headers.indexOf('income');
  const expensesIdx    = headers.indexOf('expenses');
  const debtIdx        = headers.indexOf('debt_payment');
  const creditScoreIdx = headers.indexOf('credit_score');

  const months = [];
  let credit_score = null;

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map(function (c) { return c.trim(); });

    const income   = cols[incomeIdx]   !== undefined ? cols[incomeIdx]   : '';
    const expenses = cols[expensesIdx] !== undefined ? cols[expensesIdx] : '';
    const debt     = cols[debtIdx]     !== undefined ? cols[debtIdx]     : '';
    const cs       = cols[creditScoreIdx] !== undefined ? cols[creditScoreIdx] : '';

    months.push({
      month:        monthIdx !== -1 ? (cols[monthIdx] || '') : String(i),
      income:       income,
      expenses:     expenses,
      debt_payment: debt,
      credit_score: cs,
    });

    // Take credit score from the first non-empty row
    if (credit_score === null && cs !== '') {
      credit_score = cs;
    }
  }

  return { months: months, credit_score: credit_score };
}

/**
 * Populate the grid inputs from parsed CSV rows.
 */
function populateGridFromCSV(parsed) {
  const dataRows = dom.monthGridBody.querySelectorAll('tr[data-month]');

  // Clear all inputs first
  dataRows.forEach(function (row) {
    row.querySelectorAll('.grid-input').forEach(function (input) {
      input.value = '';
    });
  });

  // Fill from CSV data (up to 12 rows)
  parsed.months.forEach(function (csvRow, idx) {
    if (idx >= dataRows.length) return;
    const row = dataRows[idx];

    const incomeInput   = row.querySelector('.income');
    const expensesInput = row.querySelector('.expenses');
    const debtInput     = row.querySelector('.debt');

    if (incomeInput   && csvRow.income   !== '') incomeInput.value   = csvRow.income;
    if (expensesInput && csvRow.expenses !== '') expensesInput.value = csvRow.expenses;
    if (debtInput     && csvRow.debt_payment !== '') debtInput.value = csvRow.debt_payment;
  });

  // Set credit score if found
  if (parsed.credit_score !== null && parsed.credit_score !== '') {
    dom.creditScore.value = parsed.credit_score;
  }
}

function initCSVUpload() {
  dom.csvUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show filename
    dom.csvFilename.textContent = file.name;

    const reader = new FileReader();
    reader.onload = function (event) {
      const text   = event.target.result;
      const parsed = parseCSV(text);
      populateGridFromCSV(parsed);
      // Clear validation UI after populating
      clearValidationUI();
    };
    reader.readAsText(file);

    // Reset the file input so the same file can be re-uploaded if needed
    e.target.value = '';
  });
}

function initDownloadTemplate() {
  dom.downloadTemplate.addEventListener('click', function () {
    const headers = 'month,income,expenses,debt_payment,credit_score';
    const rows = [];
    for (let i = 1; i <= 12; i++) {
      // Only row 1 includes a credit_score placeholder; others leave it empty
      rows.push(i + ',,,,');
    }
    const csvContent = headers + '\n' + rows.join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);

    const anchor = document.createElement('a');
    anchor.href     = url;
    anchor.download = 'financial_data_template.csv';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  });
}


// ==========================================================================
// SECTION 14: About the Model — Fetch /health
// ==========================================================================

async function fetchModelHealth() {
  try {
    const response = await fetch('/health');
    if (!response.ok) {
      throw new Error('Health endpoint returned ' + response.status);
    }
    const data = await response.json();

    // Populate metric cards with live values
    if (dom.metricRecall && data.metrics && data.metrics.recall !== undefined) {
      dom.metricRecall.textContent = (data.metrics.recall * 100).toFixed(1) + '%';
      dom.metricRecall.classList.remove('loading-placeholder');
    }

    if (dom.metricRocAuc && data.metrics && data.metrics.roc_auc !== undefined) {
      dom.metricRocAuc.textContent = data.metrics.roc_auc.toFixed(3);
      dom.metricRocAuc.classList.remove('loading-placeholder');
    }

    if (dom.metricFeatureCount && data.feature_count !== undefined) {
      dom.metricFeatureCount.textContent = data.feature_count.toString();
      dom.metricFeatureCount.classList.remove('loading-placeholder');
    }

  } catch (err) {
    // Non-fatal — About tab will show loading text; form still works
    const fallback = 'Could not load model metrics';
    if (dom.metricRecall)       dom.metricRecall.textContent       = fallback;
    if (dom.metricRocAuc)       dom.metricRocAuc.textContent       = fallback;
    if (dom.metricFeatureCount) dom.metricFeatureCount.textContent = fallback;
  }
}


// ==========================================================================
// SECTION 15: Initialization
// ==========================================================================

document.addEventListener('DOMContentLoaded', function () {
  // 1. Cache all DOM references
  initDOMRefs();

  // 2. Set initial results panel state
  showState('is-empty');

  // 3. Wire tab switching
  initTabs();

  // 4. Wire predict button / form
  initPredictBtn();

  // 5. Wire retry button
  initRetryBtn();

  // 6. Wire CSV upload
  initCSVUpload();

  // 7. Wire download template
  initDownloadTemplate();

  // 8. Fetch /health for About tab metrics (runs independently)
  fetchModelHealth();
});
