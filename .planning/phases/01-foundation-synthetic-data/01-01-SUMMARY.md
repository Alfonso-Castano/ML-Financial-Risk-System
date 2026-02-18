---
phase: 01-foundation-synthetic-data
plan: 01
subsystem: data-generation
tags: [numpy, pandas, synthetic-data, financial-modeling, stress-labeling]

# Dependency graph
requires:
  - phase: 00-initialization
    provides: Project structure and planning framework
provides:
  - Complete backend directory structure with config, data, ml, api modules
  - Configuration-driven synthetic data generator with 4 financial archetypes
  - Training dataset with 36,000 labeled financial profiles (3,000 profiles x 12 months)
  - Stress labeling logic based on savings and cash flow patterns
affects: [02-ml-model-training, 03-api-layer, feature-engineering, preprocessing]

# Tech tracking
tech-stack:
  added: [numpy>=2.0, pandas>=2.0]
  patterns: [archetype-based data generation, configuration-driven design, long-format time series data]

key-files:
  created:
    - backend/config/settings.py
    - backend/data/synthetic_generator.py
    - data/synthetic_train.csv
  modified: []

key-decisions:
  - "Used 4 financial archetypes (struggling, getting_by, stable, comfortable) matching user's salary bands"
  - "Calibrated archetype parameters iteratively to achieve 36.2% stress ratio (target: ~35%)"
  - "Applied long format (one row per person-month) for time series compatibility"
  - "Fixed debt payment per profile, variable income/expenses with 15% monthly variance"

patterns-established:
  - "Configuration in settings.py as single source of truth"
  - "Archetype-based generation for realistic financial diversity"
  - "Stress labeling via two-condition logic (savings OR cash flow)"
  - "Reproducible generation with seed parameter"

# Metrics
duration: 3min 22sec
completed: 2026-02-17
---

# Phase 1 Plan 1: Foundation & Synthetic Data Summary

**Archetype-based synthetic financial generator producing 36,000 labeled profiles with 36.2% stress ratio using numpy/pandas**

## Performance

- **Duration:** 3 minutes 22 seconds
- **Started:** 2026-02-18T04:46:16Z
- **Completed:** 2026-02-18T04:49:38Z
- **Tasks:** 2
- **Files modified:** 12

## Accomplishments
- Created complete backend directory structure aligned with locked architecture
- Built configuration-driven synthetic data generator with 4 financial archetypes
- Generated 36,000 rows of labeled training data (3,000 profiles x 12 months)
- Achieved target class balance of 36.2% stressed profiles through parameter calibration

## Task Commits

Each task was committed atomically:

1. **Task 1: Create project directory structure and configuration** - `154de72` (feat)
2. **Task 2: Build synthetic generator and produce training CSV** - `4d41325` (feat)

## Files Created/Modified

- `backend/__init__.py` - Backend module initialization
- `backend/config/__init__.py` - Configuration module initialization
- `backend/config/settings.py` - All data generation constants and archetype definitions
- `backend/data/__init__.py` - Data processing module initialization
- `backend/data/synthetic_generator.py` - Archetype-based profile generator with stress labeling
- `backend/ml/__init__.py` - ML module initialization (placeholder for Phase 2)
- `backend/api/__init__.py` - API module initialization (placeholder for Phase 3)
- `models/.gitkeep` - Model artifacts directory (placeholder for Phase 2)
- `frontend/.gitkeep` - Frontend directory (placeholder for Phase 4)
- `requirements.txt` - Python dependencies (numpy, pandas)
- `data/synthetic_train.csv` - 36,000 labeled financial profile records

## Decisions Made

**Archetype parameter calibration for target stress ratio:**
- Initial archetype parameters produced only 6.67% stress ratio, far below target
- Iteratively adjusted savings_months, expense_ratio, and archetype weights
- Final parameters: struggling (30% weight, 0-0.2 savings months, 0.95-1.10 expense ratio), getting_by (25% weight, 0.1-0.8 savings months, 0.85-0.98 expense ratio)
- Achieved 36.2% stress ratio, very close to 35% target

**Long format time series structure:**
- Used one row per person-month (36,000 rows) rather than one row per person (3,000 rows)
- Enables temporal feature engineering in Phase 2
- Stress label applied consistently across all 12 months for each profile

**Fixed vs variable financial components:**
- Debt payment and credit score remain constant per profile
- Income and expenses vary monthly with 15% standard deviation
- Savings track cumulatively month-over-month

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed archetype parameters to achieve target class balance**
- **Found during:** Task 2 (Initial data generation)
- **Issue:** Initial archetype parameters (from plan spec) produced only 6.67% stressed profiles, far below 35% target. Generator logic was correct, but parameter values didn't create realistic stress conditions.
- **Fix:** Iteratively adjusted archetype parameters in two rounds:
  - Round 1: Increased struggling weight to 0.25, reduced savings buffers, raised expense ratios (result: 23.53%)
  - Round 2: Further reduced savings buffers (0-0.2 months for struggling), increased struggling weight to 0.30, allowed expense ratios up to 1.10 (spending more than income) (result: 36.20%)
- **Files modified:** backend/config/settings.py
- **Verification:** Regenerated data three times, verified reproducibility (identical MD5 hashes), manually inspected stressed/healthy profiles to confirm stress conditions trigger correctly
- **Committed in:** 4d41325 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug fix)
**Impact on plan:** Auto-fix necessary to achieve specified must-have (35% stress ratio). No scope creep - generator structure and labeling logic unchanged, only parameter values calibrated.

## Issues Encountered

None - plan executed smoothly after archetype parameter calibration.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 2: ML Model & Training**
- Training dataset available at data/synthetic_train.csv
- 36,000 labeled samples with realistic financial patterns
- Stress labeling verified correct per two-condition rule
- All columns present for feature engineering (income, expenses, savings, debt, credit_score)

**No blockers identified**

## Self-Check: PASSED

All files verified present:
- backend/__init__.py
- backend/config/__init__.py
- backend/config/settings.py
- backend/data/__init__.py
- backend/data/synthetic_generator.py
- backend/ml/__init__.py
- backend/api/__init__.py
- models/.gitkeep
- frontend/.gitkeep
- requirements.txt
- data/synthetic_train.csv

All commits verified:
- 154de72 (Task 1)
- 4d41325 (Task 2)

---
*Phase: 01-foundation-synthetic-data*
*Completed: 2026-02-17*
