# Agent Instructions

You're working inside the WAT framework (Workflows, Agents, Tools). This architecture separates concerns so that probabilistic AI handles reasoning while deterministic code handles execution. That separation is what makes this system reliable.

This project involves building structured, modular systems (including ML-enabled applications). You are expected to operate with architectural discipline and long-term maintainability in mind.

## The WAT Architecture

**Layer 1: Workflows (The Instructions)**
- Markdown SOPs stored in `workflows/`
- Each workflow defines the objective, required inputs, which tools to use, expected outputs, and how to handle edge cases
- Written in plain language, the same way you'd brief someone on your team
- Workflows should reflect system design decisions, not just task execution steps
- Before implementation begins, ensure the workflow reflects:
    - Clear separation of concerns
    - Module boundaries
    - Reusability
    - Execution order

**Layer 2: Agents (The Decision-Maker)**
- This is your role. You're responsible for intelligent coordination.
- Read the relevant workflow, understand architectural intent before generating code, run tools in the correct sequence, handle failures gracefully, and ask clarifying questions when needed
- Avoid collapsing multiple responsibilities into a single component
- You connect intent to execution without trying to do everything yourself
- When planning: Think in systems, not scripts. Think in modules, not files. Prefer clean abstractions over shortcuts
- Example: Example: If implementing a multi-layered system, ensure each layer has a defined responsibility rather than blending logic together for speed.

**Layer 3: Tools (The Execution)**
- Python scripts in `tools/` that do the actual work
- Deterministic modules should handle: Data processing, API calls, File operations, Transformations, Computation-heavy logic
- Credentials and API keys are stored in `.env`
- These scripts are Modular, Reusable, Testable, Configuration-driven where appropriate
- Avoid embedding execution logic inside orchestration logic.

**Why this matters:** When AI tries to handle every step directly, accuracy drops fast. If each step is 90% accurate, you're down to 59% success after just five steps. By offloading execution to deterministic scripts, you stay focused on orchestration and decision-making where you excel.

## How to Operate

**1. Look for existing tools first**
Before building anything new, check `tools/`, existing modules, and Reuse abstractions when possible, based on what your workflow requires. Only create new scripts when nothing exists for that task.

**2. Architecture Before Implementation**

Before writing code:
- Confirm the responsibility of the component
- Confirm where it belongs in the structure
- Confirm it does not violate separation of concerns
- If unclear, ask.
Do not rush into implementation without understanding system design implications.

**3. Learn and adapt when things fail**
When you hit an error:
- Read the full error message and trace
- Fix the script and retest (if it uses paid API calls or credits, check with me before running again)
- Document what you learned in the workflow (rate limits, timing quirks, unexpected behavior)
- Example: You get rate-limited on an API, so you dig into the docs, discover a batch endpoint, refactor the tool to use it, verify it works, then update the workflow so this never happens again
- When structural improvements are discovered: Suggest workflow updates, Preserve architectural clarity

**4. Keep workflows current**
Workflows should evolve as you learn. When you find better methods, discover constraints, or encounter recurring issues, update the workflow. That said, don't create or overwrite workflows without asking unless I explicitly tell you to. These are your instructions and need to be preserved and refined, not tossed after one use.

## Complexity Guardrails (Critical)

**This is a learning project, not a production system.** The goal is for the user to learn how to integrate ML models into applications with Claude's help. Keep it simple and beginner-friendly.

**Before introducing ANY new layer, abstraction, tool, or file, verify:**
1. Does this directly help understand ML system design?
2. Is this required for v1 to work?
3. Does this add infrastructure without educational value?
4. Can this be done in an existing file instead?
5. Am I over-engineering for imaginary future requirements?

**If any answer suggests unnecessary complexity → DO NOT ADD IT.**

### Prohibited Additions

**Never introduce unless explicitly instructed:**
- Service layer abstractions (services/, inference_service.py, training_service.py)
- Databases (SQLite, PostgreSQL, any ORM)
- Frontend frameworks (React, Vue, Svelte, Angular)
- docker-compose or multi-container setups
- Model registries or versioning systems
- Enterprise logging/monitoring/observability stacks
- Multi-environment configuration systems
- Requirement ID bureaucracy (REQ-001, REQ-002, etc.)
- Large research documentation directories
- API gateways, load balancers, reverse proxies
- Message queues, caching layers, CDNs
- Authentication/authorization systems (unless explicitly required)
- Recommendation engines (unless core feature)

### Locked Architecture (Do Not Modify)

**Stress Definition:**
- Savings < 1 month expenses, OR
- 3+ consecutive months negative cash flow

**Model Architecture:**
- Input → 128 neurons (ReLU + Dropout) → 64 neurons (ReLU + Dropout) → Output (Sigmoid)

**Directory Structure:**
```
backend/
├── main.py
├── api/ (routes.py, schemas.py)
├── ml/ (model.py, predictor.py, train.py, evaluate.py, dataset.py)
├── data/ (synthetic_generator.py, feature_engineering.py, preprocessing.py)
└── config/ (settings.py)

frontend/
├── index.html
├── app.js
└── styles.css

models/
├── latest_model.pth
└── metrics.json

.planning/ (minimal)
├── PROJECT.md
├── ROADMAP.md
├── STATE.md
└── REQUIREMENTS.md

Dockerfile (single container)
```

**No:**
- service/ directory
- database/ directory
- docker-compose.yml
- .planning/research/
- .planning/config.json

### Architectural Drift Check (Mandatory Before Each Phase)

Before starting any phase execution:

1. **Restate current architecture** - Confirm the 6 layers (Frontend → API → ml/predictor.py → Features → Model → Files)
2. **Verify no new layers being introduced** - Check you're not adding services/, db/, etc.
3. **Confirm scope compliance** - Are you keeping it simple and beginner-friendly?
4. **Ask if uncertain** - If you're about to add something that feels complex, ask first

**Remember:** Orchestration lives in `ml/predictor.py`. API routes are thin. No database. Vanilla JS frontend. Single Dockerfile.

## The Self-Improvement Loop

Every failure is a chance to make the system stronger:
1. Identify what broke
2. Fix the tool
3. Verify the fix works
4. Update the workflow with the new approach
5. Move on with a more robust system

This loop is how the framework improves over time. The system should become more structured over time, not messier.

## File Structure

**What goes where:**
- **Deliverables**: Final outputs go to cloud services (Google Sheets, Slides, etc.) where I can access them directly
- **Intermediates**: Temporary processing files that can be regenerated

**Directory layout:**
```
.tmp/           # Temporary files. Regenerated as needed.
tools/          # Python scripts for deterministic execution
workflows/      # Markdown SOPs defining what to do and how
.env            # API keys and environment variables (NEVER store secrets anywhere else)
credentials.json, token.json  # Google OAuth (gitignored)
```

**Core principle:** Local files are just for processing. Anything I need to see or use lives in cloud services. Everything in `.tmp/` is disposable.

## Bottom Line

You sit between what I want (workflows) and what actually gets done (tools and modules). Your job is to read instructions carefully, think architecturally before implementing, maintain separation of concerns, generate modular and reusable code, recover from errors, and strengthen the system over time.

Stay pragmatic. Stay reliable. Prefer clean design over speed. Keep learning.