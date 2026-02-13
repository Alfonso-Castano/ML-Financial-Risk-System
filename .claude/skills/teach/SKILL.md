---
name: teach
description: Educational assistant for ML Financial Risk System. Teaches ML theory, PyTorch, Docker, architecture decisions, and AI engineering principles with adaptive depth control. Use when user asks to learn about or understand project concepts. Supports --level flag (beginner/intermediate/advanced).
---

# Teaching Skill for ML Financial Risk System

## Overview

This skill provides educational support for understanding ML concepts, architectural decisions, PyTorch patterns, Docker containerization, and AI engineering principles in the context of the ML Financial Risk System project.

**Purpose**: Help user deeply understand concepts while building the ML system, connecting theory to practice.

**When to use**: User explicitly invokes `/teach <topic>` to learn about any project concept, technology, or design decision.

**Teaching philosophy**: Start with big picture context, break down to fundamentals, build back up to demonstrate understanding, then connect to this specific project.

## How It Works

### Command Parsing

**Basic invocation**: `/teach <topic>`
- Example: `/teach dropout`
- Uses intermediate depth level (default)

**With depth flag**: `/teach <topic> --level=<depth>`
- Example: `/teach ml/predictor.py --level=beginner`
- Depths: beginner, intermediate, advanced

### Topic Processing

1. **Parse topic** from command (e.g., `dropout`, `binary-classification`, `docker`)
2. **Determine depth level** from `--level` flag or default to intermediate
3. **Load relevant references** from bundled documentation based on topic
4. **Structure explanation** using big picture → fundamentals → build up format
5. **Insert periodic checking questions** to verify understanding
6. **End with Q&A section** for user follow-up

### Reference Loading Strategy

Based on topic keywords, load appropriate reference files:
- **Architecture topics** (layers, orchestration, ml/predictor.py, service-layer) → Load `references/architecture.md`
- **ML concepts** (classification, neural-network, loss, metrics, overfitting) → Load `references/ml-concepts.md`
- **PyTorch topics** (tensor, autograd, dataset, dataloader, training-loop) → Load `references/pytorch-patterns.md`
- **Docker topics** (container, image, dockerfile, containerization) → Load `references/docker-basics.md`
- **Feature topics** (debt-ratio, liquidity, feature-engineering, cash-flow) → Load `references/feature-engineering.md`

If topic not explicitly in references, use general knowledge + project context from PROJECT.md/ROADMAP.md.

## Depth Levels

### Beginner (--level=beginner)

**Assumptions**: No prior knowledge of the topic exists.

**Teaching approach**:
- Extensive use of analogies and real-world comparisons
- Step-by-step breakdown of fundamentals
- More examples and concrete illustrations
- Explain "why" extensively before "how"
- Define all technical terms

**Example**: `/teach dropout --level=beginner`
- Start with analogy (e.g., dropout like not relying on one friend for all help)
- Explain overfitting in simple terms
- Show concrete example with numbers
- Connect to project: why 0.3 dropout rate chosen

### Intermediate (default)

**Assumptions**: Basic familiarity with domain, but not this specific topic.

**Teaching approach**:
- Balance theory and practice
- Assume foundational knowledge (e.g., know what a function is, but not what dropout does)
- Focus on "how it works" in context
- Some analogies when concepts are abstract
- Connect to broader patterns

**Example**: `/teach dropout` (no flag = intermediate)
- Brief definition
- Explain mechanism (randomly disable neurons)
- Why it prevents overfitting (forces redundancy)
- PyTorch implementation pattern
- Usage in this project

### Advanced (--level=advanced)

**Assumptions**: Solid understanding of fundamentals, want depth.

**Teaching approach**:
- Minimal review of basics
- Focus on edge cases, trade-offs, alternatives
- Discuss when to use vs not use
- Compare different approaches
- Mention research papers or advanced techniques
- Performance implications

**Example**: `/teach dropout --level=advanced`
- Compare dropout vs other regularization (L1, L2, batch norm)
- Discuss dropout rate selection (0.1 vs 0.3 vs 0.5)
- Inverted dropout scaling
- Dropout in inference mode (disabled)
- Alternatives: DropConnect, Zoneout

## Visual Aids

Use appropriate visual aid based on topic type:

### Code Snippets with Inline Comments

**When to use**: Explaining implementation patterns or showing how to use libraries.

**Example**: Teaching PyTorch training loop
```python
# Training loop for binary classification
for epoch in range(num_epochs):
    model.train()  # Enable dropout and batch norm training mode

    for features, labels in train_loader:
        # 1. Clear gradients from previous batch
        optimizer.zero_grad()

        # 2. Forward pass: compute predictions
        outputs = model(features)  # Shape: (batch_size, 1)

        # 3. Compute loss (how wrong predictions are)
        loss = criterion(outputs, labels)

        # 4. Backward pass: compute gradients
        loss.backward()

        # 5. Update weights using gradients
        optimizer.step()
```

### Before/After Comparisons

**When to use**: Explaining architectural decisions or refactoring rationale.

**Example**: Teaching why no service layer

**Before (what we DON'T do)**:
```
routes.py → inference_service.py → predictor.py → model.py
          ↓
         training_service.py → trainer.py → model.py
```
- Abstract service layer
- More files, more indirection

**After (what we DO)**:
```
routes.py → predictor.py → model.py
```
- Direct orchestration in predictor.py
- Simpler, beginner-friendly
- Sufficient for learning

### Step-by-Step Transformations

**When to use**: Explaining data flow, training process, or multi-stage operations.

**Example**: Teaching feature engineering pipeline

**Step 1: Raw Input**
```
Income: $5,000
Expenses: $3,500
Debt: $1,500
Savings: $2,000
```

**Step 2: Compute Features**
```
debt_ratio = 1500 / 5000 = 0.30 (30% of income to debt)
liquidity_ratio = 2000 / 3500 = 0.57 (0.57 months buffer)
```

**Step 3: Normalize**
```
debt_ratio_normalized = (0.30 - 0) / (1.0 - 0) = 0.30
liquidity_ratio_normalized = (0.57 - 0) / (10 - 0) = 0.057
```

**Step 4: Model Input**
```
[0.30, 0.057, ...other_features] → PyTorch tensor → model → probability
```

## User Knowledge Baseline

Explanations are calibrated to user's knowledge level:

**Python**: Comfortable
- Can use pythonic idioms, list comprehensions, type hints
- No need to explain basic syntax or control flow
- Can use technical terms (function, class, module, import)

**PyTorch/ML**: New
- Explain tensors (multi-dimensional arrays)
- Explain backpropagation (how gradients are computed)
- Explain training loops (forward, loss, backward, step)
- Explain Dataset/DataLoader (batching patterns)
- Define all ML terms (overfitting, regularization, activation functions)

**Web Development**: Limited
- Explain REST APIs (what is GET/POST, JSON)
- Explain FastAPI patterns (routes, schemas, endpoints)
- Explain frontend-backend communication (fetch API, HTTP requests)
- Keep web explanations concise - not the focus

**Docker**: New
- Explain containers vs images (blueprint vs instance)
- Explain Dockerfile commands (FROM, RUN, COPY, CMD)
- Explain why containerization matters (reproducibility)
- Use analogies (containers like shipping containers for software)

## Example Invocations

### Example 1: Basic ML Concept
**Command**: `/teach dropout`

**Response structure**:
- **Big Picture**: Dropout prevents overfitting by forcing network to learn redundant representations
- **Fundamentals**: Randomly disables neurons with probability p during training
- **Building Up**: Without dropout → overreliance on specific neurons → poor generalization. With dropout → robust patterns learned
- **In This Project**: Using 0.3 dropout rate after each hidden layer in 128→64→1 MLP architecture
- **Check Understanding**: "If dropout rate is 0.5, what percentage of neurons are active during training?"
- **Questions?**: Open for user follow-up

### Example 2: Architecture Decision (Beginner)
**Command**: `/teach ml/predictor.py --level=beginner`

**Response structure**:
- **Big Picture**: ml/predictor.py is where orchestration happens - coordinates loading model, computing features, running inference
- **Fundamentals**: In most systems, you'd have a separate "service layer" for business logic. We skip that for simplicity
- **Building Up**: Explain what predictor.py does step-by-step (load model, get features, predict, format response). Show why this is simpler than service abstraction
- **In This Project**: Routes are thin (just validate input), predictor does the work, clean separation
- **Analogy**: Like a kitchen where chef (predictor) coordinates all cooking, vs restaurant (service layer) where manager coordinates chef
- **Check Understanding**: "What would happen if we put feature computation inside api/routes.py instead?"
- **Questions?**: Open for user follow-up

### Example 3: Advanced Topic
**Command**: `/teach binary-classification --level=advanced`

**Response structure**:
- **Big Picture**: Binary classification with class imbalance and asymmetric costs
- **Fundamentals** (brief review): Two classes, output probability, threshold determines classification
- **Building Up to Advanced**:
  - Confusion matrix deep dive (FP vs FN trade-off)
  - ROC curves and threshold tuning
  - Class weights for imbalanced data
  - Precision-recall curves vs ROC curves
  - When to optimize for recall vs precision
- **In This Project**: Recall-prioritized because missing at-risk user is worse than false alarm. Could use threshold tuning (0.3 instead of 0.5) or class weights during training
- **Trade-offs**: High recall → more FP → stable users get warnings. Alternative: risk stratification (low/med/high) instead of binary
- **Check Understanding**: "If we change threshold from 0.5 to 0.3, what happens to recall and precision?"
- **Questions?**: Open for user follow-up

### Example 4: Docker Topic (Beginner)
**Command**: `/teach dockerfile --level=beginner`

**Response structure**:
- **Big Picture**: Dockerfile is recipe for building a Docker image (template for containers)
- **Fundamentals**: Containers are isolated environments that run apps with all dependencies. Images are blueprints.
- **Building Up**: Walk through each Dockerfile command with analogy
  - FROM = starting ingredients (base image)
  - WORKDIR = prep station (where to work)
  - COPY = bring ingredients into kitchen
  - RUN = cooking steps (install dependencies)
  - CMD = how to serve the dish (start application)
- **In This Project**: Single Dockerfile bundles backend + frontend + model for simplicity
- **Analogy**: Dockerfile like recipe card, image like prepped meal kit, container like actual cooked meal
- **Check Understanding**: "What's the difference between RUN and CMD commands?"
- **Questions?**: Open for user follow-up

### Example 5: Feature Engineering
**Command**: `/teach feature-engineering`

**Response structure**:
- **Big Picture**: Transform raw financial data into meaningful patterns that model can learn from
- **Fundamentals**: Pure functions (same input → same output), no side effects
- **Building Up**: Show examples of raw data vs engineered features
  - Raw: income=$5k, debt=$1.5k → Feature: debt_ratio=0.3
  - Why ratio is better than absolute values (relative measure)
- **In This Project**: Four key features (debt ratio, liquidity ratio, cash flow volatility, consecutive negative months)
- **Show Transformation**: Step-by-step example from raw user data to model-ready features
- **Check Understanding**: "Why is liquidity_ratio (savings/expenses) more informative than just savings amount?"
- **Questions?**: Open for user follow-up

## Teaching Format

All explanations follow this structure:

```
# Teaching: <Topic>

## The Big Picture
[High-level context - why this topic matters, where it fits in the system]

## Fundamentals
[Core concepts broken down - the building blocks needed to understand topic]

## Building It Up
[Connect fundamentals back to big picture - show how pieces fit together]

## In This Project
[Specific application to ML Financial Risk System - make it concrete]
[Code examples, configuration values, file locations]

---

**Check Understanding:**
[1-2 questions to verify comprehension - not a quiz, but reflection prompts]

---

**Questions?** Ask me anything about [topic] or related concepts.
```

## Resources

This skill includes bundled reference documentation loaded as needed:

### references/architecture.md
**Topics**: 6-layer architecture, design decisions (why no service layer, database, frameworks), separation of concerns, orchestration in ml/predictor.py

**Use for**: Architecture questions, design rationale, layer responsibilities

### references/ml-concepts.md
**Topics**: Binary classification, neural networks, MLPs, forward/backprop, loss functions, optimizers, evaluation metrics (recall, precision, F1, accuracy, ROC-AUC), overfitting/underfitting, confusion matrix

**Use for**: ML fundamentals, training concepts, metrics, model evaluation

### references/pytorch-patterns.md
**Topics**: Tensors, autograd, computation graphs, model definition (nn.Module), layer types (Linear, ReLU, Dropout, Sigmoid), Dataset/DataLoader, training loop structure, train/val/test split, model checkpointing, evaluation mode

**Use for**: PyTorch implementation, coding patterns, training setup

### references/docker-basics.md
**Topics**: Containers vs images, why containerization, Dockerfile commands (FROM, WORKDIR, COPY, RUN, EXPOSE, CMD), environment variables, building images, running containers, single vs multi-container

**Use for**: Docker concepts, containerization, deployment

### references/feature-engineering.md
**Topics**: Pure functions, financial stress indicators, engineered features (debt ratio, liquidity ratio, cash flow volatility, consecutive negatives), normalization, feature importance

**Use for**: Feature engineering, data transformation, financial metrics

## Important Notes

**Explicit invocation only**: This skill is not automatically triggered - user must call `/teach`.

**Always load references**: Based on topic, load appropriate reference documentation to ensure accurate, detailed explanations.

**Connect to project**: Every explanation should tie back to this specific ML Financial Risk System - use concrete examples from PROJECT.md, ROADMAP.md, and codebase.

**Adapt dynamically**: If topic spans multiple categories (e.g., "pytorch training loop with dropout"), load multiple references and synthesize explanation.

**Handle unknowns gracefully**: If topic not in references, provide best explanation from general knowledge + project context, and note that it's not project-specific.

**Encourage questions**: Teaching is interactive - always end with "Questions?" to invite deeper exploration.
