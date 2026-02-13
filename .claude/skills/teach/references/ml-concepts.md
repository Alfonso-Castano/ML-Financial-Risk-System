# ML Concepts Reference

## Binary Classification

### What It Is

**Binary classification**: ML task where the goal is to predict one of **two** possible categories (classes).

**Examples**:
- Email: spam vs not spam
- Medical: disease vs healthy
- Financial: stress vs stable (this project!)
- Image: cat vs dog

**Output**: Probability between 0 and 1
- Close to 0 → Class 0 (negative class)
- Close to 1 → Class 1 (positive class)

### When to Use

Use binary classification when:
- Problem has exactly 2 outcomes
- Outcomes are mutually exclusive
- Historical labeled data available

### In This Project

**Task**: Predict financial stress probability within 3-6 months

**Classes**:
- `0` (Stable): Savings ≥ 1 month expenses AND < 3 consecutive months negative cash flow
- `1` (Stress): Savings < 1 month expenses OR ≥ 3 consecutive months negative cash flow

## Neural Networks Basics

### What Is a Neural Network?

A **neural network** is a computational model inspired by biological brains:
- **Neurons**: Individual computation units
- **Layers**: Groups of neurons
- **Connections**: Weighted edges between neurons
- **Activation functions**: Non-linear transformations

**Purpose**: Learn complex patterns by combining simple operations.

### Neuron Operation

Each neuron performs:
1. **Weighted sum**: Multiply each input by a weight, sum results
2. **Add bias**: Add a learnable bias term
3. **Activation**: Apply non-linear function (ReLU, Sigmoid, etc.)

**Math**: `output = activation(sum(weights * inputs) + bias)`

### Layers

**Input Layer**: Raw features (income, debt, savings, etc.)

**Hidden Layers**: Intermediate transformations that learn patterns

**Output Layer**: Final prediction (probability for binary classification)

**Why multiple layers?**: Each layer learns increasingly abstract patterns
- Layer 1: Simple features (ratios, thresholds)
- Layer 2: Combinations of features (debt + low savings)
- Layer 3: Complex patterns (financial stress indicators)

## Multi-Layer Perceptron (MLP)

### What Is an MLP?

**MLP** (Multi-Layer Perceptron): Simplest type of neural network
- Fully connected layers (every neuron connects to every neuron in next layer)
- Multiple layers stacked sequentially
- No loops or branches - straightforward forward pass

**Also called**: Feed-forward neural network, fully-connected network

### In This Project

```
Input (n features)
    ↓
Linear Layer (128 neurons)
    ↓
ReLU Activation
    ↓
Dropout (0.3)
    ↓
Linear Layer (64 neurons)
    ↓
ReLU Activation
    ↓
Dropout (0.3)
    ↓
Linear Layer (1 neuron)
    ↓
Sigmoid Activation
    ↓
Output (probability 0-1)
```

**Why 128 → 64 → 1?**: Gradually compress information from many features to single probability

## Forward Propagation

### What It Is

**Forward propagation**: Process of computing predictions by passing data through the network layer by layer.

**Steps**:
1. Start with input features
2. Apply first layer (weighted sum + bias + activation)
3. Pass result to next layer
4. Repeat until output layer
5. Get final prediction

**Key point**: No learning happens during forward propagation - just computation.

## Backpropagation

### What It Is

**Backpropagation**: Algorithm for computing gradients (how much each weight contributed to the error).

**Purpose**: Tells optimizer which direction to adjust weights to reduce loss.

**How it works**:
1. Compute prediction (forward pass)
2. Calculate loss (how wrong the prediction was)
3. Compute gradient of loss with respect to each weight (chain rule of calculus)
4. Flow gradients backward through network
5. Update weights using optimizer

**Why "back"?**: Gradients computed in reverse order (output → input).

### Chain Rule

**Mathematical foundation**: If `y = f(g(x))`, then `dy/dx = (dy/dg) * (dg/dx)`

**In neural networks**: Loss depends on output, output depends on last layer, last layer depends on previous layer, etc.

Backpropagation applies chain rule recursively to get gradients for all weights.

## Loss Functions

### What Is Loss?

**Loss**: Measure of how wrong a prediction is.
- Low loss = accurate predictions
- High loss = inaccurate predictions

**Goal**: Minimize loss by adjusting weights.

### Binary Cross-Entropy Loss

**Used for**: Binary classification (this project!)

**Formula**: `-[y * log(ŷ) + (1-y) * log(1-ŷ)]`

Where:
- `y` = true label (0 or 1)
- `ŷ` = predicted probability (0 to 1)

**Intuition**: Penalizes confident wrong predictions heavily.

**Example**:
- True label: 1 (stress), Predicted: 0.9 → Low loss (good!)
- True label: 1 (stress), Predicted: 0.1 → High loss (bad!)

## Optimizers

### What Is an Optimizer?

**Optimizer**: Algorithm that updates weights to minimize loss.

**Steps each iteration**:
1. Compute gradients (backpropagation)
2. Update weights in direction that reduces loss
3. Repeat until convergence

### Adam Optimizer

**Why Adam?**: Most popular optimizer for deep learning
- **Adaptive learning rates**: Different learning rate for each weight
- **Momentum**: Uses past gradients to smooth updates
- **Bias correction**: Adjusts for initial bias in estimates

**Hyperparameters**:
- `lr` (learning rate): Step size for updates (typically 0.001)
- `betas`: Momentum parameters (typically (0.9, 0.999))

**Why use Adam in this project?**: Works well out-of-the-box, requires minimal tuning.

## Evaluation Metrics

### Why Multiple Metrics?

Different metrics capture different aspects of model performance. A single metric (like accuracy) can be misleading.

### Recall

**Also called**: Sensitivity, True Positive Rate

**Formula**: `Recall = TP / (TP + FN)`

**Meaning**: Of all actual positive cases, how many did we catch?

**Example**: Of 100 people who will have financial stress, how many did we correctly identify?

**In this project**: **Priority metric** - we want to catch all at-risk users (minimize false negatives).

**Trade-off**: High recall may mean more false positives (warning stable users).

### Precision

**Formula**: `Precision = TP / (TP + FP)`

**Meaning**: Of all predicted positives, how many were actually positive?

**Example**: Of 100 people we warned about stress, how many actually experienced stress?

**Trade-off**: High precision may mean more false negatives (missing at-risk users).

### F1 Score

**Formula**: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`

**Meaning**: Harmonic mean of precision and recall.

**Use case**: Balance between precision and recall.

### Accuracy

**Formula**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

**Meaning**: Overall correctness - how many predictions were right?

**Limitation**: Misleading for imbalanced data
- If 95% of cases are stable, predicting "stable" always gives 95% accuracy!

### ROC-AUC

**ROC Curve**: Plot of True Positive Rate vs False Positive Rate at different thresholds.

**AUC** (Area Under Curve): Single number summarizing ROC curve (0 to 1).
- AUC = 0.5: Random guessing
- AUC = 1.0: Perfect classifier

**Use case**: Measure how well model ranks predictions (not just accuracy at one threshold).

## Why Recall Is Prioritized in This Project

### The Problem

Financial stress prediction has **asymmetric costs**:
- **False Negative** (miss at-risk user): User experiences stress without warning - HIGH COST
- **False Positive** (warn stable user): User gets unnecessary warning - LOW COST

### The Solution

**Optimize for recall**: Catch all at-risk users, even if it means warning some stable users.

**Trade-off accepted**: Better to warn 10 stable users than miss 1 at-risk user.

### How to Optimize for Recall

1. **Threshold tuning**: Lower prediction threshold (e.g., 0.3 instead of 0.5)
2. **Class weights**: Penalize false negatives more during training
3. **Evaluation**: Report recall as primary metric, secondary metrics for context

## Overfitting vs Underfitting

### Overfitting

**Problem**: Model learns training data too well - memorizes instead of generalizes.

**Symptoms**:
- High training accuracy, low test accuracy
- Model performs well on seen data, poorly on new data

**Causes**:
- Too many parameters (complex model)
- Too few training examples
- Training too long

**Solutions**:
- **Dropout**: Randomly disable neurons during training
- **Early stopping**: Stop training when validation loss stops improving
- **Regularization**: Penalize large weights
- **More data**: Increase training set size

### Underfitting

**Problem**: Model is too simple to capture patterns.

**Symptoms**:
- Low training accuracy, low test accuracy
- Model performs poorly everywhere

**Causes**:
- Too few parameters (simple model)
- Insufficient training

**Solutions**:
- **Larger model**: Add more layers or neurons
- **Train longer**: More epochs
- **Better features**: Engineer more informative features

### Finding the Balance

**Goal**: Model generalizes well to unseen data.

**Strategy**:
1. Start with simple model
2. Gradually increase complexity
3. Monitor validation performance
4. Stop when validation performance plateaus

### In This Project

**Dropout** (0.3 rate): Prevents overfitting by randomly dropping 30% of neurons during each training step.

**Early stopping**: Monitor validation loss, stop if it doesn't improve for several epochs.

**Train/val/test split** (70/15/15): Validate during training, test after final model selected.

## Confusion Matrix

### What It Is

**Confusion Matrix**: Table showing true vs predicted labels.

```
                Predicted Negative    Predicted Positive
Actual Negative       TN                    FP
Actual Positive       FN                    TP
```

**Terms**:
- **TN** (True Negative): Correctly predicted negative (stable users correctly identified)
- **TP** (True Positive): Correctly predicted positive (at-risk users correctly identified)
- **FP** (False Positive): Incorrectly predicted positive (stable users wrongly warned)
- **FN** (False Negative): Incorrectly predicted negative (at-risk users missed) - **AVOID THIS!**

### Using Confusion Matrix

**Calculate metrics**:
- Recall = TP / (TP + FN)
- Precision = TP / (TP + FP)
- Accuracy = (TP + TN) / (TP + TN + FP + FN)

**Identify problems**:
- High FN → Low recall → Need lower threshold or better features
- High FP → Low precision → Model too sensitive

## Key Takeaways

1. **Binary classification**: Two classes, output probability
2. **Neural networks**: Layers of neurons learn patterns
3. **MLPs**: Simplest neural network architecture
4. **Forward prop**: Compute predictions
5. **Backprop**: Compute gradients for learning
6. **Loss**: Measure error, minimize with optimizer
7. **Recall prioritized**: Catch all at-risk users, accept some false positives
8. **Overfitting**: Use dropout, early stopping, validation split
