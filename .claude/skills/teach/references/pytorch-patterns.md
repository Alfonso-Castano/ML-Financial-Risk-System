# PyTorch Patterns Reference

## PyTorch Tensors

### What Are Tensors?

**Tensor**: Multi-dimensional array, like NumPy's `ndarray` but optimized for:
- **GPU acceleration**: Massive parallel computation
- **Automatic differentiation**: Tracks operations for backpropagation

**Dimensions**:
- 0D tensor (scalar): Single number
- 1D tensor (vector): `[1, 2, 3]`
- 2D tensor (matrix): `[[1, 2], [3, 4]]`
- 3D+ tensor: Image batches, video, etc.

### Creating Tensors

```python
import torch

# From Python list
x = torch.tensor([1, 2, 3])

# With specific dtype
x = torch.tensor([1.0, 2.0], dtype=torch.float32)

# Random initialization
x = torch.randn(3, 5)  # 3 rows, 5 columns, random normal distribution

# Zeros/ones
x = torch.zeros(2, 3)  # 2x3 matrix of zeros
x = torch.ones(4, 4)   # 4x4 matrix of ones
```

### Tensor Operations

```python
# Element-wise operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b  # [5, 7, 9]

# Matrix multiplication
A = torch.randn(3, 4)
B = torch.randn(4, 5)
C = torch.matmul(A, B)  # or A @ B, shape: (3, 5)

# Reshaping
x = torch.randn(12)
y = x.view(3, 4)  # Reshape to 3x4
```

### Moving to GPU

```python
# Check if GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move tensor to GPU
x = x.to(device)

# Or create on GPU directly
x = torch.randn(3, 5, device=device)
```

## Autograd (Automatic Differentiation)

### What Is Autograd?

**Autograd**: PyTorch's automatic differentiation engine
- Tracks all operations on tensors
- Computes gradients automatically during backpropagation
- Enables training neural networks

### How It Works

```python
# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4

# Compute gradient
y.backward()  # Computes dy/dx

# Access gradient
print(x.grad)  # tensor([4.]) because dy/dx = 2x = 2*2 = 4
```

### Computation Graph

PyTorch builds a **computation graph** tracking:
- Which tensors were involved in each operation
- What operation was performed
- How gradients should flow backward

**Example**:
```python
x = torch.tensor([1.0], requires_grad=True)
y = x + 2    # Operation: add
z = y * 3    # Operation: multiply
z.backward() # Gradients flow: z → y → x
```

### In Neural Networks

**Forward pass**: Build computation graph
**Backward pass**: Traverse graph backwards, compute gradients

```python
model = MyModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Forward pass (builds graph)
outputs = model(inputs)
loss = criterion(outputs, targets)

# Backward pass (computes gradients)
loss.backward()

# Update weights (uses gradients)
optimizer.step()
```

## Model Definition

### Subclassing nn.Module

**Pattern**: All PyTorch models inherit from `nn.Module`.

```python
import torch.nn as nn

class FinancialRiskModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()  # MUST call parent __init__

        # Define layers
        self.layer1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)

        self.output_layer = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)

        return x
```

### Why Two Methods?

**`__init__`**: Define what layers the model has (architecture)

**`forward`**: Define how data flows through layers (computation)

**Separation reason**: PyTorch can automatically handle backward pass knowing the architecture and forward computation.

## Layer Types

### nn.Linear (Fully Connected Layer)

**Purpose**: Apply linear transformation `y = xW + b`

**Parameters**:
- `in_features`: Number of input features
- `out_features`: Number of output features

```python
layer = nn.Linear(in_features=10, out_features=20)
# Input: tensor of shape (batch_size, 10)
# Output: tensor of shape (batch_size, 20)
```

**What it learns**: Weight matrix `W` (10x20) and bias vector `b` (20,)

### nn.ReLU (Rectified Linear Unit)

**Purpose**: Apply non-linear activation

**Math**: `ReLU(x) = max(0, x)`

**Effect**:
- Positive values: unchanged
- Negative values: set to zero

```python
relu = nn.ReLU()
x = torch.tensor([-2, -1, 0, 1, 2])
y = relu(x)  # tensor([0, 0, 0, 1, 2])
```

**Why needed**: Without non-linearity, stacking layers is pointless (linear transformations are composable into single linear transformation).

### nn.Dropout (Regularization)

**Purpose**: Prevent overfitting by randomly disabling neurons during training.

**How it works**:
- **Training**: Randomly set neurons to zero with probability `p`
- **Evaluation**: All neurons active (dropout disabled)

```python
dropout = nn.Dropout(p=0.3)  # 30% dropout rate

# During training
model.train()
x = torch.randn(5, 10)
y = dropout(x)  # ~30% of values are zero

# During evaluation
model.eval()
y = dropout(x)  # All values preserved
```

**Why it works**: Forces network to learn redundant representations - can't rely on specific neurons.

### nn.Sigmoid (Output Activation)

**Purpose**: Squash output to range (0, 1) for binary classification probability.

**Math**: `Sigmoid(x) = 1 / (1 + e^(-x))`

**Effect**:
- Large positive x → output ≈ 1
- Large negative x → output ≈ 0
- x = 0 → output = 0.5

```python
sigmoid = nn.Sigmoid()
x = torch.tensor([-10, -1, 0, 1, 10])
y = sigmoid(x)  # tensor([0.0000, 0.2689, 0.5000, 0.7311, 1.0000])
```

**Use case**: Binary classification - interpret output as probability.

## Dataset and DataLoader Patterns

### Custom Dataset Class

**Pattern**: Subclass `torch.utils.data.Dataset` and implement two methods.

```python
from torch.utils.data import Dataset

class FinancialDataset(Dataset):
    def __init__(self, features, labels):
        """
        Initialize dataset with features and labels.

        Args:
            features: numpy array or pandas DataFrame of shape (N, num_features)
            labels: numpy array of shape (N,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        """
        Return total number of samples.
        MUST be implemented.
        """
        return len(self.features)

    def __getitem__(self, idx):
        """
        Return a single sample at index idx.
        MUST be implemented.

        Args:
            idx: Index of sample to retrieve

        Returns:
            Tuple of (features, label) for that sample
        """
        return self.features[idx], self.labels[idx]
```

### Why Dataset Pattern?

**Separation of concerns**:
- Dataset: How to load and access data
- DataLoader: How to batch and shuffle

**Memory efficiency**: Load samples on-demand, not all at once.

**Reusability**: Same dataset can be used with different DataLoaders (different batch sizes, shuffling, etc.).

### DataLoader

**Purpose**: Batch, shuffle, and parallelize data loading.

```python
from torch.utils.data import DataLoader

# Create dataset
dataset = FinancialDataset(features, labels)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,      # Load 32 samples at a time
    shuffle=True,       # Randomize order each epoch
    num_workers=4       # Parallel data loading (4 subprocesses)
)

# Iterate over batches
for batch_features, batch_labels in dataloader:
    # batch_features: tensor of shape (32, num_features)
    # batch_labels: tensor of shape (32,)
    outputs = model(batch_features)
    loss = criterion(outputs, batch_labels)
    # ... backprop and optimization
```

**Why batching?**:
- **GPU efficiency**: Process multiple samples in parallel
- **Memory**: Can't fit all data on GPU at once
- **Gradient stability**: Average gradient over batch more stable than single sample

## Training Loop Structure

### Standard Training Pattern

```python
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        # ========== TRAINING PHASE ==========
        model.train()  # Enable dropout, batch norm training mode
        train_loss = 0.0

        for batch_features, batch_labels in train_loader:
            # 1. Zero gradients (clear from previous iteration)
            optimizer.zero_grad()

            # 2. Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            # 3. Backward pass (compute gradients)
            loss.backward()

            # 4. Update weights
            optimizer.step()

            train_loss += loss.item()

        # ========== VALIDATION PHASE ==========
        model.eval()  # Disable dropout, batch norm eval mode
        val_loss = 0.0

        with torch.no_grad():  # Disable gradient computation (saves memory)
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
```

### Why These Steps?

**1. Zero gradients**: Gradients accumulate by default - must clear before each batch.

**2. Forward pass**: Compute predictions (builds computation graph).

**3. Backward pass**: Compute gradients (traverses graph backwards).

**4. Optimizer step**: Update weights using gradients.

**5. Validation without gradients**: Saves memory and computation - not training during validation.

### Train vs Eval Mode

```python
# Training mode
model.train()
# - Dropout is active
# - Batch normalization uses batch statistics

# Evaluation mode
model.eval()
# - Dropout is disabled (all neurons active)
# - Batch normalization uses running statistics
```

**Important**: Always switch mode appropriately!

## Train/Val/Test Split

### Why Three Splits?

**Training set (70%)**: Used to update weights - model learns from this.

**Validation set (15%)**: Used to tune hyperparameters and monitor overfitting - model doesn't learn from this.

**Test set (15%)**: Final evaluation only - model has never seen this.

### The Problem with Two Splits

If you only have train/test:
- Tune hyperparameters on test set
- Test set is no longer "unseen" - you've indirectly optimized for it
- Performance is overly optimistic

### The Solution

**Workflow**:
1. Split data: train (70%), val (15%), test (15%)
2. Train model on train set
3. Evaluate on val set, adjust hyperparameters
4. Repeat steps 2-3 until satisfied
5. **Once**: Evaluate on test set for final performance
6. Report test performance (this is the honest performance)

### In Code

```python
from sklearn.model_selection import train_test_split

# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42  # 0.15/0.85 ≈ 0.176
)

# Result: 70% train, 15% val, 15% test
```

## Model Checkpointing

### Saving Models

**Purpose**: Save trained model to disk for later use.

```python
# Save entire model state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'models/checkpoint.pth')

# Or just model weights (common)
torch.save(model.state_dict(), 'models/latest_model.pth')
```

**What's saved**:
- `state_dict()`: Dictionary mapping parameter names to tensors (weights and biases)
- Does NOT save model architecture - must define model class separately

### Loading Models

```python
# Create model instance (same architecture as when saved)
model = FinancialRiskModel(input_size=10)

# Load weights
model.load_state_dict(torch.load('models/latest_model.pth'))

# Set to evaluation mode
model.eval()

# Now ready for inference
```

**Important**: Model architecture must match exactly - can't load weights from different architecture.

### Best Practices

**Save best validation model**: Not final model, but model with lowest validation loss.

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # ... training code ...

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/best_model.pth')
```

**Why**: Final model might be overfit. Best validation model generalizes better.

## Evaluation Mode

### torch.no_grad()

**Purpose**: Disable gradient computation during inference.

```python
model.eval()
with torch.no_grad():
    predictions = model(features)
```

**Why**:
- **Memory**: Gradients not needed during inference - saves memory
- **Speed**: Skips computation graph building - faster
- **Safety**: Prevents accidental weight updates

### model.eval() vs torch.no_grad()

**`model.eval()`**: Changes model behavior
- Disables dropout
- Changes batch norm behavior

**`torch.no_grad()`**: Disables autograd
- No gradient computation
- No computation graph

**Best practice**: Use both during evaluation/inference.

## Common Patterns in This Project

### Model Definition

```python
class FinancialRiskModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
```

### Training

```python
model = FinancialRiskModel(input_size=num_features)
criterion = nn.BCELoss()  # Binary cross-entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Inference

```python
model.load_state_dict(torch.load('models/latest_model.pth'))
model.eval()

with torch.no_grad():
    features_tensor = torch.tensor(features, dtype=torch.float32)
    probability = model(features_tensor).item()
    risk_score = int(probability * 100)
```

## Key Takeaways

1. **Tensors**: Multi-dimensional arrays with GPU support
2. **Autograd**: Automatic differentiation for backpropagation
3. **nn.Module**: Base class for all models
4. **Layers**: Linear, ReLU, Dropout, Sigmoid
5. **Dataset**: Load and access data samples
6. **DataLoader**: Batch, shuffle, parallelize
7. **Training loop**: zero_grad → forward → backward → step
8. **Train/val/test**: 70/15/15 split prevents overfitting
9. **Checkpointing**: Save best model based on validation loss
10. **Evaluation**: Use model.eval() and torch.no_grad() during inference
