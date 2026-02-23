import torch
import torch.nn as nn


class FinancialRiskModel(nn.Module):
    """
    Multi-Layer Perceptron for financial stress prediction.

    Architecture (locked):
        Input (9 features) -> 128 neurons (ReLU + Dropout)
                            -> 64 neurons (ReLU + Dropout)
                            -> 1 output (Sigmoid)

    The architecture is intentionally simple and fixed. Keeping each layer as a
    named attribute (rather than nn.Sequential) makes the forward pass readable
    and the architecture easy to inspect during learning.

    How it connects to the training pipeline:
        - train.py instantiates this model with input_size=9
        - The 9 inputs correspond to the engineered features produced by
          feature_engineering.py (e.g. savings_ratio, debt_to_income, etc.)
        - After training, the model weights are saved to models/latest_model.pth
        - predictor.py loads those weights and calls model.eval() + model(features)

    Dropout (p=0.3) is applied during training to prevent overfitting.
    It is automatically disabled during evaluation (model.eval() mode).
    """

    def __init__(self, input_size: int = 9):
        super().__init__()

        # Layer 1: input -> 128 neurons
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Layer 2: 128 -> 64 neurons
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Output layer: 64 -> 1 probability
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 1) with values in [0, 1].
            Values close to 1 indicate predicted financial stress.
        """
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x
