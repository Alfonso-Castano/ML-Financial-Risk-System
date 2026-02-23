import numpy as np
import torch
from torch.utils.data import Dataset


class FinancialDataset(Dataset):
    """
    PyTorch Dataset for financial profile data.

    Wraps pre-scaled numpy arrays as float32 tensors. This class has a single
    responsibility: serve (features, label) pairs to a DataLoader.

    Why scaling is external (done in train.py, not here):
        The StandardScaler must be fit on the training set only, then applied
        to both train and validation sets. Fitting inside this class would
        require passing the scaler around or re-fitting on subsets, which
        leaks information and complicates the API. Keeping scaling outside
        makes this class stateless and reusable.

    How it connects to the training pipeline:
        1. train.py loads synthetic_train.csv
        2. feature_engineering.py produces 9-column feature matrix X and labels y
        3. StandardScaler.fit_transform(X_train) -> X_train_scaled
        4. StandardScaler.transform(X_val) -> X_val_scaled
        5. FinancialDataset(X_train_scaled, y_train) -> DataLoader(train_ds)
        6. FinancialDataset(X_val_scaled, y_val)   -> DataLoader(val_ds)
        7. DataLoader yields batches of (features_tensor, label_tensor)
    """

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Args:
            features: 2D numpy array of shape (n_samples, n_features).
                      Should already be scaled before passing in.
            labels:   1D numpy array of shape (n_samples,) with binary labels
                      (0 = no stress, 1 = financial stress).
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single (features, label) pair.

        Args:
            idx: Sample index.

        Returns:
            features: Tensor of shape (n_features,)
            label:    Scalar tensor (0.0 or 1.0)
        """
        return self.features[idx], self.labels[idx]
