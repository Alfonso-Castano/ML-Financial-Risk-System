"""
Training pipeline for the financial risk ML model.

Orchestrates the full flow from raw CSV to a saved model checkpoint:
    CSV -> feature engineering -> profile-level split -> scaling -> train -> save

Key design decisions (see .planning/STATE.md for rationale):
    - Split is by profile_id, not by row, to prevent data leakage
    - StandardScaler is fit on train data ONLY, then applied to val and test
    - Best model by validation loss is saved (not the final epoch model)
    - Early stopping halts training when val loss stops improving
    - tqdm progress bar shows per-epoch loss without spamming stdout

Run from project root:
    python -m backend.ml.train
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from backend.config import settings
from backend.data.feature_engineering import build_feature_matrix, FEATURE_NAMES
from backend.ml.model import FinancialRiskModel
from backend.ml.dataset import FinancialDataset


def train_epoch(
    model: FinancialRiskModel,
    loader: DataLoader,
    criterion: nn.BCELoss,
    optimizer: torch.optim.Adam,
) -> float:
    """
    Run one full training epoch over all batches.

    Sets model to training mode (enables dropout), iterates over the DataLoader,
    computes loss, backpropagates, and updates weights.

    Args:
        model:     The FinancialRiskModel instance.
        loader:    DataLoader yielding (features, labels) batches.
        criterion: Binary cross-entropy loss function.
        optimizer: Adam optimizer holding model parameters.

    Returns:
        Average training loss across all batches for this epoch.
    """
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        optimizer.zero_grad()
        outputs = model(features).squeeze(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(
    model: FinancialRiskModel,
    loader: DataLoader,
    criterion: nn.BCELoss,
) -> float:
    """
    Run one full validation pass over all batches without updating weights.

    Sets model to eval mode (disables dropout) and uses torch.no_grad() to
    skip gradient computation, reducing memory and compute overhead.

    Args:
        model:     The FinancialRiskModel instance.
        loader:    DataLoader yielding (features, labels) batches.
        criterion: Binary cross-entropy loss function.

    Returns:
        Average validation loss across all batches for this epoch.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)


def save_scaler_stats(
    scaler: StandardScaler,
    feature_names: list,
    path: str,
) -> None:
    """
    Persist StandardScaler mean and scale to JSON for Phase 3 inference.

    The API predictor needs these statistics to apply the same scaling
    transformation to new single-profile inputs at inference time.

    Args:
        scaler:        Fitted StandardScaler (must have mean_ and scale_ attrs).
        feature_names: List of feature name strings in canonical order.
        path:          Output file path for the JSON file.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stats = {
        "feature_names": feature_names,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def main() -> dict:
    """
    Execute the full training pipeline from CSV to saved model.

    Steps:
        1. Set random seeds for reproducibility
        2. Load the synthetic training CSV
        3. Build 9-feature matrix via feature engineering
        4. Split profiles into train/val/test (no row-level leakage)
        5. Fit StandardScaler on train only; transform all splits
        6. Save scaler stats for Phase 3 API inference
        7. Create Datasets and DataLoaders for each split
        8. Initialize model, loss, and optimizer
        9. Train with early stopping; save best val-loss checkpoint
       10. Load the best checkpoint and return results

    Returns:
        Dict containing: model, X_test, y_test, train_losses, val_losses,
        num_epochs_trained, and split sizes.
    """
    # 1. Reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. Load raw data
    df = pd.read_csv(settings.DATA_PATH)

    # 3. Engineer features: (n_profiles, 9), (n_profiles,), [profile_ids]
    X, y, profile_ids = build_feature_matrix(df)

    # 4. Split by shuffled index with stratification to preserve class balance.
    #    Splitting indices (not profile_id arrays) is simpler and avoids issues
    #    when profile_ids are non-contiguous.
    indices = np.arange(len(y))

    # First carve out test set (15% of total)
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=settings.TEST_SIZE,
        stratify=y,
        random_state=42,
    )

    # Then split train_val into train and val.
    # VAL_SIZE=0.176 gives ~15% of the original total (0.176 * 0.85 ≈ 0.15).
    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=settings.VAL_SIZE,
        stratify=y[idx_train_val],
        random_state=42,
    )

    X_train, y_train = X[idx_train], y[idx_train]
    X_val,   y_val   = X[idx_val],   y[idx_val]
    X_test,  y_test  = X[idx_test],  y[idx_test]

    # 5. Scale — fit on train only to prevent data leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 6. Persist scaler statistics for Phase 3 API inference
    save_scaler_stats(scaler, FEATURE_NAMES, settings.SCALER_PATH)

    # 7. Datasets and DataLoaders
    train_ds = FinancialDataset(X_train.astype(np.float32), y_train)
    val_ds   = FinancialDataset(X_val.astype(np.float32),   y_val)
    test_ds  = FinancialDataset(X_test.astype(np.float32),  y_test)

    train_loader = DataLoader(
        train_ds, batch_size=settings.BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_ds,   batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(  # noqa: F841  (available for evaluate.py in __main__)
        test_ds,  batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # 8. Model, loss, optimizer
    model     = FinancialRiskModel(input_size=settings.INPUT_SIZE)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.LEARNING_RATE)

    # Ensure models/ directory exists before first save
    Path(settings.MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 9. Training loop with early stopping
    train_losses: list = []
    val_losses: list   = []
    best_val_loss      = float("inf")
    patience_counter   = 0

    with tqdm(range(settings.NUM_EPOCHS), desc="Training") as pbar:
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            val_loss   = validate_epoch(model, val_loader, criterion)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            pbar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
            )

            # Save checkpoint if val loss improved
            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), settings.MODEL_PATH)
            else:
                patience_counter += 1
                if patience_counter >= settings.EARLY_STOPPING_PATIENCE:
                    # Early stopping triggered — break before max epochs
                    break

    num_epochs_trained = len(train_losses)

    # 10. Load the best saved checkpoint (not the final-epoch model)
    model.load_state_dict(torch.load(settings.MODEL_PATH, weights_only=True))

    return {
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "num_epochs_trained": num_epochs_trained,
        "best_val_loss": best_val_loss,
        "split_sizes": {
            "train": len(idx_train),
            "val": len(idx_val),
            "test": len(idx_test),
        },
    }


if __name__ == "__main__":
    results = main()

    # Save training history so evaluate.py can generate loss-curve plots
    # independently without needing to re-run training.
    history_path = "models/training_history.json"
    with open(history_path, "w") as f:
        json.dump(
            {
                "train_losses": results["train_losses"],
                "val_losses": results["val_losses"],
            },
            f,
            indent=2,
        )

    sizes = results["split_sizes"]
    print(
        f"Training complete. "
        f"Model saved to {settings.MODEL_PATH} | "
        f"Epochs: {results['num_epochs_trained']}/{settings.NUM_EPOCHS} | "
        f"Best val loss: {results['best_val_loss']:.4f} | "
        f"Splits: {sizes['train']}/{sizes['val']}/{sizes['test']}"
    )
