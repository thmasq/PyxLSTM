"""
Embeddings Data Utilities - Multi-Step Prediction Version

This module provides utilities for loading and processing financial embeddings data
with support for multi-step forecasting.

Author: Mudit Bhargava
Date: October 2025
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingsDataset(Dataset):
    """Dataset for financial embeddings data with multi-step prediction support."""

    def __init__(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        sequence_length: int,
        device: Optional[torch.device] = None,
        prediction_offset: int = 1,
        prediction_horizon: int = 1,  # NEW: number of steps to predict
    ):
        self.embeddings = embeddings
        self.prices = prices
        self.sequence_length = sequence_length
        self.device = device if device is not None else torch.device("cpu")
        self.prediction_offset = prediction_offset
        self.prediction_horizon = prediction_horizon

        self.sequences = []
        self.targets = []
        self.price_info = []  # Store (current_price, [future_prices])

        # Need enough data for sequence + offset + horizon
        num_sequences = len(embeddings) - sequence_length - prediction_offset - prediction_horizon + 1

        for i in range(num_sequences):
            # Input: sequence of embeddings
            seq = embeddings[i : i + sequence_length]

            # Current price (last price in sequence)
            current_price = prices[i + sequence_length - 1]

            # Target: predict multiple steps starting from offset
            target_prices = []
            relative_changes = []

            for step in range(prediction_horizon):
                target_idx = i + sequence_length - 1 + prediction_offset + step
                target_price = prices[target_idx]
                relative_change = (target_price - current_price) / current_price

                target_prices.append(target_price)
                relative_changes.append(relative_change)

            self.sequences.append(seq)
            self.targets.append(relative_changes)
            self.price_info.append((current_price, target_prices))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx]).to(self.device)
        target = torch.FloatTensor(self.targets[idx]).to(self.device)
        return seq, target

    def get_price_info(self, idx):
        """Get the (current_price, [future_prices]) for a sequence."""
        return self.price_info[idx]


def load_embeddings_csv(csv_path: str) -> Tuple[List[List[float]], List[float]]:
    """
    Load embeddings from CSV file.

    Expected format:
    trading_date,trading_code,company_name,last_price,emb_0,emb_1,...,emb_127

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (embeddings, prices)
    """
    df = pd.read_csv(csv_path)

    embeddings = []
    prices = []

    # Extract embedding columns (emb_0 through emb_127)
    emb_cols = [f"emb_{i}" for i in range(128)]

    for _, row in df.iterrows():
        emb_vec = [row[col] for col in emb_cols if col in df.columns]

        if len(emb_vec) == 128:
            embeddings.append(emb_vec)
            prices.append(row["last_price"])

    return embeddings, prices


def create_embeddings_dataloaders(
    embeddings: List[List[float]],
    prices: List[float],
    sequence_length: int,
    train_split: float = 0.8,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
    prediction_offset: int = 1,
    prediction_horizon: int = 1,  # NEW PARAMETER
) -> Tuple[DataLoader, DataLoader, List[Tuple[float, List[float]]]]:
    """
    Create train and test dataloaders from embeddings.

    Args:
        embeddings: List of embedding vectors
        prices: List of prices
        sequence_length: Length of input sequences
        train_split: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        device: Device to place tensors on
        prediction_offset: Days ahead to start prediction
        prediction_horizon: Number of days to predict

    Returns:
        Tuple of (train_loader, test_loader, test_price_info)
    """
    embeddings = np.array(embeddings, dtype=np.float32)
    prices = np.array(prices, dtype=np.float32)

    # Split into train and test
    split_idx = int(len(embeddings) * train_split)

    train_embeddings = embeddings[:split_idx]
    train_prices = prices[:split_idx]

    test_embeddings = embeddings[split_idx:]
    test_prices = prices[split_idx:]

    # Create datasets
    train_dataset = EmbeddingsDataset(
        train_embeddings,
        train_prices,
        sequence_length,
        device,
        prediction_offset=prediction_offset,
        prediction_horizon=prediction_horizon,
    )
    test_dataset = EmbeddingsDataset(
        test_embeddings,
        test_prices,
        sequence_length,
        device,
        prediction_offset=prediction_offset,
        prediction_horizon=prediction_horizon,
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get test price info for evaluation
    test_price_info = [test_dataset.get_price_info(i) for i in range(len(test_dataset))]

    return train_loader, test_loader, test_price_info


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calculate regression metrics for multi-step predictions.

    Args:
        predictions: Predicted values of shape (n_samples, n_steps)
        actuals: Actual values of shape (n_samples, n_steps)

    Returns:
        Dictionary with metrics for each step and overall
    """
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    if actuals.ndim == 1:
        actuals = actuals.reshape(-1, 1)

    n_steps = predictions.shape[1]

    metrics = {}

    # Overall metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    metrics["overall"] = {"MSE": mse, "RMSE": rmse, "MAE": mae}

    # Per-step metrics
    for step in range(n_steps):
        pred_step = predictions[:, step]
        actual_step = actuals[:, step]

        mse_step = np.mean((pred_step - actual_step) ** 2)
        rmse_step = np.sqrt(mse_step)
        mae_step = np.mean(np.abs(pred_step - actual_step))

        metrics[f"step_{step + 1}"] = {"MSE": mse_step, "RMSE": rmse_step, "MAE": mae_step}

    return metrics
