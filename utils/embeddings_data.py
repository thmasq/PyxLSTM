"""
Embeddings Data Utilities

This module provides utilities for loading and processing financial embeddings data
from CSV files in the format: trading_date, trading_code, company_name, last_price, emb_0...emb_127

Author: Mudit Bhargava
Date: October 2025
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class EmbeddingsDataset(Dataset):
    """Dataset for financial embeddings data."""

    def __init__(
        self,
        embeddings: np.ndarray,
        prices: np.ndarray,
        sequence_length: int,
        device: Optional[torch.device] = None,
        prediction_offset: int = 1,  # NEW
    ):
        self.embeddings = embeddings
        self.prices = prices
        self.sequence_length = sequence_length
        self.device = device if device is not None else torch.device("cpu")
        self.prediction_offset = prediction_offset

        self.sequences = []
        self.targets = []
        self.price_pairs = []

        # KEY CHANGE: End earlier to accommodate prediction offset
        num_sequences = len(embeddings) - sequence_length - prediction_offset + 1

        for i in range(num_sequences):
            # Input: sequence of embeddings (BEFORE the prediction point)
            seq = embeddings[i : i + sequence_length]

            # Target: predict from END of sequence to offset steps ahead
            current_price = prices[i + sequence_length - 1]
            # Predict offset steps into the future
            target_price = prices[i + sequence_length - 1 + prediction_offset]
            relative_change = (target_price - current_price) / current_price

            self.sequences.append(seq)
            self.targets.append(relative_change)
            self.price_pairs.append((current_price, target_price))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.FloatTensor(self.sequences[idx]).to(self.device)
        target = torch.FloatTensor([self.targets[idx]]).to(self.device)
        return seq, target

    def get_price_pair(self, idx):
        """Get the (current_price, next_price) pair for a sequence."""
        return self.price_pairs[idx]


def load_embeddings_csv(csv_path: str) -> Tuple[List[List[float]], List[float]]:
    """
    Load embeddings from CSV file.

    Expected format:
    trading_date,trading_code,company_name,last_price,emb_0,emb_1,...,emb_127

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (embeddings, prices) where:
        - embeddings: List of embedding vectors (each is list of 128 floats)
        - prices: List of prices
    """
    df = pd.read_csv(csv_path)

    embeddings = []
    prices = []

    # Extract embedding columns (emb_0 through emb_127)
    emb_cols = [f"emb_{i}" for i in range(128)]

    for _, row in df.iterrows():
        # Get embeddings
        emb_vec = [row[col] for col in emb_cols if col in df.columns]

        # Only include if we have all 128 embeddings
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
    prediction_offset: int = 1,  # NEW PARAMETER
) -> Tuple[DataLoader, DataLoader, List[Tuple[float, float]]]:
    """
    Create train and test dataloaders from embeddings.

    Args:
        embeddings: List of embedding vectors
        prices: List of prices
        sequence_length: Length of input sequences
        train_split: Fraction of data to use for training
        batch_size: Batch size for dataloaders
        device: Device to place tensors on

    Returns:
        Tuple of (train_loader, test_loader, test_price_pairs)
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
    )
    test_dataset = EmbeddingsDataset(test_embeddings, test_prices, sequence_length, device)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get test price pairs for evaluation
    test_price_pairs = [test_dataset.get_price_pair(i) for i in range(len(test_dataset))]

    return train_loader, test_loader, test_price_pairs


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calculate regression metrics.

    Args:
        predictions: Predicted values
        actuals: Actual values

    Returns:
        Dictionary with metrics
    """
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    return {"MSE": mse, "RMSE": rmse, "MAE": mae}
