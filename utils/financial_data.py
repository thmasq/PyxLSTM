"""
Financial Data Utilities

This module provides utilities for loading, preprocessing, and batching
financial time series data for xLSTM models.

Author: Mudit Bhargava
Date: October 2025
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class StandardScaler:
    """Simple standard scaler for financial data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        """Fit the scaler on data."""
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0) + 1e-8
        return self

    def transform(self, data):
        """Transform data using fitted parameters."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before transform")
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        """Fit and transform in one step."""
        return self.fit(data).transform(data)

    def inverse_transform(self, data):
        """Inverse transform data."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        return data * self.std + self.mean


class FinancialDataset(Dataset):
    """Dataset for financial time series data."""

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        prediction_horizon: int = 1,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
        stride: int = 1,
        normalize: bool = True,
        scaler: Optional[StandardScaler] = None,
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride

        # Convert to numpy if pandas
        if isinstance(data, pd.DataFrame):
            self.column_names = data.columns.tolist()
            if target_column is None:
                target_column = self.column_names[0]
            if feature_columns is None:
                feature_columns = self.column_names

            self.target_idx = self.column_names.index(target_column)
            self.feature_indices = [self.column_names.index(col) for col in feature_columns]

            data = data.values
        else:
            self.column_names = None
            self.target_idx = target_column if target_column is not None else 0
            self.feature_indices = feature_columns if feature_columns is not None else list(range(data.shape[1]))

        # Normalize if requested
        if normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                self.data = self.scaler.fit_transform(data)
            else:
                self.scaler = scaler
                self.data = self.scaler.transform(data)
        else:
            self.data = data
            self.scaler = None

        # Create sequences
        self.sequences = []
        self.targets = []

        for i in range(0, len(self.data) - sequence_length - prediction_horizon + 1, stride):
            seq = self.data[i : i + sequence_length][:, self.feature_indices]
            target = self.data[i + sequence_length : i + sequence_length + prediction_horizon, self.target_idx]
            self.sequences.append(seq)
            self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.targets[idx])


def create_financial_dataloaders(
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    test_data: Optional[pd.DataFrame] = None,
    sequence_length: int = 60,
    prediction_horizon: int = 1,
    batch_size: int = 32,
    target_column: Optional[str] = None,
    feature_columns: Optional[List[str]] = None,
    normalize: bool = True,
    num_workers: int = 0,
):
    """Create train, validation, and test dataloaders for financial data."""

    # Create training dataset - it will fit its own scaler
    train_dataset = FinancialDataset(
        train_data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        target_column=target_column,
        feature_columns=feature_columns,
        normalize=normalize,
        scaler=None,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Get the scaler from training dataset
    fitted_scaler = train_dataset.scaler

    val_loader = None
    if val_data is not None:
        val_dataset = FinancialDataset(
            val_data,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_column=target_column,
            feature_columns=feature_columns,
            normalize=normalize,
            scaler=fitted_scaler,
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_loader = None
    if test_data is not None:
        test_dataset = FinancialDataset(
            test_data,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_column=target_column,
            feature_columns=feature_columns,
            normalize=normalize,
            scaler=fitted_scaler,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, fitted_scaler


def generate_synthetic_financial_data(
    num_samples: int = 10000,
    num_features: int = 5,
    trend: bool = True,
    seasonality: bool = True,
    noise_level: float = 0.1,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic financial time series data for testing."""
    if seed is not None:
        np.random.seed(seed)

    t = np.arange(num_samples)
    data = np.zeros((num_samples, num_features))

    for i in range(num_features):
        base = 100 + i * 10

        if trend:
            trend_component = 0.01 * t * (1 + 0.1 * i)
        else:
            trend_component = 0

        if seasonality:
            period = 252 // (i + 1)
            seasonal_component = 5 * np.sin(2 * np.pi * t / period)
        else:
            seasonal_component = 0

        random_walk = np.cumsum(np.random.randn(num_samples) * noise_level)

        data[:, i] = base + trend_component + seasonal_component + random_walk

    columns = [f"feature_{i}" for i in range(num_features)]
    columns[0] = "price"

    return pd.DataFrame(data, columns=columns)


def calculate_technical_indicators(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Calculate common technical indicators for financial data."""
    df = df.copy()

    # Simple Moving Averages
    df["SMA_10"] = df[price_col].rolling(window=10).mean()
    df["SMA_20"] = df[price_col].rolling(window=20).mean()
    df["SMA_50"] = df[price_col].rolling(window=50).mean()

    # Exponential Moving Average
    df["EMA_12"] = df[price_col].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df[price_col].ewm(span=26, adjust=False).mean()

    # MACD
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_middle"] = df[price_col].rolling(window=20).mean()
    bb_std = df[price_col].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + 2 * bb_std
    df["BB_lower"] = df["BB_middle"] - 2 * bb_std

    # Returns
    df["returns"] = df[price_col].pct_change()
    df["log_returns"] = np.log(df[price_col] / df[price_col].shift(1))

    # Volatility
    df["volatility"] = df["returns"].rolling(window=20).std()

    # NEW: Use bfill() and ffill() instead of deprecated fillna method
    df = df.bfill().ffill()

    return df
