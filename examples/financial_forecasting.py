"""
Financial Time Series Forecasting with xLSTM

This example demonstrates how to use xLSTM for financial forecasting tasks.
Includes both synthetic and real-world scenarios with proper evaluation metrics.

Author: Mudit Bhargava
Date: October 2025
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.financial_data import (
    calculate_technical_indicators,
    create_financial_dataloaders,
    generate_synthetic_financial_data,
)
from xLSTM import xLSTM


def calculate_metrics(predictions, targets):
    """Calculate regression metrics."""
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mape = np.mean(np.abs((predictions - targets) / (targets + 1e-8))) * 100

    # Direction accuracy (for financial forecasting)
    pred_direction = np.sign(np.diff(predictions.flatten()))
    true_direction = np.sign(np.diff(targets.flatten()))
    direction_accuracy = np.mean(pred_direction == true_direction) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Direction_Accuracy": direction_accuracy}


def train_epoch(model, train_loader, criterion, optimizer, device, clip_value=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass - get only last prediction
        predictions, _ = model.predict_last(sequences)

        # Handle multi-step prediction
        if predictions.dim() == 2 and targets.dim() == 1:
            targets = targets.unsqueeze(-1)

        loss = criterion(predictions, targets)

        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping batch")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions, _ = model.predict_last(sequences)

            if predictions.dim() == 2 and targets.dim() == 1:
                targets = targets.unsqueeze(-1)

            loss = criterion(predictions, targets)
            total_loss += loss.item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    avg_loss = total_loss / len(data_loader)
    metrics = calculate_metrics(all_predictions, all_targets)

    return avg_loss, metrics, all_predictions, all_targets


def main():
    # Hyperparameters
    sequence_length = 60  # 60 time steps
    prediction_horizon = 1  # Predict next step
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0001
    hidden_size = 128
    num_layers = 2
    num_blocks = 4
    dropout = 0.2

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # Generate synthetic financial data
    print("Generating synthetic financial data...")
    full_data = generate_synthetic_financial_data(
        num_samples=10000, num_features=5, trend=True, seasonality=True, noise_level=0.5, seed=42
    )

    # Add technical indicators
    print("Calculating technical indicators...")
    full_data = calculate_technical_indicators(full_data, price_col="price")

    # Remove columns with NaN
    full_data = full_data.dropna()

    print(f"Data shape: {full_data.shape}")
    print(f"Features: {full_data.columns.tolist()}\n")

    # Split data: 70% train, 15% val, 15% test
    train_size = int(0.7 * len(full_data))
    val_size = int(0.15 * len(full_data))

    train_data = full_data[:train_size]
    val_data = full_data[train_size : train_size + val_size]
    test_data = full_data[train_size + val_size :]

    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}\n")

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler = create_financial_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        target_column="price",
        normalize=True,
    )

    # Get input size from dataset
    sample_batch, _ = next(iter(train_loader))
    input_size = sample_batch.shape[-1]
    output_size = prediction_horizon

    print(f"Input size: {input_size}")
    print(f"Output size: {output_size}\n")

    # Create model with mixed blocks
    print("=" * 70)
    print("Creating xLSTM model with alternating blocks...")
    print("=" * 70)

    model = xLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_blocks=num_blocks,
        output_size=output_size,
        dropout=dropout,
        lstm_type="alternate",  # Mix sLSTM and mLSTM
        use_projection=True,
    ).to(device)

    model.print_architecture()
    print()

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training loop
    print("=" * 70)
    print("Starting Training...")
    print("=" * 70)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Update learning rate
        scheduler.step(val_loss)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  Val MAE: {val_metrics['MAE']:.6f}")
            print(f"  Val RMSE: {val_metrics['RMSE']:.6f}")
            print(f"  Val Direction Acc: {val_metrics['Direction_Accuracy']:.2f}%")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_financial_model.pt")

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}\n")

    # Load best model and evaluate on test set
    print("=" * 70)
    print("Evaluating on Test Set...")
    print("=" * 70)

    model.load_state_dict(torch.load("best_financial_model.pt"))
    test_loss, test_metrics, test_predictions, test_targets = evaluate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test MAE: {test_metrics['MAE']:.6f}")
    print(f"Test RMSE: {test_metrics['RMSE']:.6f}")
    print(f"Test MAPE: {test_metrics['MAPE']:.2f}%")
    print(f"Test Direction Accuracy: {test_metrics['Direction_Accuracy']:.2f}%")

    # Plot results
    plt.figure(figsize=(15, 10))

    # Training curves
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Predictions vs Targets
    plt.subplot(2, 2, 2)
    plt.scatter(test_targets[:500], test_predictions[:500], alpha=0.5)
    plt.plot([test_targets.min(), test_targets.max()], [test_targets.min(), test_targets.max()], "r--", lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predictions vs Actual (Test Set)")
    plt.grid(True)

    # Time series comparison
    plt.subplot(2, 2, 3)
    plt.plot(test_targets[:200], label="Actual", alpha=0.7)
    plt.plot(test_predictions[:200], label="Predicted", alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Time Series Comparison (First 200 samples)")
    plt.legend()
    plt.grid(True)

    # Prediction errors
    plt.subplot(2, 2, 4)
    errors = test_predictions.flatten() - test_targets.flatten()
    plt.hist(errors, bins=50, edgecolor="black")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.title("Distribution of Prediction Errors")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("financial_forecasting_results.png", dpi=300, bbox_inches="tight")
    print("\nResults plot saved as 'financial_forecasting_results.png'")


if __name__ == "__main__":
    main()
