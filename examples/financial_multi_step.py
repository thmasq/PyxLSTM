"""
Multi-Step Financial Forecasting with xLSTM

This example demonstrates multi-step ahead forecasting using xLSTM.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xLSTM import xLSTM
from utils.financial_data import generate_synthetic_financial_data, create_financial_dataloaders


def main():
    # Hyperparameters
    sequence_length = 60
    prediction_horizon = 10  # Predict 10 steps ahead
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001
    hidden_size = 128
    num_layers = 2
    num_blocks = 4

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # Generate data
    print("Generating synthetic financial data...")
    full_data = generate_synthetic_financial_data(num_samples=5000, num_features=8, seed=42)

    # Split data
    train_size = int(0.8 * len(full_data))
    train_data = full_data[:train_size]
    test_data = full_data[train_size:]

    # Create dataloaders
    train_loader, _, test_loader, scaler = create_financial_dataloaders(
        train_data=train_data,
        test_data=test_data,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        batch_size=batch_size,
        target_column="price",
    )

    # Get dimensions
    sample_batch, sample_target = next(iter(train_loader))
    input_size = sample_batch.shape[-1]
    output_size = prediction_horizon

    print(f"Input size: {input_size}")
    print(f"Output size: {output_size} (multi-step)\n")

    # Create model
    model = xLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_blocks=num_blocks,
        output_size=output_size,
        dropout=0.2,
        lstm_type="alternate",
    ).to(device)

    model.print_architecture()

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nTraining multi-step forecasting model...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions, _ = model.predict_last(sequences)

            loss = criterion(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    # Evaluate
    print("\nEvaluating on test set...")
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions, _ = model.predict_last(sequences)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics for each horizon
    print("\nMetrics by prediction horizon:")
    for h in range(prediction_horizon):
        mae = np.mean(np.abs(all_predictions[:, h] - all_targets[:, h]))
        print(f"  Horizon {h + 1}: MAE = {mae:.6f}")

    # Plot multi-step predictions
    plt.figure(figsize=(15, 8))

    # Plot first few examples
    num_examples = 5
    for i in range(num_examples):
        plt.subplot(num_examples, 1, i + 1)
        plt.plot(range(prediction_horizon), all_targets[i], "bo-", label="Actual", alpha=0.7)
        plt.plot(range(prediction_horizon), all_predictions[i], "r^-", label="Predicted", alpha=0.7)
        plt.ylabel("Value")
        plt.title(f"Example {i + 1}")
        if i == 0:
            plt.legend()
        plt.grid(True)

    plt.xlabel("Prediction Horizon")
    plt.tight_layout()
    plt.savefig("multi_step_predictions.png", dpi=300)
    print("\nMulti-step predictions plot saved as 'multi_step_predictions.png'")


if __name__ == "__main__":
    main()
