"""
Financial Time Series Forecasting with xLSTM - Multi-Step Prediction

This example demonstrates multi-step forecasting using xLSTM with embeddings.
Predicts multiple future timesteps starting from a specified offset.

Author: Mudit Bhargava
Date: October 2025
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.embeddings_data import calculate_metrics, create_embeddings_dataloaders, load_embeddings_csv
from utils.financial_losses import HybridDirectionalMSE
from xLSTM.model import LearningRateConfig, PerBlockOptimizer, xLSTM


def train_model(
    model,
    train_loader,
    test_loader,
    test_price_info,
    lr_config,
    num_epochs=20,
    device="cpu",
    weight_decay=1e-4,
    use_cosine_annealing=True,
    eta_min=1e-6,
    warmup_epochs=0,
):
    """Train the model with per-block learning rates and cosine annealing."""

    optimizer = PerBlockOptimizer(
        model, torch.optim.Adam, lr_config, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay
    )

    if use_cosine_annealing:
        T_max = num_epochs - warmup_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, T_max=T_max, eta_min=eta_min)
        print(f"Using Cosine Annealing: T_max={T_max}, eta_min={eta_min}, warmup_epochs={warmup_epochs}\n")
    else:
        scheduler = None

    criterion = HybridDirectionalMSE(alpha=0.3, beta=0.5, gamma=0.2)

    print("Starting training...\n")
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Warmup
        if use_cosine_annealing and epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["initial_lr"] * warmup_factor

        model.train()
        total_loss = 0.0
        num_batches = 0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            # Forward pass - predictions shape: (batch, seq_len, output_size)
            predictions, _ = model.predict_last(sequences)

            # Compute loss
            loss = criterion(predictions, targets)

            # Backward and update
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches

        # Step scheduler after warmup
        if use_cosine_annealing and epoch >= warmup_epochs:
            scheduler.step()

        current_lrs = [param_group["lr"] for param_group in optimizer.param_groups]

        # Validation
        if epoch % 5 == 0:
            val_loss = evaluate(model, test_loader, criterion, device)
            print(f"Epoch [{epoch + 1:2d}/{num_epochs}], Train Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}")
            print(f"  LRs: {[f'{lr:.2e}' for lr in current_lrs[:3]]}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "best_xlstm_model.pt")
        else:
            print(f"Epoch [{epoch + 1:2d}/{num_epochs}], Train Loss: {avg_loss:.6f}")
            if epoch % 10 == 0:
                print(f"  LRs: {[f'{lr:.2e}' for lr in current_lrs[:3]]}")

    print("\nTraining completed!")
    return best_val_loss


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            predictions, _ = model.predict_last(sequences)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

    return total_loss / len(data_loader)


def make_predictions(model, test_loader, test_price_info, device, prediction_horizon):
    """Generate multi-step predictions on test set."""
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        batch_idx = 0
        for sequences, targets in test_loader:
            sequences = sequences.to(device)
            predictions, _ = model.predict_last(sequences)

            # Convert predictions back to actual prices
            batch_size = predictions.shape[0]
            for i in range(batch_size):
                data_idx = batch_idx * test_loader.batch_size + i
                if data_idx < len(test_price_info):
                    current_price, actual_future_prices = test_price_info[data_idx]

                    # Predicted relative changes for each step
                    pred_relatives = predictions[i].cpu().numpy()

                    # Convert to predicted prices
                    predicted_prices = [current_price * (1.0 + rel) for rel in pred_relatives]

                    all_predictions.append(predicted_prices)
                    all_actuals.append(actual_future_prices)

            batch_idx += 1

    return np.array(all_predictions), np.array(all_actuals)


def plot_results(predictions, actuals, prediction_horizon, filename="predictions_vs_actual.png"):
    """Plot multi-step predictions vs actuals."""
    n_samples = min(200, len(predictions))

    # Determine subplot layout based on prediction horizon
    n_rows = min(3, prediction_horizon)
    n_cols = 2

    plt.figure(figsize=(15, 5 * n_rows))

    # Plot for each prediction step
    for step in range(min(prediction_horizon, 3)):
        pred_step = predictions[:, step]
        actual_step = actuals[:, step]

        # Time series plot
        plt.subplot(n_rows, n_cols, step * 2 + 1)
        plt.plot(actual_step[:n_samples], label="Actual", alpha=0.7)
        plt.plot(pred_step[:n_samples], label="Predicted", alpha=0.7)
        plt.xlabel("Sample")
        plt.ylabel("Price")
        plt.title(f"Step {step + 1} Prediction vs Actual")
        plt.legend()
        plt.grid(True)

        # Scatter plot
        plt.subplot(n_rows, n_cols, step * 2 + 2)
        plt.scatter(actual_step, pred_step, alpha=0.5)
        min_val = min(actual_step.min(), pred_step.min())
        max_val = max(actual_step.max(), pred_step.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        plt.xlabel("Actual Price")
        plt.ylabel("Predicted Price")
        plt.title(f"Step {step + 1} Scatter Plot")
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nResults plot saved as '{filename}'")


def display_metrics(predictions, actuals, prediction_horizon):
    """Display multi-step prediction metrics."""
    metrics = calculate_metrics(predictions, actuals)

    print("\nOverall Prediction Metrics:")
    print(f"  RMSE: {metrics['overall']['RMSE']:.4f}")
    print(f"  MAE:  {metrics['overall']['MAE']:.4f}")

    print("\nPer-Step Metrics:")
    for step in range(prediction_horizon):
        step_metrics = metrics[f"step_{step + 1}"]
        print(f"  Step {step + 1}:")
        print(f"    RMSE: {step_metrics['RMSE']:.4f}")
        print(f"    MAE:  {step_metrics['MAE']:.4f}")

    # Direction accuracy for multi-step
    print("\nDirection Accuracy (per step):")
    for step in range(prediction_horizon):
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions[:, step]))
            true_direction = np.sign(np.diff(actuals[:, step]))
            direction_accuracy = np.mean(pred_direction == true_direction) * 100
            print(f"  Step {step + 1}: {direction_accuracy:.2f}%")


def train_mode(args):
    """Training mode."""
    print("xLSTM Multi-Step Financial Forecasting with Embeddings")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading {args.data_file}...")
    embeddings, prices = load_embeddings_csv(args.data_file)
    print(f"Loaded {len(prices)} records")

    # Hyperparameters
    input_size = 128
    hidden_size = args.hidden_size
    num_layers = args.num_layers
    num_blocks = args.num_blocks
    output_size = args.prediction_horizon  # Multi-step output
    dropout = args.dropout

    seq_length = args.seq_length
    batch_size = args.batch_size
    num_epochs = args.num_epochs

    lr_config = LearningRateConfig.per_block_type(
        slstm_lr=args.slstm_lr, mlstm_lr=args.mlstm_lr, other_lr=args.other_lr
    )

    train_split = args.train_split

    print("\nTraining configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Prediction offset: {args.prediction_offset}")
    print(f"  Prediction horizon: {args.prediction_horizon} steps")
    print("  Learning rates:")
    print(f"    sLSTM blocks: {args.slstm_lr}")
    print(f"    mLSTM blocks: {args.mlstm_lr}")
    print(f"    Other layers: {args.other_lr}")
    print(f"  Weight decay: {args.weight_decay}")
    if args.use_cosine_annealing:
        print(f"  Cosine Annealing: enabled (eta_min={args.eta_min}, warmup={args.warmup_epochs})")
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # Create dataloaders
    print("Creating sequences...")
    train_loader, test_loader, test_price_info = create_embeddings_dataloaders(
        embeddings,
        prices,
        seq_length,
        train_split,
        batch_size,
        device,
        prediction_offset=args.prediction_offset,
        prediction_horizon=args.prediction_horizon,
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Testing samples: {len(test_loader.dataset)}\n")

    # Create model
    print("=" * 70)
    print("Creating xLSTM model with alternating sLSTM/mLSTM blocks...")
    print("=" * 70)

    model = xLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_blocks=num_blocks,
        output_size=output_size,
        dropout=dropout,
        lstm_type="alternate",
        use_projection=True,
    ).to(device)

    model.print_architecture()
    print()

    # Train
    train_model(
        model,
        train_loader,
        test_loader,
        test_price_info,
        lr_config,
        num_epochs,
        device,
        args.weight_decay,
        use_cosine_annealing=args.use_cosine_annealing,
        eta_min=args.eta_min,
        warmup_epochs=args.warmup_epochs,
    )

    # Save model
    model_path = args.model_path
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_blocks": num_blocks,
                "output_size": output_size,
                "dropout": dropout,
                "prediction_horizon": args.prediction_horizon,
            },
        },
        model_path,
    )
    print(f"\nModel saved to: {model_path}")

    # Make predictions
    print("\nGenerating predictions on test set...")
    predictions, actuals = make_predictions(model, test_loader, test_price_info, device, args.prediction_horizon)

    display_metrics(predictions, actuals, args.prediction_horizon)
    plot_results(predictions, actuals, args.prediction_horizon)


def infer_mode(args):
    """Inference mode."""
    print("xLSTM Multi-Step Inference Mode")
    print("=" * 70)
    print()

    # Load data
    print(f"Loading {args.data_file}...")
    embeddings, prices = load_embeddings_csv(args.data_file)
    print(f"Loaded {len(prices)} records")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # Load model
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    config = checkpoint["config"]

    prediction_horizon = config.get("prediction_horizon", 1)

    model = xLSTM(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_blocks=config["num_blocks"],
        output_size=config["output_size"],
        dropout=config["dropout"],
        lstm_type="alternate",
        use_projection=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded successfully!")
    print(f"Prediction horizon: {prediction_horizon} steps\n")

    # Create dataloaders
    print("Creating sequences...")
    _, test_loader, test_price_info = create_embeddings_dataloaders(
        embeddings,
        prices,
        args.seq_length,
        args.train_split,
        args.batch_size,
        device,
        prediction_offset=args.prediction_offset,
        prediction_horizon=prediction_horizon,
    )

    print(f"Testing samples: {len(test_loader.dataset)}\n")

    # Make predictions
    print("Generating predictions...")
    predictions, actuals = make_predictions(model, test_loader, test_price_info, device, prediction_horizon)

    display_metrics(predictions, actuals, prediction_horizon)
    plot_results(predictions, actuals, prediction_horizon, "predictions_vs_actual_infer.png")


def main():
    parser = argparse.ArgumentParser(description="xLSTM Multi-Step Financial Forecasting")

    # Mode
    parser.add_argument("mode", choices=["train", "infer"], help="Mode: train or infer")

    # Data
    parser.add_argument("--data-file", type=str, default="embeddings_128.csv", help="Path to embeddings CSV file")
    parser.add_argument("--model-path", type=str, default="xlstm_model.pt", help="Path to save/load model")

    # Model hyperparameters
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of layers per block")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of blocks")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Training hyperparameters
    parser.add_argument("--seq-length", type=int, default=20, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio")

    # Multi-step prediction parameters
    parser.add_argument("--prediction-offset", type=int, default=2, help="Days ahead to start prediction")
    parser.add_argument("--prediction-horizon", type=int, default=3, help="Number of steps to predict")

    # Learning rates
    parser.add_argument("--slstm-lr", type=float, default=1e-4, help="Learning rate for sLSTM blocks")
    parser.add_argument("--mlstm-lr", type=float, default=1e-5, help="Learning rate for mLSTM blocks")
    parser.add_argument("--other-lr", type=float, default=1e-4, help="Learning rate for other components")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")

    # Cosine annealing parameters
    parser.add_argument("--use-cosine-annealing", action="store_true", default=True)
    parser.add_argument("--no-cosine-annealing", action="store_false", dest="use_cosine_annealing")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for cosine annealing")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Number of warmup epochs")

    args = parser.parse_args()

    if args.mode == "train":
        train_mode(args)
    elif args.mode == "infer":
        infer_mode(args)


if __name__ == "__main__":
    main()
