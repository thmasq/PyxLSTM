"""
Custom Loss Functions for Financial Time Series Forecasting

These loss functions emphasize directional accuracy over magnitude accuracy,
which is more valuable for trading decisions.

Author: Mudit Bhargava
Date: October 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionalLoss(nn.Module):
    """
    Loss that heavily penalizes wrong direction predictions.

    Combines MSE for magnitude with a directional penalty term.

    Args:
        alpha: Weight for directional component (0-1). Higher = more emphasis on direction
        beta: Weight for MSE component
    """

    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets):
        # MSE component for magnitude
        mse_loss = F.mse_loss(predictions, targets)

        # Directional component: penalize when signs don't match
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)

        # -1 when signs match, +1 when they don't
        direction_penalty = 1 - pred_sign * target_sign
        direction_loss = direction_penalty.mean()

        return self.beta * mse_loss + self.alpha * direction_loss


class SignWeightedMSE(nn.Module):
    """
    MSE loss with additional penalty when prediction has wrong sign.

    If predicted and actual have opposite signs, multiply loss by penalty factor.

    Args:
        penalty: Multiplier for wrong-direction predictions (e.g., 2.0 = double penalty)
    """

    def __init__(self, penalty=2.0):
        super().__init__()
        self.penalty = penalty

    def forward(self, predictions, targets):
        # Basic squared errors
        squared_errors = (predictions - targets) ** 2

        # Check if signs match
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        signs_match = (pred_sign == target_sign).float()

        # Apply penalty when signs don't match
        weights = torch.where(
            signs_match > 0, torch.ones_like(signs_match), torch.ones_like(signs_match) * self.penalty
        )

        weighted_errors = squared_errors * weights
        return weighted_errors.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss that penalizes overestimation and underestimation differently.

    Useful when one type of error is more costly (e.g., overestimating gains is worse
    than underestimating for risk management).

    Args:
        over_penalty: Weight for overestimation errors
        under_penalty: Weight for underestimation errors
    """

    def __init__(self, over_penalty=1.5, under_penalty=1.0):
        super().__init__()
        self.over_penalty = over_penalty
        self.under_penalty = under_penalty

    def forward(self, predictions, targets):
        errors = predictions - targets

        # Separate over and under predictions
        over_pred = F.relu(errors)  # Positive errors
        under_pred = F.relu(-errors)  # Negative errors (made positive)

        loss = self.over_penalty * over_pred.pow(2).mean() + self.under_penalty * under_pred.pow(2).mean()

        return loss


class SharpeRatioLoss(nn.Module):
    """
    Loss based on Sharpe ratio - optimizes for risk-adjusted returns.

    Negative Sharpe ratio as loss encourages high returns with low volatility.
    This is closer to actual trading objectives.
    """

    def __init__(self, risk_free_rate=0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate

    def forward(self, predictions, targets):
        # Treat predictions as returns
        returns = predictions

        # Calculate mean return and std
        mean_return = returns.mean()
        std_return = returns.std() + 1e-8  # Avoid division by zero

        # Sharpe ratio (higher is better, so negate for loss)
        sharpe = (mean_return - self.risk_free_rate) / std_return

        # Also include MSE to keep predictions grounded
        mse = F.mse_loss(predictions, targets)

        return -sharpe + 0.1 * mse


class ProfitLoss(nn.Module):
    """
    Loss based on actual trading profit/loss.

    Simulates taking long/short positions based on predictions and calculates P&L.
    This directly optimizes for trading performance.

    Args:
        transaction_cost: Cost per trade (as fraction, e.g., 0.001 = 0.1%)
    """

    def __init__(self, transaction_cost=0.001):
        super().__init__()
        self.transaction_cost = transaction_cost

    def forward(self, predictions, targets):
        # Trading signal: 1 if predict positive, -1 if predict negative
        signals = torch.sign(predictions)

        # P&L: signal * actual_return - transaction costs
        pnl = signals * targets - self.transaction_cost * torch.abs(signals)

        # Negative mean P&L as loss (we want to maximize profit)
        return -pnl.mean()


class HybridDirectionalMSE(nn.Module):
    """
    RECOMMENDED: Best balance for financial forecasting.

    Combines:
    1. MSE for magnitude accuracy
    2. Directional penalty for wrong sign
    3. Weighted emphasis on larger moves

    Args:
        alpha: Weight for MSE component
        beta: Weight for directional component
        gamma: Weight for magnitude-scaled directional penalty
    """

    def __init__(self, alpha=0.3, beta=0.5, gamma=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, predictions, targets):
        # 1. MSE for basic magnitude
        mse_loss = F.mse_loss(predictions, targets)

        # 2. Simple directional penalty
        pred_sign = torch.sign(predictions)
        target_sign = torch.sign(targets)
        direction_penalty = (1 - pred_sign * target_sign).mean()

        # 3. Magnitude-weighted directional penalty
        # Wrong direction on big moves should hurt more
        magnitude_weight = torch.abs(targets)
        magnitude_weighted_penalty = ((1 - pred_sign * target_sign) * magnitude_weight).mean()

        total_loss = self.alpha * mse_loss + self.beta * direction_penalty + self.gamma * magnitude_weighted_penalty

        return total_loss


class LogCoshLoss(nn.Module):
    """
    Log-cosh loss: More robust to outliers than MSE, smoother than MAE.

    Good for financial data with occasional extreme values.
    Can be combined with directional component.

    Args:
        directional_weight: Weight for directional component (0 = pure log-cosh)
    """

    def __init__(self, directional_weight=0.3):
        super().__init__()
        self.directional_weight = directional_weight

    def forward(self, predictions, targets):
        errors = predictions - targets
        logcosh = torch.log(torch.cosh(errors))
        magnitude_loss = logcosh.mean()

        if self.directional_weight > 0:
            # Add directional component
            pred_sign = torch.sign(predictions)
            target_sign = torch.sign(targets)
            direction_loss = (1 - pred_sign * target_sign).mean()

            return (1 - self.directional_weight) * magnitude_loss + self.directional_weight * direction_loss

        return magnitude_loss


# Convenience function to get loss by name
def get_loss_function(name, **kwargs):
    """
    Get a loss function by name.

    Args:
        name: Loss function name
        **kwargs: Parameters for the loss function

    Returns:
        Loss function instance
    """
    losses = {
        "mse": nn.MSELoss,
        "directional": DirectionalLoss,
        "sign_weighted": SignWeightedMSE,
        "asymmetric": AsymmetricLoss,
        "sharpe": SharpeRatioLoss,
        "profit": ProfitLoss,
        "hybrid": HybridDirectionalMSE,
        "logcosh": LogCoshLoss,
    }

    if name.lower() not in losses:
        raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")

    return losses[name.lower()](**kwargs)


# Example usage and comparison
if __name__ == "__main__":
    # Simulate some predictions and targets
    predictions = torch.tensor(
        [
            [0.01, -0.02, 0.015],  # Predicted returns for 3 days
            [0.02, 0.01, -0.01],
            [-0.01, 0.03, 0.02],
        ]
    )

    targets = torch.tensor(
        [
            [0.015, 0.02, 0.01],  # Actual returns
            [0.025, -0.01, -0.015],
            [-0.008, 0.025, -0.01],
        ]
    )

    print("Loss Function Comparison")
    print("=" * 50)

    # Test different losses
    losses = {
        "MSE": nn.MSELoss(),
        "Directional": DirectionalLoss(alpha=0.5, beta=0.5),
        "SignWeighted": SignWeightedMSE(penalty=2.0),
        "Hybrid": HybridDirectionalMSE(),
        "LogCosh": LogCoshLoss(directional_weight=0.3),
    }

    for name, loss_fn in losses.items():
        loss_value = loss_fn(predictions, targets)
        print(f"{name:15s}: {loss_value.item():.6f}")
