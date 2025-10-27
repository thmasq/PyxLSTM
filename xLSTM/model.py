"""
xLSTM: Extended Long Short-Term Memory Model for Financial Time Series

This module implements the xLSTM model adapted for financial forecasting.
The model processes continuous financial features and embeddings.

Author: Mudit Bhargava (Adapted for Financial Data)
Date: October 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Type

import torch.nn as nn
from torch.optim import Optimizer

from .block import xLSTMBlock


@dataclass
class LearningRateConfig:
    """Configuration for per-block learning rates."""

    learning_rates: Dict[str, float]

    @classmethod
    def uniform(cls, lr: float):
        """Create uniform learning rate config."""
        return cls(learning_rates={"default": lr})

    @classmethod
    def per_block_type(cls, slstm_lr: float, mlstm_lr: float, other_lr: float):
        """Create learning rate config with different rates for block types."""
        return cls(learning_rates={"slstm": slstm_lr, "mlstm": mlstm_lr, "other": other_lr})


class PerBlockOptimizer:
    """
    Optimizer wrapper that applies different learning rates to different parts of the model.

    This allows for fine-grained control over learning rates for sLSTM blocks, mLSTM blocks,
    and other model components.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_class: Type[Optimizer],
        lr_config: LearningRateConfig,
        **optimizer_kwargs,
    ):
        self.model = model
        self.lr_config = lr_config

        # Group parameters by block type
        param_groups = self._create_param_groups()

        # Create optimizer with per-group learning rates
        self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)

        # Store initial LRs for warmup/scheduling
        for group in self.optimizer.param_groups:
            group["initial_lr"] = group["lr"]

    def _create_param_groups(self) -> List[Dict]:
        """Create parameter groups with appropriate learning rates."""
        param_groups = []

        # Get default LR
        default_lr = self.lr_config.learning_rates.get("default", 1e-4)
        other_lr = self.lr_config.learning_rates.get("other", default_lr)

        # Group parameters by block type
        slstm_params = []
        mlstm_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Check if parameter belongs to a specific block
            if "blocks." in name:
                block_idx = int(name.split("blocks.")[1].split(".")[0])
                block_type = self.model.block_types[block_idx]

                if block_type == "slstm":
                    slstm_params.append(param)
                elif block_type == "mlstm":
                    mlstm_params.append(param)
                else:
                    other_params.append(param)
            else:
                other_params.append(param)

        # Create parameter groups with appropriate LRs
        if slstm_params:
            slstm_lr = self.lr_config.learning_rates.get("slstm", default_lr)
            param_groups.append({"params": slstm_params, "lr": slstm_lr})

        if mlstm_params:
            mlstm_lr = self.lr_config.learning_rates.get("mlstm", default_lr)
            param_groups.append({"params": mlstm_params, "lr": mlstm_lr})

        if other_params:
            param_groups.append({"params": other_params, "lr": other_lr})

        return param_groups

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        """Perform optimization step."""
        return self.optimizer.step(closure)

    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access parameter groups."""
        return self.optimizer.param_groups


class xLSTM(nn.Module):
    """
    xLSTM model for financial time series forecasting.

    This model uses a combination of sLSTM and mLSTM blocks to process
    continuous financial features and produce predictions.

    Args:
        input_size (int): Size of the input features (e.g., OHLCV + technical indicators).
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        output_size (int): Size of the output (e.g., 1 for price prediction, N for multi-step).
        dropout (float, optional): Dropout probability. Default: 0.0.
        bidirectional (bool, optional): If True, use bidirectional LSTM. Default: False.
        lstm_type (str or list, optional): Type of LSTM to use. Can be:
            - A single string ('slstm' or 'mlstm') to use the same type for all blocks
            - A list of strings specifying the type for each block
            - 'mixed' or 'alternate' for automatic alternating pattern
            Default: 'slstm'.
        use_projection (bool, optional): If True, add input projection layer. Default: True.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_blocks,
        output_size,
        dropout=0.0,
        bidirectional=False,
        lstm_type="slstm",
        use_projection=True,
    ):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_projection = use_projection

        self.block_types = self._parse_lstm_type(lstm_type, num_blocks)

        # Input projection layer (optional)
        if use_projection:
            self.input_projection = nn.Sequential(
                nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(dropout)
            )
            block_input_size = hidden_size
        else:
            self.input_projection = None
            block_input_size = input_size

        # xLSTM blocks
        self.blocks = nn.ModuleList(
            [
                xLSTMBlock(block_input_size, hidden_size, num_layers, dropout, block_type)
                for block_type in self.block_types
            ]
        )

        # Output head for prediction
        self.output_head = nn.Sequential(
            nn.Linear(block_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def _parse_lstm_type(self, lstm_type, num_blocks):
        """Parse the lstm_type argument and return a list of block types."""
        if isinstance(lstm_type, list):
            if len(lstm_type) != num_blocks:
                raise ValueError(f"Length of lstm_type list ({len(lstm_type)}) must match num_blocks ({num_blocks})")
            for i, lt in enumerate(lstm_type):
                if lt not in ["slstm", "mlstm"]:
                    raise ValueError(f"Invalid LSTM type at index {i}: {lt}. Must be 'slstm' or 'mlstm'")
            return lstm_type

        elif isinstance(lstm_type, str):
            if lstm_type in ["mixed", "alternate"]:
                return ["slstm" if i % 2 == 0 else "mlstm" for i in range(num_blocks)]
            elif lstm_type in ["slstm", "mlstm"]:
                return [lstm_type] * num_blocks
            else:
                raise ValueError(f"Invalid LSTM type: {lstm_type}. Must be 'slstm', 'mlstm', 'mixed', or 'alternate'")
        else:
            raise TypeError(f"lstm_type must be a string or list, got {type(lstm_type)}")

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_states (list of tuples, optional): Initial hidden states for each block.

        Returns:
            tuple: Output predictions and final hidden states.
                - output: (batch_size, seq_length, output_size) or (batch_size, output_size) if only last step
                - hidden_states: List of hidden states for each block
        """
        # Project input if needed
        if self.input_projection is not None:
            x = self.input_projection(input_seq)
        else:
            x = input_seq

        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks

        # Pass through xLSTM blocks
        for i, block in enumerate(self.blocks):
            x, hidden_states[i] = block(x, hidden_states[i])

        # Generate predictions
        output = self.output_head(x)

        return output, hidden_states

    def predict_last(self, input_seq, hidden_states=None):
        """
        Forward pass that only returns prediction for the last timestep.
        Useful for single-step forecasting.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_states (list of tuples, optional): Initial hidden states for each block.

        Returns:
            tuple: Last timestep prediction and final hidden states.
        """
        output, hidden_states = self.forward(input_seq, hidden_states)
        return output[:, -1, :], hidden_states

    def get_block_config(self):
        """Get the configuration of block types in the model."""
        return self.block_types.copy()

    def print_architecture(self):
        """Print a summary of the model architecture."""
        print("xLSTM Financial Model Architecture:")
        print(f"  Input size: {self.input_size}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Output size: {self.output_size}")
        print(f"  Layers per block: {self.num_layers}")
        print(f"  Number of blocks: {self.num_blocks}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Bidirectional: {self.bidirectional}")
        print(f"  Use input projection: {self.use_projection}")
        print("\nBlock Configuration:")
        for i, block_type in enumerate(self.block_types):
            print(f"    Block {i + 1}: {block_type.upper()}")
        print(f"\nTotal parameters: {sum(p.numel() for p in self.parameters()):,}")
