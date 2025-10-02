"""
xLSTM: Extended Long Short-Term Memory Model for Financial Time Series

This module implements the xLSTM model adapted for financial forecasting.
The model processes continuous financial features and embeddings.

Author: Mudit Bhargava (Adapted for Financial Data)
Date: October 2025
"""

import torch.nn as nn

from .block import xLSTMBlock


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
