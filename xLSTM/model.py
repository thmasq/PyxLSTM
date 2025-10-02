"""
xLSTM: Extended Long Short-Term Memory Model

This module implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM model combines sLSTM and mLSTM blocks in a residual architecture
to achieve state-of-the-art performance on various language modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
from .block import xLSTMBlock

class xLSTM(nn.Module):
    """
    xLSTM model implementation.

    This model uses a combination of sLSTM and mLSTM blocks in a residual architecture.
    Supports mixing different LSTM types across blocks as described in the original paper.

    Args:
        vocab_size (int): Size of the vocabulary.
        embedding_size (int): Size of the token embeddings.
        hidden_size (int): Size of the hidden state in LSTM blocks.
        num_layers (int): Number of LSTM layers in each block.
        num_blocks (int): Number of xLSTM blocks.
        dropout (float, optional): Dropout probability. Default: 0.0.
        bidirectional (bool, optional): If True, use bidirectional LSTM. Default: False.
        lstm_type (str or list, optional): Type of LSTM to use. Can be:
            - A single string ('slstm' or 'mlstm') to use the same type for all blocks
            - A list of strings specifying the type for each block
            - 'mixed' or 'alternate' for automatic alternating pattern (slstm, mlstm, slstm, ...)
            Default: 'slstm'.
    """

    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                 dropout=0.0, bidirectional=False, lstm_type="slstm"):
        super(xLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.block_types = self._parse_lstm_type(lstm_type, num_blocks)
        
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.blocks = nn.ModuleList([
            xLSTMBlock(embedding_size, hidden_size, num_layers, dropout, block_type)
            for block_type in self.block_types
        ])
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def _parse_lstm_type(self, lstm_type, num_blocks):
        """
        Parse the lstm_type argument and return a list of block types.
        
        Args:
            lstm_type (str or list): LSTM type specification
            num_blocks (int): Number of blocks in the model
            
        Returns:
            list: List of LSTM types for each block
        """
        if isinstance(lstm_type, list):
            # Validate list length
            if len(lstm_type) != num_blocks:
                raise ValueError(f"Length of lstm_type list ({len(lstm_type)}) must match num_blocks ({num_blocks})")
            # Validate each type
            for i, lt in enumerate(lstm_type):
                if lt not in ['slstm', 'mlstm']:
                    raise ValueError(f"Invalid LSTM type at index {i}: {lt}. Must be 'slstm' or 'mlstm'")
            return lstm_type
        
        elif isinstance(lstm_type, str):
            if lstm_type in ['mixed', 'alternate']:
                # Alternating pattern: slstm, mlstm, slstm, mlstm, ...
                return ['slstm' if i % 2 == 0 else 'mlstm' for i in range(num_blocks)]
            elif lstm_type in ['slstm', 'mlstm']:
                # Same type for all blocks
                return [lstm_type] * num_blocks
            else:
                raise ValueError(f"Invalid LSTM type: {lstm_type}. Must be 'slstm', 'mlstm', 'mixed', or 'alternate'")
        
        else:
            raise TypeError(f"lstm_type must be a string or list, got {type(lstm_type)}")

    def forward(self, input_seq, hidden_states=None):
        """
        Forward pass of the xLSTM model.

        Args:
            input_seq (Tensor): Input sequence of token indices.
            hidden_states (list of tuples, optional): Initial hidden states for each block. Default: None.

        Returns:
            tuple: Output logits and final hidden states.
        """
        embedded_seq = self.embedding(input_seq)
        
        if hidden_states is None:
            hidden_states = [None] * self.num_blocks
        
        output_seq = embedded_seq
        for i, block in enumerate(self.blocks):
            output_seq, hidden_states[i] = block(output_seq, hidden_states[i])
        
        output_seq = self.output_layer(output_seq)
        return output_seq, hidden_states
    
    def get_block_config(self):
        """
        Get the configuration of block types in the model.
        
        Returns:
            list: List of LSTM types for each block
        """
        return self.block_types.copy()
    
    def print_architecture(self):
        """
        Print a summary of the model architecture showing block types.
        """
        print(f"xLSTM Model Architecture:")
        print(f"  Vocabulary size: {self.vocab_size}")
        print(f"  Embedding size: {self.embedding_size}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Layers per block: {self.num_layers}")
        print(f"  Number of blocks: {self.num_blocks}")
        print(f"  Dropout: {self.dropout}")
        print(f"  Bidirectional: {self.bidirectional}")
        print(f"\nBlock Configuration:")
        for i, block_type in enumerate(self.block_types):
            print(f"    Block {i+1}: {block_type.upper()}")
