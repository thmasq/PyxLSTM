"""
sLSTM: Scalar Long Short-Term Memory

This module implements the sLSTM (scalar LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The sLSTM extends the traditional LSTM by using exponential gating and a new memory mixing technique,
allowing for improved performance on various sequence modeling tasks.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class sLSTM(nn.Module):
    """
    sLSTM layer implementation.

    This layer applies multiple sLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of sLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(sLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([sLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the sLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, c = hidden_state[layer_idx]
                h, c = layer(x, (h, c))
                hidden_state[layer_idx] = (h, c)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)
        
        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size):
        """Initialize hidden state for all layers."""
        return [(torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device),
                 torch.zeros(batch_size, self.hidden_size, device=self.layers[0].weight_ih.device))
                for _ in range(self.num_layers)]

class sLSTMCell(nn.Module):
    """
    sLSTM cell implementation.

    This cell uses exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        with torch.no_grad():
            # Input and forget gate biases (first 2*hidden_size elements)
            # Start with negative values for exponential gates
            self.bias[:self.hidden_size].fill_(-2.0)  # input gate
            self.bias[self.hidden_size:2*self.hidden_size].fill_(-2.0)  # forget gate
            # Cell gate bias
            self.bias[2*self.hidden_size:3*self.hidden_size].fill_(0.0)  # g gate
            # Output gate bias
            self.bias[3*self.hidden_size:].fill_(0.0)  # output gate

    def forward(self, input, hx):
        """
        Forward pass of the sLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state.

        Returns:
            tuple: New hidden state and cell state.
        """
        h, c = hx
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        
        i, f, g, o = gates.chunk(4, 1)
        
        # Stabilized exponential gating
        # Clamp values before exp to prevent overflow
        i = torch.exp(torch.clamp(i, min=-10, max=10))
        f = torch.exp(torch.clamp(f, min=-10, max=10))
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        # Update cell state
        c_new = f * c + i * g
        
        # Optional: Add cell state normalization for very long sequences
        # Uncomment if you still experience instability
        # c_new = c_new / (torch.norm(c_new, dim=1, keepdim=True) + 1e-8)
        
        # Update hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new
