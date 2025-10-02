"""
mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.

Author: Mudit Bhargava
Date: June 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class mLSTM(nn.Module):
    """
    mLSTM layer implementation.

    This layer applies multiple mLSTM cells in sequence, with optional dropout between layers.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
        num_layers (int): Number of mLSTM layers.
        dropout (float, optional): Dropout probability between layers. Default: 0.0.
    """

    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList([mLSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_seq, hidden_state=None):
        """
        Forward pass of the mLSTM layer.

        Args:
            input_seq (Tensor): Input sequence of shape (batch_size, seq_length, input_size).
            hidden_state (tuple of Tensors, optional): Initial hidden state. Default: None.

        Returns:
            tuple: Output sequence and final hidden state.
        """
        batch_size, seq_length, _ = input_seq.size()
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, input_seq.device)
        
        outputs = []
        for t in range(seq_length):
            x = input_seq[:, t, :]
            for layer_idx, layer in enumerate(self.layers):
                h, C = hidden_state[layer_idx]
                h, C = layer(x, (h, C))
                hidden_state[layer_idx] = (h, C)
                x = self.dropout_layer(h) if layer_idx < self.num_layers - 1 else h
            outputs.append(x)
        
        return torch.stack(outputs, dim=1), hidden_state

    def init_hidden(self, batch_size, device=None):
        """Initialize hidden state for all layers."""
        if device is None:
            device = self.layers[0].weight_ih.device
        return [(torch.zeros(batch_size, self.hidden_size, device=device),
                 torch.zeros(batch_size, self.hidden_size, self.hidden_size, device=device))
                for _ in range(self.num_layers)]

class mLSTMCell(nn.Module):
    """
    mLSTM cell implementation.

    This cell uses a matrix memory state and exponential gating as described in the xLSTM paper.

    Args:
        input_size (int): Size of input features.
        hidden_size (int): Size of hidden state.
    """

    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Gate parameters (input, forget, output)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.randn(3 * hidden_size))
        
        # Query, Key, Value projections
        self.W_q = nn.Linear(input_size, hidden_size, bias=False)
        self.W_k = nn.Linear(input_size, hidden_size, bias=False)
        self.W_v = nn.Linear(input_size, hidden_size, bias=False)
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization with stability improvements."""
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        
        # Initialize biases for gates to negative values for stability
        # This starts with smaller gate values (closer to 0 after exp)
        with torch.no_grad():
            # Input and forget gate biases
            self.bias[:self.hidden_size].fill_(-3.0)  # input gate
            self.bias[self.hidden_size:2*self.hidden_size].fill_(-3.0)  # forget gate
            # Output gate bias
            self.bias[2*self.hidden_size:].fill_(0.0)  # output gate
        
        # Initialize projection weights with smaller scale
        nn.init.xavier_uniform_(self.W_q.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_k.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_v.weight, gain=0.5)

    def forward(self, input, hx):
        """
        Forward pass of the mLSTM cell.

        Args:
            input (Tensor): Input tensor of shape (batch_size, input_size).
            hx (tuple of Tensors): Previous hidden state and cell state (h, C).

        Returns:
            tuple: New hidden state and cell state (h_new, C_new).
        """
        h, C = hx
        
        # Compute gates
        gates = F.linear(input, self.weight_ih, self.bias) + F.linear(h, self.weight_hh)
        i, f, o = gates.chunk(3, 1)
        
        # Stabilized exponential gating with stricter clamping
        i = torch.exp(torch.clamp(i, min=-15, max=8))  # exp(8) ≈ 2981
        f = torch.exp(torch.clamp(f, min=-15, max=8))  
        o = torch.sigmoid(o)
        
        # Compute query, key, value with scaling
        q = self.W_q(input) / (self.hidden_size ** 0.5)  # Scale by sqrt(d)
        k = self.W_k(input) / (self.hidden_size ** 0.5)
        v = self.W_v(input)
        
        # Update cell state: C = f*C + i*(v ⊗ k^T)
        # Using scaled outer product for stability
        outer_product = torch.bmm(v.unsqueeze(2), k.unsqueeze(1))
        
        # Apply gates with stability factor
        C_new = f.unsqueeze(2) * C + i.unsqueeze(2) * outer_product
        
        # Optional: Apply soft normalization to prevent unbounded growth
        # This doesn't strictly normalize but dampens extreme values
        scale = 1.0 / (1.0 + C_new.abs().max().item() / 10.0)
        C_new = C_new * scale
        
        # Compute hidden state: h = o * (q^T @ C)
        h_new = o * torch.bmm(q.unsqueeze(1), C_new).squeeze(1)
        
        return h_new, C_new
