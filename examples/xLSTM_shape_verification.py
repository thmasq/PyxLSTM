import torch

from xLSTM import xLSTMBlock
from xLSTM import sLSTM
from xLSTM import mLSTM
from xLSTM import xLSTM

print("="*70)
print("xLSTM Shape Verification - Including Mixed Block Types")
print("="*70)

# Define the model hyperparameters
vocab_size = 10000
embedding_size = 256
hidden_size = 512
num_layers = 2
num_blocks = 4
dropout = 0.1
bidirectional = False

print("\n" + "="*70)
print("Test 1: Model with all sLSTM blocks")
print("="*70)
model_slstm = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                    dropout, bidirectional, lstm_type="slstm")
model_slstm.print_architecture()

# Generate random input sequence
batch_size = 4
seq_length = 10
input_seq = torch.randint(0, vocab_size, (batch_size, seq_length))

# Forward pass
output_seq, hidden_states = model_slstm(input_seq)

print("\nOutput sequence shape:", output_seq.shape)
print("Number of hidden states:", len(hidden_states))
print("Hidden states for first block:")
if hidden_states[0] is not None:
    print(f"  - Number of layers: {len(hidden_states[0])}")
    print(f"  - Hidden state (layer 0) shape: {hidden_states[0][0][0].shape}")
    print(f"  - Cell state (layer 0) shape: {hidden_states[0][0][1].shape}")

print("\n" + "="*70)
print("Test 2: Model with all mLSTM blocks")
print("="*70)
model_mlstm = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                    dropout, bidirectional, lstm_type="mlstm")
model_mlstm.print_architecture()

# Forward pass
output_seq, hidden_states = model_mlstm(input_seq)

print("\nOutput sequence shape:", output_seq.shape)
print("Number of hidden states:", len(hidden_states))
print("Hidden states for first block:")
if hidden_states[0] is not None:
    print(f"  - Number of layers: {len(hidden_states[0])}")
    print(f"  - Hidden state (layer 0) shape: {hidden_states[0][0][0].shape}")
    print(f"  - Cell state (layer 0) shape: {hidden_states[0][0][1].shape}")
    print(f"  - Note: mLSTM cell state is a matrix with shape (batch, hidden, hidden)")

print("\n" + "="*70)
print("Test 3: Model with alternating mixed blocks (slstm, mlstm, slstm, mlstm)")
print("="*70)
model_mixed = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                    dropout, bidirectional, lstm_type="alternate")
model_mixed.print_architecture()

# Forward pass
output_seq, hidden_states = model_mixed(input_seq)

print("\nOutput sequence shape:", output_seq.shape)
print("Number of hidden states:", len(hidden_states))
for i in range(num_blocks):
    block_type = model_mixed.block_types[i]
    print(f"\nBlock {i+1} ({block_type.upper()}) hidden states:")
    if hidden_states[i] is not None:
        print(f"  - Number of layers: {len(hidden_states[i])}")
        print(f"  - Hidden state (layer 0) shape: {hidden_states[i][0][0].shape}")
        print(f"  - Cell state (layer 0) shape: {hidden_states[i][0][1].shape}")

print("\n" + "="*70)
print("Test 4: Model with custom block pattern")
print("="*70)
custom_pattern = ["slstm", "slstm", "mlstm", "mlstm"]
model_custom = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                     dropout, bidirectional, lstm_type=custom_pattern)
model_custom.print_architecture()

# Forward pass
output_seq, hidden_states = model_custom(input_seq)

print("\nOutput sequence shape:", output_seq.shape)
print("Block configuration:", model_custom.get_block_config())

print("\n" + "="*70)
print("Test 5: Individual sLSTM and mLSTM modules")
print("="*70)

# Test the sLSTM module
slstm = sLSTM(embedding_size, hidden_size, num_layers, dropout)
input_seq_slstm = torch.randn(batch_size, seq_length, embedding_size)
output_seq_slstm, hidden_state_slstm = slstm(input_seq_slstm)
print("\nsLSTM module:")
print("  Output sequence shape:", output_seq_slstm.shape)
print("  Number of layers:", len(hidden_state_slstm))
print("  Hidden state (layer 0) shape:", hidden_state_slstm[0][0].shape)
print("  Cell state (layer 0) shape:", hidden_state_slstm[0][1].shape)

# Test the mLSTM module
mlstm = mLSTM(embedding_size, hidden_size, num_layers, dropout)
input_seq_mlstm = torch.randn(batch_size, seq_length, embedding_size)
output_seq_mlstm, hidden_state_mlstm = mlstm(input_seq_mlstm)
print("\nmLSTM module:")
print("  Output sequence shape:", output_seq_mlstm.shape)
print("  Number of layers:", len(hidden_state_mlstm))
print("  Hidden state (layer 0) shape:", hidden_state_mlstm[0][0].shape)
print("  Cell state (layer 0) shape:", hidden_state_mlstm[0][1].shape)
print("  Note: mLSTM cell state is a 3D tensor (batch, hidden, hidden)")

print("\n" + "="*70)
print("All tests completed successfully!")
print("="*70)
