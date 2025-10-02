"""
Mixed Block Architecture Demo

This example demonstrates the use of mixed sLSTM and mLSTM blocks in the xLSTM model.
As described in the original paper, mixing these block types can leverage the strengths of both:
- sLSTM: Better for local dependencies and detailed sequence modeling
- mLSTM: Better for long-range dependencies with matrix memory

This script compares different configurations on a dummy language modeling task.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from xLSTM import xLSTM
import time

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, vocab_size, seq_length, num_samples):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_length))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def init_weights(m):
    """Initialize model weights for better stability."""
    if type(m) in [nn.Linear, nn.Embedding]:
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def train_model(model, train_loader, device, num_epochs=3, learning_rate=0.001):
    """Train a model and return the final average loss."""
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        nan_detected = False
        
        for batch in train_loader:
            optimizer.zero_grad()
            input_seq = batch[:, :-1].to(device)
            target_seq = batch[:, 1:].to(device)
            
            output, _ = model(input_seq)
            
            if torch.isnan(output).any():
                print(f"  WARNING: NaN detected in output at epoch {epoch+1}")
                nan_detected = True
                break
            
            output = output.contiguous().view(-1, model.vocab_size)
            target_seq = target_seq.contiguous().view(-1)
            
            loss = criterion(output, target_seq)
            
            if torch.isnan(loss):
                print(f"  WARNING: NaN detected in loss at epoch {epoch+1}")
                nan_detected = True
                break
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if nan_detected:
            print(f"  Training stopped early due to NaN")
            return float('nan')
        
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    vocab_size = 1000
    embedding_size = 128
    hidden_size = 256
    num_layers = 1
    num_blocks = 4
    batch_size = 32
    seq_length = 20
    num_epochs = 3
    learning_rate = 0.001
    
    device = torch.device("mps" if torch.backends.mps.is_available() 
                         else "cuda" if torch.cuda.is_available() 
                         else "cpu")
    print(f"Using device: {device}\n")
    
    train_dataset = DummyDataset(vocab_size, seq_length, 500)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Configuration 1: All sLSTM blocks
    print("="*70)
    print("Configuration 1: All sLSTM blocks")
    print("="*70)
    model_slstm = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                        dropout=0.1, lstm_type="slstm").to(device)
    model_slstm.print_architecture()
    print("\nTraining...")
    start_time = time.time()
    final_loss_slstm = train_model(model_slstm, train_loader, device, num_epochs, learning_rate)
    time_slstm = time.time() - start_time
    print(f"Training time: {time_slstm:.2f}s\n")
    
    # Configuration 2: All mLSTM blocks
    print("="*70)
    print("Configuration 2: All mLSTM blocks")
    print("="*70)
    model_mlstm = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                        dropout=0.1, lstm_type="mlstm").to(device)
    model_mlstm.print_architecture()
    print("\nTraining...")
    start_time = time.time()
    final_loss_mlstm = train_model(model_mlstm, train_loader, device, num_epochs, learning_rate)
    time_mlstm = time.time() - start_time
    print(f"Training time: {time_mlstm:.2f}s\n")
    
    # Configuration 3: Alternating mixed blocks (recommended)
    print("="*70)
    print("Configuration 3: Alternating mixed blocks (sLSTM, mLSTM, sLSTM, mLSTM)")
    print("="*70)
    model_alternate = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                           dropout=0.1, lstm_type="alternate").to(device)
    model_alternate.print_architecture()
    print("\nTraining...")
    start_time = time.time()
    final_loss_alternate = train_model(model_alternate, train_loader, device, num_epochs, learning_rate)
    time_alternate = time.time() - start_time
    print(f"Training time: {time_alternate:.2f}s\n")
    
    # Configuration 4: Custom pattern (sLSTM heavy for local context)
    print("="*70)
    print("Configuration 4: Custom pattern - sLSTM-heavy for local dependencies")
    print("="*70)
    custom_pattern_1 = ["slstm", "slstm", "slstm", "mlstm"]
    model_custom_1 = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                          dropout=0.1, lstm_type=custom_pattern_1).to(device)
    model_custom_1.print_architecture()
    print("\nTraining...")
    start_time = time.time()
    final_loss_custom_1 = train_model(model_custom_1, train_loader, device, num_epochs, learning_rate)
    time_custom_1 = time.time() - start_time
    print(f"Training time: {time_custom_1:.2f}s\n")
    
    # Configuration 5: Custom pattern (mLSTM heavy for long-range)
    print("="*70)
    print("Configuration 5: Custom pattern - mLSTM-heavy for long-range dependencies")
    print("="*70)
    custom_pattern_2 = ["slstm", "mlstm", "mlstm", "mlstm"]
    model_custom_2 = xLSTM(vocab_size, embedding_size, hidden_size, num_layers, num_blocks,
                          dropout=0.1, lstm_type=custom_pattern_2).to(device)
    model_custom_2.print_architecture()
    print("\nTraining...")
    start_time = time.time()
    final_loss_custom_2 = train_model(model_custom_2, train_loader, device, num_epochs, learning_rate)
    time_custom_2 = time.time() - start_time
    print(f"Training time: {time_custom_2:.2f}s\n")
    
    # Summary
    print("="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"Configuration 1 (All sLSTM):           Final Loss: {final_loss_slstm:.4f}, Time: {time_slstm:.2f}s")
    print(f"Configuration 2 (All mLSTM):           Final Loss: {final_loss_mlstm:.4f}, Time: {time_mlstm:.2f}s")
    print(f"Configuration 3 (Alternating):         Final Loss: {final_loss_alternate:.4f}, Time: {time_alternate:.2f}s")
    print(f"Configuration 4 (sLSTM-heavy):         Final Loss: {final_loss_custom_1:.4f}, Time: {time_custom_1:.2f}s")
    print(f"Configuration 5 (mLSTM-heavy):         Final Loss: {final_loss_custom_2:.4f}, Time: {time_custom_2:.2f}s")
    print("="*70)
    
    print("\nKey Insights:")
    print("- sLSTM blocks are generally faster due to simpler memory structure")
    print("- mLSTM blocks provide better capacity for long-range dependencies")
    print("- Mixed architectures can balance speed and modeling capacity")
    print("- The optimal pattern depends on your specific task requirements")
    print("\nNote: Results on dummy data are not meaningful for real performance.")
    print("Use real datasets to properly evaluate different configurations.")

if __name__ == "__main__":
    main()
