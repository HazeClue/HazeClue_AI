import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eeg_lstm import EEG_LSTM

def train_and_export():
    data_path = "/home/ameen/ameen/projects/grad/ai_model_unzipped/stew_preprocessed.npz"
    onnx_path = "/home/ameen/ameen/projects/grad/hazeclue-ai/onnx_models/lstm_workload.onnx"
    
    print("Loading data...")
    npz = np.load(data_path, allow_pickle=True)
    X = npz["X"].astype(np.float32)
    y = npz["y"].astype(np.int64)
    
    # Reshape: (n, 14, 256) → (n, 256, 14)  [seq_len, features]
    X = X.transpose(0, 2, 1)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}...")
    
    ds = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    
    model = EEG_LSTM()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    
    model.train()
    epochs = 60
    
    for epoch in range(1, epochs + 1):
        total_loss, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
            
        scheduler.step()
        print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss/n:.4f} | Acc: {correct/n:.4f}")
            
    print("Training complete. Exporting to ONNX...")
    model.eval()
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # Dummy input for ONNX tracing: (Batch, TimeSteps, Channels)
    dummy_input = torch.randn(1, 256, 14).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Model successfully exported to {onnx_path}")

if __name__ == "__main__":
    train_and_export()
