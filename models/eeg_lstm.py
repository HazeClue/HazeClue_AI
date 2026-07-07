import torch
import torch.nn as nn

class EEG_LSTM(nn.Module):
    def __init__(self, n_ch=14, hidden=128, n_layers=2, bidir=True,
                 dropout=0.4, n_classes=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size   = n_ch,
            hidden_size  = hidden,
            num_layers   = n_layers,
            batch_first  = True,
            bidirectional= bidir,
            dropout      = dropout if n_layers > 1 else 0.0,
        )
        d = hidden * 2 if bidir else hidden

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d, 64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, n_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, x):
        # x : (B, T, C)
        out, _ = self.lstm(x)        # (B, T, d)
        out    = out[:, -1, :]       # last time-step  → (B, d)
        return self.head(out)        # (B, n_classes)
