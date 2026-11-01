"""Causal dilated TCN criticality head + streaming runtime detector."""
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from criticality_data import normalize_state


class CausalConv1d(nn.Module):
    """Left-padded 1-D conv: output[t] depends only on input[<=t]."""

    def __init__(self, in_ch, out_ch, kernel=3, dilation=1):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x):                       # x: [B, C, T]
        return self.conv(F.pad(x, (self.pad, 0)))


class CriticalityTCN(nn.Module):
    """[B, in_ch, W] -> [B] logit for the window's last frame."""

    def __init__(self, in_ch=14, hidden=64, levels=3, kernel=3):
        super().__init__()
        layers = []
        c = in_ch
        for l in range(levels):
            layers += [CausalConv1d(c, hidden, kernel, dilation=2 ** l), nn.ReLU()]
            c = hidden
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.tcn(x)              # [B, hidden, W]
        return self.head(h[:, :, -1]).squeeze(-1)   # [B]


class StreamingCriticalityDetector:
    """Rolling-buffer causal runtime API. Feed raw 14-dim states one at a time."""

    def __init__(self, model, state_min, state_max, W, device="cpu"):
        self.model = model.to(device).eval()
        self.state_min = np.asarray(state_min, dtype=np.float64)
        self.state_max = np.asarray(state_max, dtype=np.float64)
        self.W = W
        self.device = device
        self.buffer = deque(maxlen=W)

    def reset(self):
        self.buffer.clear()

    @torch.no_grad()
    def predict(self, state_raw):
        self.buffer.append(np.asarray(state_raw, dtype=np.float64))
        frames = list(self.buffer)
        while len(frames) < self.W:             # left-pad with earliest frame
            frames.insert(0, frames[0])
        window = normalize_state(np.stack(frames), self.state_min, self.state_max)  # [W, C]
        x = torch.tensor(window.T[None], dtype=torch.float32, device=self.device)   # [1, C, W]
        return float(torch.sigmoid(self.model(x)).item())
