"""Trajectory LSTM — per-player movement prediction model.

2-layer LSTM that takes (x, y, t) position sequences and learns
individual movement patterns for next-position prediction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class TrajectoryDataset(Dataset):
    """Dataset of player trajectories as (x, y) sequences."""

    def __init__(self, trajectories: list[np.ndarray], seq_len: int = 20):
        """
        Parameters
        ----------
        trajectories : list[np.ndarray]
            Each element is shape (T, 2) — a sequence of (x, y) positions.
        seq_len : int
            Fixed sequence length for input windows.
        """
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        for traj in trajectories:
            if len(traj) < seq_len + 1:
                continue
            for i in range(len(traj) - seq_len):
                inp = traj[i: i + seq_len]
                target = traj[i + seq_len]
                self.samples.append((inp.astype(np.float32), target.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inp, target = self.samples[idx]
        return torch.from_numpy(inp), torch.from_numpy(target)


class TrajectoryLSTM(nn.Module):
    """2-layer LSTM for player movement prediction.

    Input: (batch, seq_len, 2)  — sequence of (x, y)
    Output: (batch, 2) — predicted next (x, y)
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last = lstm_out[:, -1, :]
        return self.fc(last)


def train_trajectory_model(
    trajectories: list[np.ndarray],
    epochs: int = 50,
    lr: float = 0.001,
    seq_len: int = 20,
    batch_size: int = 32,
    save_path: str | Path | None = None,
) -> TrajectoryLSTM:
    """Train the trajectory LSTM on player movement data.

    Parameters
    ----------
    trajectories : list[np.ndarray]
        Each element is shape (T, 2) with (x, y) positions.
    epochs : int
        Training epochs.
    lr : float
        Learning rate.
    seq_len : int
        Input sequence length.
    batch_size : int
        Batch size.
    save_path : str | Path, optional
        Path to save trained weights.

    Returns
    -------
    TrajectoryLSTM
        Trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TrajectoryDataset(trajectories, seq_len)
    if len(dataset) == 0:
        logger.warning("No valid trajectory samples (need sequences > %d steps)", seq_len)
        return TrajectoryLSTM()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TrajectoryLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logger.info("Training trajectory LSTM: %d samples, %d epochs", len(dataset), epochs)

    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            logger.info("Epoch %d/%d — loss: %.6f", epoch + 1, epochs, avg_loss)

    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
        logger.info("Trajectory model saved to %s", path)

    return model


def predict_next_position(
    model: TrajectoryLSTM,
    recent_positions: np.ndarray,
) -> np.ndarray:
    """Predict the next (x, y) position from recent history.

    Parameters
    ----------
    model : TrajectoryLSTM
        Trained model.
    recent_positions : np.ndarray
        Shape (seq_len, 2) — recent (x, y) positions.

    Returns
    -------
    np.ndarray
        Predicted (x, y) position.
    """
    device = next(model.parameters()).device
    model.eval()

    inp = torch.from_numpy(recent_positions.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp)

    return pred.cpu().numpy().squeeze()
