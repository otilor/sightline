"""Event Transformer — round event sequence model.

Transformer encoder that learns which event sequences correlate with
round wins vs losses. Powers loss analysis and tipping-point detection.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Event type vocabulary
EVENT_TYPES = [
    "PAD", "ROUND_START", "MOVE", "FIRST_BLOOD", "KILL", "TRADE",
    "DEATH", "PLANT", "DEFUSE", "ROUND_END_WIN", "ROUND_END_LOSS",
]
EVENT_TO_IDX = {e: i for i, e in enumerate(EVENT_TYPES)}


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class EventTransformer(nn.Module):
    """Transformer encoder for round event sequences.

    Each event is embedded as: event_type + grid_cell + player_id + timestamp.
    The [CLS] token output is used for round outcome classification.

    Input: (batch, seq_len) event token indices
    Output: (batch, 2) — win/loss probability
    """

    def __init__(
        self,
        vocab_size: int = len(EVENT_TYPES),
        grid_cells: int = 25,
        max_players: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        max_seq_len: int = 100,
    ):
        super().__init__()

        self.event_embed = nn.Embedding(vocab_size, d_model // 2)
        self.grid_embed = nn.Embedding(grid_cells + 1, d_model // 4)  # +1 for unknown
        self.player_embed = nn.Embedding(max_players + 1, d_model // 4)  # +1 for unknown
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.projection = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 2)  # win / loss

    def forward(
        self,
        event_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        player_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Embed each component
        e = self.event_embed(event_ids)       # (B, S, d_model//2)
        g = self.grid_embed(grid_ids)         # (B, S, d_model//4)
        p = self.player_embed(player_ids)     # (B, S, d_model//4)

        # Concatenate and project
        combined = torch.cat([e, g, p], dim=-1)  # (B, S, d_model)
        combined = self.projection(combined)

        # Add positional encoding
        combined = self.pos_encoding(combined)

        # Transformer encode
        encoded = self.encoder(combined, src_key_padding_mask=mask)

        # Use first token ([CLS]-like) for classification
        cls_output = encoded[:, 0, :]

        return self.classifier(cls_output)

    def get_attention_weights(
        self,
        event_ids: torch.Tensor,
        grid_ids: torch.Tensor,
        player_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Extract attention weights for tipping-point analysis.

        Returns attention weights from each layer to identify
        which events the model focuses on most for its prediction.
        """
        e = self.event_embed(event_ids)
        g = self.grid_embed(grid_ids)
        p = self.player_embed(player_ids)
        combined = self.projection(torch.cat([e, g, p], dim=-1))
        combined = self.pos_encoding(combined)

        weights = []
        x = combined
        for layer in self.encoder.layers:
            # Manual forward to capture attention
            attn_out, attn_weights = layer.self_attn(
                x, x, x, need_weights=True
            )
            x = layer.norm1(x + layer.dropout1(attn_out))
            ff_out = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff_out))
            weights.append(attn_weights)

        return weights


def find_tipping_point(
    model: EventTransformer,
    event_sequence: dict[str, torch.Tensor],
) -> int:
    """Identify the tipping-point event in a losing round.

    Feeds progressively longer prefixes of the event sequence
    and finds the event after which the model flips from predicting
    a win to predicting a loss.

    Parameters
    ----------
    model : EventTransformer
        Trained transformer model.
    event_sequence : dict
        Contains 'event_ids', 'grid_ids', 'player_ids' tensors.

    Returns
    -------
    int
        Index of the tipping-point event in the sequence.
    """
    model.eval()
    device = next(model.parameters()).device

    seq_len = event_sequence["event_ids"].size(1)
    prev_pred = None

    for length in range(2, seq_len + 1):
        # Take prefix of length
        prefix = {
            k: v[:, :length].to(device)
            for k, v in event_sequence.items()
        }

        with torch.no_grad():
            logits = model(**prefix)
            probs = torch.softmax(logits, dim=1)
            win_prob = probs[0, 0].item()

        if prev_pred is not None and prev_pred > 0.5 and win_prob <= 0.5:
            # Model flipped from predicting win to loss
            return length - 1

        prev_pred = win_prob

    return seq_len - 1  # Last event is tipping point if no flip found
