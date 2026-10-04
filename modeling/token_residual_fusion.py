"""Token-level FC-anchor residual fusion for pseudo-label adaptation."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class TokenResidualFusion(nn.Module):
    """Fuse a frozen FC logit anchor with a trainable Full residual.

    The output projection is zero initialized and the gate starts closed.  A
    newly attached module therefore returns the anchor logits exactly (up to
    floating point evaluation), while the residual branch can learn from
    direct pseudo-sequence cross entropy.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        rank: int = 32,
        gate_hidden: int = 64,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if d_model < 1 or vocab_size < 1 or rank < 1 or gate_hidden < 1:
            raise ValueError("fusion dimensions must be positive")
        self.delta_down = nn.Linear(d_model, rank, bias=False)
        self.delta_up = nn.Linear(rank, vocab_size, bias=False)
        self.gate_network = nn.Sequential(
            nn.Linear(2 * d_model + 1, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.zeros_(self.delta_up.weight)
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, gate_bias)
        self.initial_gate_bias = float(gate_bias)

    def forward(
        self,
        anchor_logits: torch.Tensor,
        full_logits: torch.Tensor,
        anchor_hidden: torch.Tensor,
        full_hidden: torch.Tensor,
        *,
        external_gate: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return mixed logits, residual gate, and residual logits."""
        if anchor_logits.shape != full_logits.shape:
            raise ValueError("anchor and full logits must have identical shapes")
        if anchor_hidden.shape != full_hidden.shape:
            raise ValueError("anchor and full hidden states must have identical shapes")
        if anchor_hidden.shape[:-1] != anchor_logits.shape[:-1]:
            raise ValueError("hidden and logit token dimensions are inconsistent")
        disagreement = (full_logits - anchor_logits).abs().mean(dim=-1, keepdim=True)
        features = torch.cat((anchor_hidden, full_hidden, disagreement), dim=-1)
        gate = torch.sigmoid(self.gate_network(features))
        if external_gate is not None:
            if external_gate.shape != gate.shape:
                raise ValueError("external gate has an incompatible shape")
            gate = gate * external_gate
        residual = self.delta_up(self.delta_down(full_hidden))
        return anchor_logits + gate * residual, gate, residual

