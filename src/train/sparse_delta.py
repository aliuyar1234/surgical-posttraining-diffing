from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SparseDeltaOutputs:
    delta_hat: torch.Tensor
    features: torch.Tensor


def standardize_hidden(hidden: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, *, eps: float) -> torch.Tensor:
    return (hidden - mean) / (std + eps)


def topk_sparse_activations(activations: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0:
        return torch.zeros_like(activations)
    if top_k >= activations.shape[-1]:
        return activations
    values, indices = torch.topk(activations, k=top_k, dim=-1)
    sparse = torch.zeros_like(activations)
    sparse.scatter_(-1, indices, values)
    return sparse


def make_feature_mask(width: int, active_features: Iterable[int], *, device: torch.device | str | None = None) -> torch.Tensor:
    mask = torch.zeros(width, dtype=torch.float32, device=device)
    for feature_id in active_features:
        mask[int(feature_id)] = 1.0
    return mask


def compute_r2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    centered_target = target - target.mean(dim=0, keepdim=True)
    denom = torch.sum(centered_target.pow(2)).item()
    if denom <= 0.0:
        return 0.0
    numer = torch.sum((target - prediction).pow(2)).item()
    return 1.0 - (numer / denom)


class SparseDeltaModule(nn.Module):
    def __init__(self, *, d_model: int, width: int, top_k: int) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if top_k <= 0 or top_k > width:
            raise ValueError("top_k must be in [1, width]")
        self.d_model = d_model
        self.width = width
        self.top_k = top_k
        self.encoder = nn.Linear(d_model, width)
        self.decoder = nn.Linear(width, d_model)

    def encode(self, standardized_hidden: torch.Tensor) -> torch.Tensor:
        dense = F.relu(self.encoder(standardized_hidden))
        return topk_sparse_activations(dense, self.top_k)

    def decode(self, features: torch.Tensor, *, include_bias: bool) -> torch.Tensor:
        bias = self.decoder.bias if include_bias else None
        return F.linear(features, self.decoder.weight, bias)

    def forward(self, standardized_hidden: torch.Tensor) -> SparseDeltaOutputs:
        features = self.encode(standardized_hidden)
        delta_hat = self.decode(features, include_bias=True)
        return SparseDeltaOutputs(delta_hat=delta_hat, features=features)

    def masked_decode(self, standardized_hidden: torch.Tensor, mask: torch.Tensor) -> SparseDeltaOutputs:
        outputs = self.forward(standardized_hidden)
        broadcast_mask = _broadcast_mask(mask, outputs.features)
        if torch.all(broadcast_mask == 1):
            masked_delta = outputs.delta_hat
        else:
            masked_delta = self.decode(outputs.features * broadcast_mask, include_bias=False)
        return SparseDeltaOutputs(delta_hat=masked_delta, features=outputs.features * broadcast_mask)

    def decoder_column_norms(self) -> torch.Tensor:
        return torch.linalg.vector_norm(self.decoder.weight.detach(), dim=0)


def _broadcast_mask(mask: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    if mask.shape[-1] != features.shape[-1]:
        raise ValueError(
            f"Mask width {mask.shape[-1]} does not match feature width {features.shape[-1]}"
        )
    if mask.ndim == 1:
        view_shape = [1] * (features.ndim - 1) + [mask.shape[0]]
        return mask.view(*view_shape).to(device=features.device, dtype=features.dtype)
    return mask.to(device=features.device, dtype=features.dtype)
