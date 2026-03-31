from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from src.cache.cache_utils import extract_hidden_from_layer_output
from src.train.sparse_delta import SparseDeltaModule, standardize_hidden


class SparseDeltaIntervention:
    def __init__(
        self,
        *,
        module: SparseDeltaModule,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        alpha: float,
        mask: torch.Tensor | None = None,
        eps_std: float = 1e-6,
        position_mask: torch.Tensor | None = None,
        incremental_only: bool = False,
    ) -> None:
        self.module = module
        self.input_mean = input_mean.detach().clone().to(torch.float32)
        self.input_std = input_std.detach().clone().to(torch.float32)
        self.alpha = float(alpha)
        self.mask = None if mask is None else mask.detach().clone().to(torch.float32)
        self.eps_std = float(eps_std)
        self.position_mask = None if position_mask is None else position_mask.detach().clone()
        self.incremental_only = bool(incremental_only)

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return torch.zeros_like(hidden)

        module_device = self.module.encoder.weight.device
        standardized = standardize_hidden(
            hidden.to(torch.float32),
            self.input_mean.to(hidden.device),
            self.input_std.to(hidden.device),
            eps=self.eps_std,
        ).to(device=module_device, dtype=self.module.encoder.weight.dtype)
        if self.mask is None:
            delta_hat = self.module(standardized).delta_hat
        else:
            delta_hat = self.module.masked_decode(standardized, self.mask.to(module_device)).delta_hat
        return (self.alpha * delta_hat).to(device=hidden.device, dtype=hidden.dtype)

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + apply_position_mask(self.delta(hidden), hidden=hidden, position_mask=self.position_mask)

    def hook(self, _module, _inputs, output: Any):
        hidden = extract_hidden_from_layer_output(output)
        if self.incremental_only and hidden.ndim >= 2 and hidden.shape[-2] != 1:
            return output
        updated_hidden = self.apply(hidden)
        return replace_hidden_in_layer_output(output, updated_hidden)


def register_sparse_delta_hook(layer, intervention: SparseDeltaIntervention):
    return layer.register_forward_hook(intervention.hook)


class DenseAdditiveIntervention:
    def __init__(
        self,
        *,
        delta_vector: torch.Tensor,
        alpha: float,
        position_mask: torch.Tensor | None = None,
        incremental_only: bool = False,
    ) -> None:
        self.delta_vector = delta_vector.detach().clone().to(torch.float32)
        self.alpha = float(alpha)
        self.position_mask = None if position_mask is None else position_mask.detach().clone()
        self.incremental_only = bool(incremental_only)

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.alpha == 0.0:
            return torch.zeros_like(hidden)
        vector = self.delta_vector.to(device=hidden.device, dtype=hidden.dtype)
        view_shape = [1] * max(hidden.ndim - 1, 0) + [vector.shape[0]]
        return self.alpha * vector.view(*view_shape).expand_as(hidden)

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + apply_position_mask(self.delta(hidden), hidden=hidden, position_mask=self.position_mask)

    def hook(self, _module, _inputs, output: Any):
        hidden = extract_hidden_from_layer_output(output)
        if self.incremental_only and hidden.ndim >= 2 and hidden.shape[-2] != 1:
            return output
        updated_hidden = self.apply(hidden)
        return replace_hidden_in_layer_output(output, updated_hidden)


class CompositeIntervention:
    def __init__(
        self,
        *,
        interventions: Iterable[Any],
        position_mask: torch.Tensor | None = None,
        incremental_only: bool = False,
    ) -> None:
        self.interventions = list(interventions)
        self.position_mask = None if position_mask is None else position_mask.detach().clone()
        self.incremental_only = bool(incremental_only)

    def delta(self, hidden: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(hidden)
        for intervention in self.interventions:
            total = total + intervention.delta(hidden)
        return total

    def apply(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + apply_position_mask(self.delta(hidden), hidden=hidden, position_mask=self.position_mask)

    def hook(self, _module, _inputs, output: Any):
        hidden = extract_hidden_from_layer_output(output)
        if self.incremental_only and hidden.ndim >= 2 and hidden.shape[-2] != 1:
            return output
        updated_hidden = self.apply(hidden)
        return replace_hidden_in_layer_output(output, updated_hidden)


def apply_position_mask(delta: torch.Tensor, *, hidden: torch.Tensor, position_mask: torch.Tensor | None) -> torch.Tensor:
    if position_mask is None:
        return delta
    mask = position_mask.to(device=hidden.device, dtype=hidden.dtype)
    if mask.ndim == 1:
        view_shape = [1] * max(hidden.ndim - 2, 0) + [mask.shape[0], 1]
        return delta * mask.view(*view_shape)
    if mask.ndim == hidden.ndim - 1:
        return delta * mask.unsqueeze(-1)
    return delta * mask


def replace_hidden_in_layer_output(output: Any, updated_hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (updated_hidden, *output[1:])
    return updated_hidden
