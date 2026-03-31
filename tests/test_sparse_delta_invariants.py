from __future__ import annotations

import torch
from torch import nn

from src.train.intervention import SparseDeltaIntervention, register_sparse_delta_hook
from src.train.sparse_delta import SparseDeltaModule, make_feature_mask


class TinyHookedCausalLM(nn.Module):
    def __init__(self, *, vocab_size: int = 17, d_model: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model)])
        self.activation = nn.Tanh()
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        hidden = self.layers[0](hidden)
        hidden = self.activation(hidden)
        return self.lm_head(hidden)


def make_intervention(mask: torch.Tensor | None, *, alpha: float) -> SparseDeltaIntervention:
    module = SparseDeltaModule(d_model=8, width=6, top_k=2)
    return SparseDeltaIntervention(
        module=module,
        input_mean=torch.zeros(8),
        input_std=torch.ones(8),
        alpha=alpha,
        mask=mask,
        eps_std=1e-6,
    )


def test_empty_mask_leaves_logits_unchanged() -> None:
    torch.manual_seed(0)
    model = TinyHookedCausalLM()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    baseline = model(input_ids)

    intervention = make_intervention(make_feature_mask(6, []), alpha=1.0)
    handle = register_sparse_delta_hook(model.layers[0], intervention)
    try:
        hooked = model(input_ids)
    finally:
        handle.remove()

    assert torch.allclose(hooked, baseline, atol=0.0, rtol=0.0)


def test_zero_gate_leaves_logits_unchanged() -> None:
    torch.manual_seed(1)
    model = TinyHookedCausalLM()
    input_ids = torch.tensor([[4, 3, 2, 1]], dtype=torch.long)
    baseline = model(input_ids)

    intervention = make_intervention(None, alpha=0.0)
    handle = register_sparse_delta_hook(model.layers[0], intervention)
    try:
        hooked = model(input_ids)
    finally:
        handle.remove()

    assert torch.allclose(hooked, baseline, atol=0.0, rtol=0.0)


def test_full_mask_matches_full_delta_path() -> None:
    torch.manual_seed(2)
    input_ids = torch.tensor([[5, 1, 6, 2]], dtype=torch.long)

    model_full = TinyHookedCausalLM()
    model_masked = TinyHookedCausalLM()
    model_masked.load_state_dict(model_full.state_dict())

    module = SparseDeltaModule(d_model=8, width=6, top_k=2)
    full_intervention = SparseDeltaIntervention(
        module=module,
        input_mean=torch.zeros(8),
        input_std=torch.ones(8),
        alpha=1.0,
        mask=None,
        eps_std=1e-6,
    )
    masked_intervention = SparseDeltaIntervention(
        module=module,
        input_mean=torch.zeros(8),
        input_std=torch.ones(8),
        alpha=1.0,
        mask=make_feature_mask(6, range(6)),
        eps_std=1e-6,
    )

    handle_full = register_sparse_delta_hook(model_full.layers[0], full_intervention)
    handle_masked = register_sparse_delta_hook(model_masked.layers[0], masked_intervention)
    try:
        logits_full = model_full(input_ids)
        logits_masked = model_masked(input_ids)
    finally:
        handle_full.remove()
        handle_masked.remove()

    assert torch.allclose(logits_full, logits_masked, atol=1e-6, rtol=0.0)


def test_sparse_module_respects_topk_limit() -> None:
    torch.manual_seed(3)
    module = SparseDeltaModule(d_model=8, width=10, top_k=3)
    outputs = module(torch.randn(5, 8))
    nonzero_counts = (outputs.features > 0).sum(dim=-1)
    assert int(nonzero_counts.max().item()) <= 3
