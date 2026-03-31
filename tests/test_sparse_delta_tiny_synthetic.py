from __future__ import annotations

import torch

from src.train.sparse_delta import SparseDeltaModule
from src.train.train_sparse_delta import train_sparse_delta_model


def make_realizable_synthetic_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    d_model = 6
    width = 8
    top_k = 2
    true_module = SparseDeltaModule(d_model=d_model, width=width, top_k=top_k)
    with torch.no_grad():
        true_module.encoder.weight.zero_()
        true_module.encoder.bias.zero_()
        true_module.encoder.weight[1, 0] = 2.0
        true_module.encoder.weight[5, 3] = 1.5
        true_module.decoder.weight.zero_()
        true_module.decoder.bias.zero_()
        true_module.decoder.weight[:, 1] = torch.tensor([0.8, 0.0, -0.5, 0.0, 0.1, 0.0])
        true_module.decoder.weight[:, 5] = torch.tensor([0.0, -0.6, 0.0, 0.7, 0.0, 0.2])

    h_pt = torch.randn(192, d_model)
    delta = true_module(h_pt).delta_hat.detach()
    return h_pt, delta


def test_tiny_synthetic_training_recovers_a_useful_delta() -> None:
    h_pt, delta = make_realizable_synthetic_dataset()
    train_h_pt, val_h_pt = h_pt[:144], h_pt[144:]
    train_delta, val_delta = delta[:144], delta[144:]

    result = train_sparse_delta_model(
        train_h_pt=train_h_pt,
        train_delta=train_delta,
        val_h_pt=val_h_pt,
        val_delta=val_delta,
        sparse_config={"width": 8, "top_k": 2},
        training_config={
            "lr": 0.01,
            "weight_decay": 0.0,
            "batch_size": 32,
            "max_epochs": 40,
            "early_stopping_patience": 4,
            "dtype": "float32",
            "eps_std": 1e-6,
            "sanity_panel_size": 24,
        },
        seed=17,
        device="cpu",
    )

    assert result["best_val_mse"] < (result["initial_val_mse"] * 0.2)
    assert result["best_val_r2"] > 0.8
    assert result["sanity"]["mean_distance_after"] < result["sanity"]["mean_distance_before"]
    assert result["sanity"]["mean_distance_reduction"] > 0.0
