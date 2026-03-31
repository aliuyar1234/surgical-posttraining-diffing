from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.common import layer_count_map
from src.analysis.select_feature_masks import (
    build_activation_mass_baseline_members,
    build_prompt_frame,
    build_random_baseline_members,
    forward_select_mask,
)


def test_forward_selection_respects_mask_size_cap() -> None:
    candidate_keys = ["layer_16::feature_0", "layer_16::feature_1", "layer_28::feature_0"]
    score_lookup = {
        "layer_16::feature_0": {"layer": 16, "feature_id": 0},
        "layer_16::feature_1": {"layer": 16, "feature_id": 1},
        "layer_28::feature_0": {"layer": 28, "feature_id": 0},
    }
    target_contributions = {
        "qa_correct_delta": {
            "layer_16::feature_0": np.array([0.6], dtype=np.float64),
            "layer_16::feature_1": np.array([0.5], dtype=np.float64),
            "layer_28::feature_0": np.array([0.4], dtype=np.float64),
        }
    }

    result = forward_select_mask(
        mask_name="capability",
        candidate_keys=candidate_keys,
        score_lookup=score_lookup,
        target_contributions=target_contributions,
        objective=lambda current: float(current["qa_correct_delta"][0]),
        max_size=2,
        min_gain=0.0,
        objective_name="J_cap",
    )

    assert len(result["members"]) == 2
    assert result["members"] == [{"layer": 16, "feature_id": 0}, {"layer": 16, "feature_id": 1}]


def test_baseline_masks_match_reference_layer_distribution() -> None:
    reference_members = [
        {"layer": 16, "feature_id": 1},
        {"layer": 28, "feature_id": 4},
        {"layer": 28, "feature_id": 6},
    ]
    available_rows = [
        {"layer": 16, "feature_id": 1, "mass": 0.9},
        {"layer": 16, "feature_id": 2, "mass": 0.8},
        {"layer": 16, "feature_id": 3, "mass": 0.7},
        {"layer": 28, "feature_id": 4, "mass": 0.95},
        {"layer": 28, "feature_id": 5, "mass": 0.92},
        {"layer": 28, "feature_id": 6, "mass": 0.91},
        {"layer": 28, "feature_id": 7, "mass": 0.90},
        {"layer": 28, "feature_id": 8, "mass": 0.89},
    ]

    random_members = build_random_baseline_members(reference_members=reference_members, available_rows=available_rows, seed=7)
    activation_members = build_activation_mass_baseline_members(
        reference_members=reference_members,
        available_rows=available_rows,
    )

    assert len(random_members) == len(reference_members)
    assert len(activation_members) == len(reference_members)
    assert layer_count_map(random_members) == layer_count_map(reference_members)
    assert layer_count_map(activation_members) == layer_count_map(reference_members)


def test_build_prompt_frame_casts_boolean_outcomes_to_numeric_deltas() -> None:
    candidate_rows = pd.DataFrame(
        [
            {
                "prompt_id": "qa_1",
                "split": "select_train",
                "slice": "QA",
                "candidate_key": "layer_16::feature_0",
                "max_answer": 1.0,
                "mean_answer": 1.0,
                "last_answer": 1.0,
                "mean_contribution_norm": 1.0,
            }
        ]
    )
    generation_examples = [
        {
            "variant": "PT",
            "prompt_id": "qa_1",
            "split": "select_train",
            "slice": "QA",
            "passed": False,
            "token_len": 4,
            "brevity_excess_tokens": 0,
            "refused": False,
        },
        {
            "variant": "PT_plus_FullDelta",
            "prompt_id": "qa_1",
            "split": "select_train",
            "slice": "QA",
            "passed": True,
            "token_len": 3,
            "brevity_excess_tokens": 0,
            "refused": False,
        },
        {
            "variant": "IT_neutral",
            "prompt_id": "qa_1",
            "split": "select_train",
            "slice": "QA",
            "passed": True,
            "token_len": 2,
            "brevity_excess_tokens": 0,
            "refused": False,
        },
    ]

    prompt_frame = build_prompt_frame(candidate_rows, generation_examples)

    assert prompt_frame.loc[0, "pt_passed"] == 0.0
    assert prompt_frame.loc[0, "full_passed"] == 1.0
    assert prompt_frame.loc[0, "qa_correct_delta"] == 1.0
