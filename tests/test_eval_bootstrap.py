from __future__ import annotations

import pytest

from src.eval.bootstrap import build_bootstrap_summary, default_bootstrap_comparisons


def _generation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slices = ["QA", "Math", "Format", "Brevity", "Harmful", "BenignAdjacent"]
    for index, slice_name in enumerate(slices):
        prompt_id = f"prompt_{index}"
        rows.append(
            {
                "variant": "PT",
                "prompt_id": prompt_id,
                "split": "test",
                "slice": slice_name,
                "passed": slice_name == "Brevity",
                "token_len": 8,
                "brevity_excess_tokens": 3 if slice_name == "Brevity" else 0,
                "refused": slice_name == "Harmful",
            }
        )
        rows.append(
            {
                "variant": "IT_neutral",
                "prompt_id": prompt_id,
                "split": "test",
                "slice": slice_name,
                "passed": slice_name in {"QA", "Math", "Format", "Brevity", "BenignAdjacent"},
                "token_len": 4,
                "brevity_excess_tokens": 0,
                "refused": slice_name == "Harmful",
            }
        )
        rows.append(
            {
                "variant": "PT_plus_FullDelta",
                "prompt_id": prompt_id,
                "split": "test",
                "slice": slice_name,
                "passed": slice_name in {"QA", "Math", "Format", "Brevity"},
                "token_len": 5,
                "brevity_excess_tokens": 1 if slice_name == "Brevity" else 0,
                "refused": slice_name == "Harmful",
            }
        )
    return rows


def _teacher_forced_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slices = ["QA", "Math", "Format", "Brevity", "Harmful", "BenignAdjacent"]
    for index, slice_name in enumerate(slices):
        prompt_id = f"prompt_{index}"
        rows.append(
            {
                "variant": "PT",
                "prompt_id": prompt_id,
                "split": "test",
                "slice": slice_name,
                "KL_ans_to_IT": 1.0,
                "answer_token_count": 4,
            }
        )
        rows.append(
            {
                "variant": "PT_plus_FullDelta",
                "prompt_id": prompt_id,
                "split": "test",
                "slice": slice_name,
                "KL_ans_to_IT": 0.4,
                "answer_token_count": 4,
            }
        )
    return rows


def test_build_bootstrap_summary_tracks_constant_kl_delta_exactly() -> None:
    payload = build_bootstrap_summary(
        generation_rows=_generation_rows(),
        teacher_forced_rows=_teacher_forced_rows(),
        comparisons=[
            {
                "name": "full_delta_vs_pt_kl",
                "variant": "PT_plus_FullDelta",
                "baseline_variant": "PT",
                "metrics": ["KL_ans_to_IT"],
            }
        ],
        resamples=64,
        seed=20260327,
    )

    metric = payload["comparisons"][0]["metrics"]["KL_ans_to_IT"]
    assert metric["source"] == "teacher_forced"
    assert metric["delta"] == pytest.approx(-0.6)
    assert metric["ci_lower"] == pytest.approx(-0.6)
    assert metric["ci_upper"] == pytest.approx(-0.6)


def test_default_bootstrap_comparisons_cover_main_m6_rows() -> None:
    comparisons = default_bootstrap_comparisons(
        [
            "PT",
            "IT_neutral",
            "PT_plus_FullDelta",
            "PT_plus_CapMask",
            "PT_plus_RandomMask",
            "PT_plus_ActivationMassMask",
            "PT_plus_FullDelta_minus_VerbosityMask",
            "PT_plus_MeanDiff",
        ]
    )

    comparison_names = {row["name"] for row in comparisons}
    assert "full_delta_vs_pt" in comparison_names
    assert "capmask_vs_pt" in comparison_names
    assert "capmask_vs_random" in comparison_names
    assert "capmask_vs_activation_mass" in comparison_names
    assert "verbosity_subtraction_vs_fulldelta" in comparison_names
    assert "mean_diff_vs_fulldelta" in comparison_names
