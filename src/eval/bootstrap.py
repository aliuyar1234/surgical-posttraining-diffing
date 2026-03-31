from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.eval.common import add_recovery_metrics

GENERATION_METRICS = {
    "QA_EM",
    "Math_EM",
    "Format_Pass",
    "Cap",
    "Cap_Recovery",
    "Len",
    "BrevEx",
    "VerbCarry",
    "VerbClose",
    "HarmfulRefusal",
    "BenignRefusal",
    "Policy",
}
TEACHER_FORCED_METRICS = {"KL_ans_to_IT"}


@dataclass(frozen=True)
class GenerationVariantBootstrapData:
    prompt_ids: tuple[str, ...]
    slices: np.ndarray
    passed: np.ndarray
    token_len: np.ndarray
    brevity_excess_tokens: np.ndarray
    refused: np.ndarray


@dataclass(frozen=True)
class TeacherForcedVariantBootstrapData:
    prompt_ids: tuple[str, ...]
    kl_values: np.ndarray


def build_bootstrap_summary(
    *,
    generation_rows: list[dict[str, Any]],
    teacher_forced_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]] | None,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    generation_data = build_generation_bootstrap_data(generation_rows)
    teacher_forced_data = build_teacher_forced_bootstrap_data(teacher_forced_rows)
    canonical_prompt_ids = validate_generation_prompt_alignment(generation_data)
    normalized_comparisons = normalize_bootstrap_comparisons(
        comparisons=comparisons,
        generation_variants=sorted(generation_data),
        teacher_forced_variants=sorted(teacher_forced_data),
    )

    point_generation_metrics = {
        variant: generation_metrics_from_indices(data, np.arange(len(canonical_prompt_ids), dtype=np.int64))
        for variant, data in generation_data.items()
    }
    add_recovery_metrics(point_generation_metrics)
    point_teacher_metrics = {
        variant: {"KL_ans_to_IT": float(np.mean(data.kl_values)) if len(data.kl_values) else 0.0}
        for variant, data in teacher_forced_data.items()
    }

    required_generation_variants = required_variants_for_generation_metrics(normalized_comparisons)
    required_teacher_variants = required_variants_for_teacher_forced_metrics(normalized_comparisons)
    rng = np.random.default_rng(seed)

    comparison_payloads: list[dict[str, Any]] = []
    for comparison in normalized_comparisons:
        delta_draws: dict[str, list[float]] = {metric_name: [] for metric_name in comparison["metrics"]}
        for _ in range(resamples):
            sample_indices = rng.integers(0, len(canonical_prompt_ids), size=len(canonical_prompt_ids), dtype=np.int64)
            sampled_generation_metrics = {
                variant: generation_metrics_from_indices(generation_data[variant], sample_indices)
                for variant in required_generation_variants
            }
            if sampled_generation_metrics:
                add_recovery_metrics(sampled_generation_metrics)
            sampled_teacher_metrics = {
                variant: {"KL_ans_to_IT": float(np.mean(teacher_forced_data[variant].kl_values[sample_indices]))}
                for variant in required_teacher_variants
            }
            for metric_name in comparison["metrics"]:
                delta_draws[metric_name].append(
                    metric_value(
                        metric_name,
                        comparison["variant"],
                        generation_metrics=sampled_generation_metrics,
                        teacher_forced_metrics=sampled_teacher_metrics,
                    )
                    - metric_value(
                        metric_name,
                        comparison["baseline_variant"],
                        generation_metrics=sampled_generation_metrics,
                        teacher_forced_metrics=sampled_teacher_metrics,
                    )
                )

        metrics_payload = {}
        for metric_name in comparison["metrics"]:
            deltas = np.asarray(delta_draws[metric_name], dtype=np.float64)
            lower, upper = np.quantile(deltas, [0.025, 0.975])
            variant_value = metric_value(
                metric_name,
                comparison["variant"],
                generation_metrics=point_generation_metrics,
                teacher_forced_metrics=point_teacher_metrics,
            )
            baseline_value = metric_value(
                metric_name,
                comparison["baseline_variant"],
                generation_metrics=point_generation_metrics,
                teacher_forced_metrics=point_teacher_metrics,
            )
            metrics_payload[metric_name] = {
                "variant_value": variant_value,
                "baseline_value": baseline_value,
                "delta": variant_value - baseline_value,
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "source": metric_source(metric_name),
            }
        comparison_payloads.append(
            {
                "name": comparison["name"],
                "variant": comparison["variant"],
                "baseline_variant": comparison["baseline_variant"],
                "prompt_count": len(canonical_prompt_ids),
                "metrics": metrics_payload,
            }
        )

    return {
        "seed": int(seed),
        "resamples": int(resamples),
        "prompt_count": len(canonical_prompt_ids),
        "comparisons": comparison_payloads,
    }


def build_generation_bootstrap_data(
    generation_rows: list[dict[str, Any]],
) -> dict[str, GenerationVariantBootstrapData]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in generation_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    payload: dict[str, GenerationVariantBootstrapData] = {}
    for variant, rows in by_variant.items():
        ordered = sorted(rows, key=lambda item: str(item["prompt_id"]))
        prompt_ids = tuple(str(row["prompt_id"]) for row in ordered)
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError(f"Duplicate prompt ids found in generation rows for variant {variant}")
        payload[variant] = GenerationVariantBootstrapData(
            prompt_ids=prompt_ids,
            slices=np.asarray([str(row["slice"]) for row in ordered], dtype=object),
            passed=np.asarray([float(row["passed"]) for row in ordered], dtype=np.float64),
            token_len=np.asarray([float(row["token_len"]) for row in ordered], dtype=np.float64),
            brevity_excess_tokens=np.asarray([float(row["brevity_excess_tokens"]) for row in ordered], dtype=np.float64),
            refused=np.asarray([float(row["refused"]) for row in ordered], dtype=np.float64),
        )
    return payload


def build_teacher_forced_bootstrap_data(
    teacher_forced_rows: list[dict[str, Any]],
) -> dict[str, TeacherForcedVariantBootstrapData]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in teacher_forced_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)

    payload: dict[str, TeacherForcedVariantBootstrapData] = {}
    for variant, rows in by_variant.items():
        ordered = sorted(rows, key=lambda item: str(item["prompt_id"]))
        prompt_ids = tuple(str(row["prompt_id"]) for row in ordered)
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError(f"Duplicate prompt ids found in teacher-forced rows for variant {variant}")
        payload[variant] = TeacherForcedVariantBootstrapData(
            prompt_ids=prompt_ids,
            kl_values=np.asarray([float(row["KL_ans_to_IT"]) for row in ordered], dtype=np.float64),
        )
    return payload


def validate_generation_prompt_alignment(
    generation_data: dict[str, GenerationVariantBootstrapData],
) -> tuple[str, ...]:
    if "PT" not in generation_data or "IT_neutral" not in generation_data:
        raise ValueError("Bootstrap requires PT and IT_neutral generation rows")
    canonical_prompt_ids = generation_data["PT"].prompt_ids
    for variant, data in generation_data.items():
        if data.prompt_ids != canonical_prompt_ids:
            raise ValueError(f"Generation prompt ids for variant {variant} do not align with PT")
    return canonical_prompt_ids


def normalize_bootstrap_comparisons(
    *,
    comparisons: list[dict[str, Any]] | None,
    generation_variants: list[str],
    teacher_forced_variants: list[str],
) -> list[dict[str, Any]]:
    raw_comparisons = comparisons if comparisons is not None else default_bootstrap_comparisons(generation_variants)
    generation_variant_set = set(generation_variants)
    teacher_forced_variant_set = set(teacher_forced_variants)
    normalized: list[dict[str, Any]] = []
    for raw in raw_comparisons:
        if not isinstance(raw, dict):
            raise TypeError(f"Bootstrap comparison must be a mapping, got {type(raw).__name__}")
        missing = {"name", "variant", "baseline_variant", "metrics"} - set(raw)
        if missing:
            raise ValueError(f"Bootstrap comparison is missing keys: {sorted(missing)}")
        variant = str(raw["variant"])
        baseline_variant = str(raw["baseline_variant"])
        if variant not in generation_variant_set or baseline_variant not in generation_variant_set:
            raise ValueError(f"Bootstrap comparison references unknown generation variants: {variant}, {baseline_variant}")
        metrics = [str(metric_name) for metric_name in raw["metrics"]]
        for metric_name in metrics:
            if metric_name in GENERATION_METRICS:
                continue
            if metric_name in TEACHER_FORCED_METRICS:
                if variant not in teacher_forced_variant_set or baseline_variant not in teacher_forced_variant_set:
                    raise ValueError(
                        f"Bootstrap KL comparison {raw['name']!r} requires teacher-forced rows for {variant} and {baseline_variant}"
                    )
                continue
            raise ValueError(f"Unsupported bootstrap metric: {metric_name}")
        normalized.append(
            {
                "name": str(raw["name"]),
                "variant": variant,
                "baseline_variant": baseline_variant,
                "metrics": metrics,
            }
        )
    return normalized


def default_bootstrap_comparisons(generation_variants: list[str]) -> list[dict[str, Any]]:
    variant_set = set(generation_variants)
    comparisons: list[dict[str, Any]] = []
    if {"PT", "PT_plus_FullDelta"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "full_delta_vs_pt",
                "variant": "PT_plus_FullDelta",
                "baseline_variant": "PT",
                "metrics": ["KL_ans_to_IT", "Cap", "Cap_Recovery", "Len", "BrevEx", "VerbCarry", "Policy"],
            }
        )
    if {"PT", "PT_plus_CapMask"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "capmask_vs_pt",
                "variant": "PT_plus_CapMask",
                "baseline_variant": "PT",
                "metrics": ["Cap", "Cap_Recovery", "Len", "BrevEx", "VerbCarry", "Policy"],
            }
        )
    if {"PT_plus_CapMask", "PT_plus_RandomMask"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "capmask_vs_random",
                "variant": "PT_plus_CapMask",
                "baseline_variant": "PT_plus_RandomMask",
                "metrics": ["Cap", "Cap_Recovery", "VerbCarry"],
            }
        )
    if {"PT_plus_CapMask", "PT_plus_ActivationMassMask"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "capmask_vs_activation_mass",
                "variant": "PT_plus_CapMask",
                "baseline_variant": "PT_plus_ActivationMassMask",
                "metrics": ["Cap", "Cap_Recovery", "VerbCarry"],
            }
        )
    if {"PT_plus_FullDelta", "PT_plus_FullDelta_minus_VerbosityMask"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "verbosity_subtraction_vs_fulldelta",
                "variant": "PT_plus_FullDelta_minus_VerbosityMask",
                "baseline_variant": "PT_plus_FullDelta",
                "metrics": ["Cap", "Cap_Recovery", "Len", "BrevEx", "VerbCarry"],
            }
        )
    if {"PT_plus_FullDelta", "PT_plus_MeanDiff"}.issubset(variant_set):
        comparisons.append(
            {
                "name": "mean_diff_vs_fulldelta",
                "variant": "PT_plus_MeanDiff",
                "baseline_variant": "PT_plus_FullDelta",
                "metrics": ["KL_ans_to_IT", "Cap", "Cap_Recovery", "VerbCarry"],
            }
        )
    return comparisons


def required_variants_for_generation_metrics(comparisons: list[dict[str, Any]]) -> list[str]:
    required = {"PT", "IT_neutral"}
    for comparison in comparisons:
        if any(metric_name in GENERATION_METRICS for metric_name in comparison["metrics"]):
            required.add(comparison["variant"])
            required.add(comparison["baseline_variant"])
    return sorted(required)


def required_variants_for_teacher_forced_metrics(comparisons: list[dict[str, Any]]) -> list[str]:
    required: set[str] = set()
    for comparison in comparisons:
        if any(metric_name in TEACHER_FORCED_METRICS for metric_name in comparison["metrics"]):
            required.add(comparison["variant"])
            required.add(comparison["baseline_variant"])
    return sorted(required)


def generation_metrics_from_indices(
    data: GenerationVariantBootstrapData,
    sample_indices: np.ndarray,
) -> dict[str, float]:
    slices = data.slices[sample_indices]
    passed = data.passed[sample_indices]
    token_len = data.token_len[sample_indices]
    brevity_excess_tokens = data.brevity_excess_tokens[sample_indices]
    refused = data.refused[sample_indices]
    qa_mask = slices == "QA"
    math_mask = slices == "Math"
    format_mask = slices == "Format"
    brevity_mask = slices == "Brevity"
    harmful_mask = slices == "Harmful"
    benign_mask = slices == "BenignAdjacent"
    length_mask = qa_mask | math_mask | format_mask | brevity_mask

    qa_em = masked_mean(passed, qa_mask)
    math_em = masked_mean(passed, math_mask)
    format_pass = masked_mean(passed, format_mask)
    harmful_refusal = masked_mean(refused, harmful_mask)
    benign_refusal = masked_mean(refused, benign_mask)
    return {
        "QA_EM": qa_em,
        "Math_EM": math_em,
        "Format_Pass": format_pass,
        "Cap": (qa_em + math_em + format_pass) / 3.0,
        "Len": masked_mean(token_len, length_mask),
        "BrevEx": masked_mean(brevity_excess_tokens, brevity_mask),
        "HarmfulRefusal": harmful_refusal,
        "BenignRefusal": benign_refusal,
        "Policy": harmful_refusal - benign_refusal,
    }


def metric_value(
    metric_name: str,
    variant: str,
    *,
    generation_metrics: dict[str, dict[str, float]],
    teacher_forced_metrics: dict[str, dict[str, float]],
) -> float:
    if metric_name in TEACHER_FORCED_METRICS:
        return float(teacher_forced_metrics[variant][metric_name])
    return float(generation_metrics[variant][metric_name])


def metric_source(metric_name: str) -> str:
    if metric_name in TEACHER_FORCED_METRICS:
        return "teacher_forced"
    return "generation"


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(values[mask]))
