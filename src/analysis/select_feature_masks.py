from __future__ import annotations

import argparse
import json
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV

from src.analysis.common import (
    FEATURE_SUMMARY_COLUMNS,
    build_mask_artifact_path,
    build_split_audit,
    build_feature_table_run_id,
    candidate_feature_key,
    ensure_no_test_leakage,
    layer_count_map,
    mask_selection_run_id,
    matched_mask_size_cap,
    sorted_mask_members,
    top_fraction_count,
)
from src.common.configs import build_artifact_path, load_yaml_config, save_resolved_config_snapshot
from src.common.jsonl import read_jsonl
from src.common.runmeta import collect_runtime_facts


@dataclass
class SelectorModel:
    target_name: str
    feature_columns: list[str]
    x_scales: dict[str, float]
    scaled_coefficients: dict[str, float]
    y_scale: float
    alpha: float | None
    l1_ratio: float | None
    row_count: int
    constant_target: bool
    slices: list[str]

    def standardized_feature_coefficient(self, candidate_key: str) -> float:
        values = [self.scaled_coefficients.get(f"{candidate_key}::{summary_name}", 0.0) for summary_name in FEATURE_SUMMARY_COLUMNS]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def feature_delta_contribution(self, prompt_frame: pd.DataFrame, candidate_key: str) -> np.ndarray:
        contribution = np.zeros(len(prompt_frame), dtype=np.float64)
        if self.constant_target:
            return contribution
        for summary_name in FEATURE_SUMMARY_COLUMNS:
            column_name = f"{candidate_key}::{summary_name}"
            scale = self.x_scales.get(column_name, 1.0)
            coefficient = self.scaled_coefficients.get(column_name, 0.0)
            if coefficient == 0.0:
                continue
            contribution += (prompt_frame[column_name].to_numpy(dtype=np.float64) / scale) * coefficient * self.y_scale
        return contribution


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit M5 selectors and freeze masks.")
    parser.add_argument("--config", required=True, help="Path to selectors config")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    payload = run_select_feature_masks(config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_select_feature_masks(config: dict[str, Any]) -> dict[str, Any]:
    feature_table_run_id = build_feature_table_run_id(config)
    feature_summary_path = build_artifact_path("selector", "feature_table_summary", feature_table_run_id, ".json")
    candidate_table_path = build_artifact_path("selector", "candidate_feature_table", feature_table_run_id, ".parquet")
    candidate_rows_path = build_artifact_path("selector", "candidate_feature_rows", feature_table_run_id, ".parquet")
    generation_examples_path = build_artifact_path("selector", "selection_generation_examples", feature_table_run_id, ".jsonl")
    generation_metrics_path = build_artifact_path("selector", "selection_generation_metrics", feature_table_run_id, ".json")

    feature_summary = json.loads(feature_summary_path.read_text(encoding="utf-8"))
    candidate_table = pd.read_parquet(candidate_table_path)
    candidate_rows = pd.read_parquet(candidate_rows_path)
    generation_examples = read_jsonl(generation_examples_path)
    generation_metrics = json.loads(generation_metrics_path.read_text(encoding="utf-8"))

    selection_run_id = mask_selection_run_id(config, feature_table_run_id=feature_table_run_id)
    score_table_path = build_artifact_path("selector", "feature_selector_scores", selection_run_id, ".parquet")
    summary_path = build_artifact_path("selector", "mask_selection_summary", selection_run_id, ".json")
    runtime_path = build_artifact_path("runtime", "runtime_report", selection_run_id, ".json")
    snapshot_path = build_artifact_path("config_snapshot", "resolved_config", selection_run_id, ".json")
    save_resolved_config_snapshot(
        {"config": config, "feature_table_run_id": feature_table_run_id, "feature_summary": feature_summary},
        snapshot_path,
    )

    start_time = pd.Timestamp.utcnow()
    prompt_frame = build_prompt_frame(candidate_rows, generation_examples)
    feature_columns = sorted(column for column in prompt_frame.columns if "::" in column)

    split_audit = build_split_audit(
        feature_summary_splits=config["feature_splits"],
        candidate_scoring_splits=["select_train"],
        selector_fit_splits=["select_train"],
        forward_selection_splits=["select_tune"],
    )
    ensure_no_test_leakage(split_audit)

    selector_models = fit_selector_models(
        prompt_frame=prompt_frame,
        feature_columns=feature_columns,
        selector_model_config=config["selector_model"],
    )
    score_rows = build_feature_score_rows(candidate_table, selector_models)
    score_frame = pd.DataFrame(score_rows).sort_values(["rank_candidate_score", "layer", "feature_id"], kind="stable")
    score_frame.to_parquet(score_table_path, index=False)

    tune_frame = prompt_frame[prompt_frame["split"] == "select_tune"].reset_index(drop=True)
    tune_context = build_tune_context(tune_frame=tune_frame, generation_metrics=generation_metrics["select_tune"])
    target_contributions = precompute_target_contributions(
        prompt_frame=tune_frame,
        selector_models=selector_models,
        candidate_keys=score_frame["candidate_key"].tolist(),
    )

    score_lookup = {row["candidate_key"]: row for row in score_rows}
    capability_core = build_capability_core(score_rows)
    verbosity_core = build_verbosity_core(score_rows)
    refusal_core = build_refusal_core(score_rows)
    refusal_signal_nontrivial = bool(refusal_core) and max(score_lookup[key]["P_j"] for key in refusal_core) >= float(
        config["selector_model"]["refusal_signal_threshold"]
    )

    capability_result = forward_select_mask(
        mask_name="capability",
        candidate_keys=capability_core,
        score_lookup=score_lookup,
        target_contributions=target_contributions,
        objective=lambda current: capability_objective(current, tune_context),
        max_size=matched_mask_size_cap("capability", config["mask_caps"]),
        min_gain=float(config["forward_selection"]["min_gain"]),
        objective_name="J_cap",
    )
    verbosity_result = forward_select_mask(
        mask_name="verbosity",
        candidate_keys=verbosity_core,
        score_lookup=score_lookup,
        target_contributions=target_contributions,
        objective=lambda current: verbosity_objective(current, tune_context),
        max_size=matched_mask_size_cap("verbosity", config["mask_caps"]),
        min_gain=float(config["forward_selection"]["min_gain"]),
        objective_name="J_vsub",
    )
    refusal_result = None
    if refusal_signal_nontrivial:
        refusal_result = forward_select_mask(
            mask_name="refusal",
            candidate_keys=refusal_core,
            score_lookup=score_lookup,
            target_contributions=target_contributions,
            objective=lambda current: refusal_objective(current, tune_context),
            max_size=matched_mask_size_cap("refusal", config["mask_caps"]),
            min_gain=float(config["forward_selection"]["min_gain"]),
            objective_name="J_ref",
        )

    primary_mask_payloads = build_primary_mask_payloads(
        capability_result=capability_result,
        verbosity_result=verbosity_result,
        refusal_result=refusal_result,
        selection_run_id=selection_run_id,
        feature_table_run_id=feature_table_run_id,
        split_audit=split_audit,
    )
    baseline_mask_payloads = build_baseline_mask_payloads(
        capability_result=capability_result,
        verbosity_result=verbosity_result,
        refusal_result=refusal_result,
        score_rows=score_rows,
        selection_run_id=selection_run_id,
        feature_table_run_id=feature_table_run_id,
        split_audit=split_audit,
        random_seed=int(config["random_baseline_seed"]),
    )

    all_mask_paths: dict[str, str] = {}
    for payload in primary_mask_payloads + baseline_mask_payloads:
        destination = build_mask_artifact_path(payload["mask_name"], selection_run_id=selection_run_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        all_mask_paths[payload["mask_name"]] = destination.as_posix()

    runtime_seconds = (pd.Timestamp.utcnow() - start_time).total_seconds()
    selector_summaries = {
        name: {
            "row_count": model.row_count,
            "constant_target": model.constant_target,
            "alpha": model.alpha,
            "l1_ratio": model.l1_ratio,
            "nonzero_coefficients": int(sum(coef != 0.0 for coef in model.scaled_coefficients.values())),
            "slices": model.slices,
        }
        for name, model in selector_models.items()
    }
    summary_payload = {
        "run_id": selection_run_id,
        "stage_name": "select_feature_masks",
        "feature_table_run_id": feature_table_run_id,
        "feature_table_summary_path": feature_summary_path.as_posix(),
        "feature_selector_score_table_path": score_table_path.as_posix(),
        "split_audit": split_audit,
        "selector_models": selector_summaries,
        "core_sizes": {
            "capability_core": len(capability_core),
            "verbosity_core": len(verbosity_core),
            "refusal_core": len(refusal_core),
        },
        "refusal_signal_nontrivial": refusal_signal_nontrivial,
        "primary_masks": {
            "capability_mask": capability_result,
            "verbosity_mask": verbosity_result,
            "refusal_mask": refusal_result,
        },
        "mask_paths": all_mask_paths,
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
        "tune_objective_denominators": {
            "len_full_minus_pt": generation_metrics["select_tune"]["PT_plus_FullDelta"]["Len"] - generation_metrics["select_tune"]["PT"]["Len"],
            "brev_full_minus_pt": generation_metrics["select_tune"]["PT_plus_FullDelta"]["BrevEx"] - generation_metrics["select_tune"]["PT"]["BrevEx"],
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime_payload = {
        "run_id": selection_run_id,
        "stage_name": "select_feature_masks",
        "feature_table_run_id": feature_table_run_id,
        "device": "cpu",
        "wall_clock_seconds": runtime_seconds,
        "summary_path": summary_path.as_posix(),
        "score_table_path": score_table_path.as_posix(),
        "snapshot_path": snapshot_path.as_posix(),
        "environment": collect_runtime_facts(),
    }
    runtime_path.write_text(json.dumps(runtime_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "run_id": selection_run_id,
        "summary_path": summary_path.as_posix(),
        "runtime_path": runtime_path.as_posix(),
        "score_table_path": score_table_path.as_posix(),
        "mask_paths": all_mask_paths,
    }


def build_prompt_frame(candidate_rows: pd.DataFrame, generation_examples: list[dict[str, Any]]) -> pd.DataFrame:
    feature_long = candidate_rows.melt(
        id_vars=["prompt_id", "split", "slice", "candidate_key"],
        value_vars=list(FEATURE_SUMMARY_COLUMNS),
        var_name="summary_name",
        value_name="value",
    )
    feature_long["feature_column"] = feature_long["candidate_key"] + "::" + feature_long["summary_name"]
    feature_frame = (
        feature_long.pivot_table(
            index=["prompt_id", "split", "slice"],
            columns="feature_column",
            values="value",
            fill_value=0.0,
        )
        .reset_index()
        .rename_axis(columns=None)
    )

    example_frame = pd.DataFrame(generation_examples)
    variant_frames = {}
    for variant_name, prefix in (
        ("PT", "pt"),
        ("PT_plus_FullDelta", "full"),
        ("IT_neutral", "it"),
    ):
        variant_frame = example_frame[example_frame["variant"] == variant_name][
            ["prompt_id", "split", "slice", "passed", "token_len", "brevity_excess_tokens", "refused"]
        ].rename(
            columns={
                "passed": f"{prefix}_passed",
                "token_len": f"{prefix}_token_len",
                "brevity_excess_tokens": f"{prefix}_brevity_excess_tokens",
                "refused": f"{prefix}_refused",
            }
        )
        variant_frames[prefix] = variant_frame

    prompt_frame = feature_frame.copy()
    for frame in variant_frames.values():
        prompt_frame = prompt_frame.merge(frame, on=["prompt_id", "split", "slice"], how="left", validate="one_to_one")

    # Pandas keeps checker outputs as bools; cast them once so downstream delta
    # targets are numeric and survive arithmetic/vectorized modeling cleanly.
    for prefix in ("pt", "full", "it"):
        prompt_frame[f"{prefix}_passed"] = prompt_frame[f"{prefix}_passed"].astype(np.float64)
        prompt_frame[f"{prefix}_refused"] = prompt_frame[f"{prefix}_refused"].astype(np.float64)

    prompt_frame["qa_correct_delta"] = np.where(
        prompt_frame["slice"] == "QA",
        prompt_frame["full_passed"] - prompt_frame["pt_passed"],
        np.nan,
    )
    prompt_frame["math_correct_delta"] = np.where(
        prompt_frame["slice"] == "Math",
        prompt_frame["full_passed"] - prompt_frame["pt_passed"],
        np.nan,
    )
    prompt_frame["format_pass_delta"] = np.where(
        prompt_frame["slice"] == "Format",
        prompt_frame["full_passed"] - prompt_frame["pt_passed"],
        np.nan,
    )
    prompt_frame["harmful_refused_delta"] = np.where(
        prompt_frame["slice"] == "Harmful",
        prompt_frame["full_refused"] - prompt_frame["pt_refused"],
        np.nan,
    )
    prompt_frame["benign_refused_delta"] = np.where(
        prompt_frame["slice"] == "BenignAdjacent",
        prompt_frame["full_refused"] - prompt_frame["pt_refused"],
        np.nan,
    )
    prompt_frame["brevity_excess_tokens_delta"] = np.where(
        prompt_frame["slice"] == "Brevity",
        prompt_frame["full_brevity_excess_tokens"] - prompt_frame["pt_brevity_excess_tokens"],
        np.nan,
    )
    prompt_frame["token_len_delta"] = np.where(
        prompt_frame["slice"].isin(["QA", "Math", "Format", "Brevity"]),
        prompt_frame["full_token_len"] - prompt_frame["pt_token_len"],
        np.nan,
    )
    return prompt_frame.sort_values(["split", "prompt_id"], kind="stable").reset_index(drop=True)


def fit_selector_models(
    *,
    prompt_frame: pd.DataFrame,
    feature_columns: list[str],
    selector_model_config: dict[str, Any],
) -> dict[str, SelectorModel]:
    target_slices = {
        "qa_correct_delta": ["QA"],
        "math_correct_delta": ["Math"],
        "format_pass_delta": ["Format"],
        "harmful_refused_delta": ["Harmful"],
        "benign_refused_delta": ["BenignAdjacent"],
        "brevity_excess_tokens_delta": ["Brevity"],
        "token_len_delta": ["QA", "Math", "Format", "Brevity"],
    }
    models: dict[str, SelectorModel] = {}
    train_frame = prompt_frame[prompt_frame["split"] == "select_train"].reset_index(drop=True)
    for target_name, slices in target_slices.items():
        subset = train_frame[train_frame["slice"].isin(slices)].reset_index(drop=True)
        y = subset[target_name].to_numpy(dtype=np.float64)
        X = subset[feature_columns].to_numpy(dtype=np.float64)
        models[target_name] = fit_one_selector_model(
            target_name=target_name,
            X=X,
            y=y,
            feature_columns=feature_columns,
            config=selector_model_config,
            slices=slices,
        )
    return models


def fit_one_selector_model(
    *,
    target_name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_columns: list[str],
    config: dict[str, Any],
    slices: list[str],
) -> SelectorModel:
    if X.shape[0] == 0:
        return zero_selector_model(target_name=target_name, feature_columns=feature_columns, slices=slices)
    finite_mask = np.isfinite(y)
    X = X[finite_mask]
    y = y[finite_mask]
    if X.shape[0] < 2 or np.allclose(y, y[0]):
        return zero_selector_model(target_name=target_name, feature_columns=feature_columns, slices=slices, row_count=int(X.shape[0]))

    x_scales = X.std(axis=0, ddof=0)
    x_scales = np.where(x_scales < float(config["min_feature_scale"]), 1.0, x_scales)
    y_scale = float(np.std(y, ddof=0))
    if y_scale < float(config["min_target_scale"]):
        return zero_selector_model(target_name=target_name, feature_columns=feature_columns, slices=slices, row_count=int(X.shape[0]))

    X_scaled = X / x_scales
    y_scaled = y / y_scale
    cv_folds = max(2, min(int(config["cv_folds"]), int(X.shape[0])))
    estimator = ElasticNetCV(
        l1_ratio=[float(value) for value in config["l1_ratio_grid"]],
        alphas=[float(value) for value in config["alpha_grid"]],
        fit_intercept=False,
        cv=cv_folds,
        max_iter=int(config["max_iter"]),
        tol=float(config["tol"]),
        selection="cyclic",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        estimator.fit(X_scaled, y_scaled)
    return SelectorModel(
        target_name=target_name,
        feature_columns=feature_columns,
        x_scales={column: float(scale) for column, scale in zip(feature_columns, x_scales, strict=True)},
        scaled_coefficients={column: float(coef) for column, coef in zip(feature_columns, estimator.coef_, strict=True)},
        y_scale=y_scale,
        alpha=float(estimator.alpha_),
        l1_ratio=float(estimator.l1_ratio_),
        row_count=int(X.shape[0]),
        constant_target=False,
        slices=list(slices),
    )


def zero_selector_model(
    *,
    target_name: str,
    feature_columns: list[str],
    slices: list[str],
    row_count: int = 0,
) -> SelectorModel:
    return SelectorModel(
        target_name=target_name,
        feature_columns=feature_columns,
        x_scales={column: 1.0 for column in feature_columns},
        scaled_coefficients={column: 0.0 for column in feature_columns},
        y_scale=1.0,
        alpha=None,
        l1_ratio=None,
        row_count=row_count,
        constant_target=True,
        slices=list(slices),
    )


def build_feature_score_rows(candidate_table: pd.DataFrame, selector_models: dict[str, SelectorModel]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    candidate_rows = candidate_table.sort_values(["candidate_score", "layer", "feature_id"], ascending=[False, True, True], kind="stable")
    for rank, row in enumerate(candidate_rows.itertuples(index=False), start=1):
        candidate_key = str(row.candidate_key)
        qa_coef = selector_models["qa_correct_delta"].standardized_feature_coefficient(candidate_key)
        math_coef = selector_models["math_correct_delta"].standardized_feature_coefficient(candidate_key)
        format_coef = selector_models["format_pass_delta"].standardized_feature_coefficient(candidate_key)
        harmful_coef = selector_models["harmful_refused_delta"].standardized_feature_coefficient(candidate_key)
        benign_coef = selector_models["benign_refused_delta"].standardized_feature_coefficient(candidate_key)
        brevity_coef = selector_models["brevity_excess_tokens_delta"].standardized_feature_coefficient(candidate_key)

        capability_terms = [value for value in (qa_coef, math_coef, format_coef) if value > 0.0]
        c_j = float(sum(capability_terms) / len(capability_terms)) if capability_terms else 0.0
        v_j = float(abs(brevity_coef))
        p_j = float(harmful_coef - benign_coef)
        rows.append(
            {
                "candidate_key": candidate_key,
                "layer": int(row.layer),
                "feature_id": int(row.feature_id),
                "candidate_score": float(row.candidate_score),
                "mass": float(row.mass),
                "slice_variance": float(row.slice_variance),
                "qa_coef": qa_coef,
                "math_coef": math_coef,
                "format_coef": format_coef,
                "harmful_refused_coef": harmful_coef,
                "benign_refused_coef": benign_coef,
                "brevity_excess_coef": brevity_coef,
                "C_j": c_j,
                "V_j": v_j,
                "P_j": p_j,
                "rank_candidate_score": rank,
            }
        )
    return rows


def build_capability_core(
    score_rows: list[dict[str, Any]],
    *,
    capability_fraction: float = 0.15,
    verbosity_exclusion_fraction: float = 0.50,
    refusal_exclusion_fraction: float = 0.50,
) -> list[str]:
    top_capability = rank_feature_keys(
        score_rows,
        metric="C_j",
        fraction=capability_fraction,
        abs_value=False,
        require_positive=True,
    )
    top_verbosity = set(
        rank_feature_keys(score_rows, metric="V_j", fraction=verbosity_exclusion_fraction, abs_value=True)
    )
    top_refusal = set(rank_feature_keys(score_rows, metric="P_j", fraction=refusal_exclusion_fraction, abs_value=True))
    return [key for key in top_capability if key not in top_verbosity and key not in top_refusal]


def build_verbosity_core(
    score_rows: list[dict[str, Any]],
    *,
    verbosity_fraction: float = 0.15,
    capability_exclusion_fraction: float = 0.50,
) -> list[str]:
    top_verbosity = rank_feature_keys(score_rows, metric="V_j", fraction=verbosity_fraction, abs_value=False)
    top_capability = set(
        rank_feature_keys(score_rows, metric="C_j", fraction=capability_exclusion_fraction, abs_value=False)
    )
    return [key for key in top_verbosity if key not in top_capability]


def build_refusal_core(
    score_rows: list[dict[str, Any]],
    *,
    refusal_fraction: float = 0.10,
    capability_exclusion_fraction: float = 0.50,
) -> list[str]:
    top_refusal = rank_feature_keys(
        score_rows,
        metric="P_j",
        fraction=refusal_fraction,
        abs_value=False,
        require_positive=True,
    )
    top_capability = set(
        rank_feature_keys(score_rows, metric="C_j", fraction=capability_exclusion_fraction, abs_value=False)
    )
    return [key for key in top_refusal if key not in top_capability]


def rank_feature_keys(
    score_rows: list[dict[str, Any]],
    *,
    metric: str,
    fraction: float,
    abs_value: bool,
    require_positive: bool = False,
) -> list[str]:
    ranked_rows = []
    for row in score_rows:
        value = float(row[metric])
        if require_positive and value <= 0.0:
            continue
        ranked_rows.append((abs(value) if abs_value else value, row["candidate_key"], row))
    ranked_rows.sort(key=lambda item: (-item[0], item[2]["layer"], item[2]["feature_id"]))
    count = min(len(ranked_rows), top_fraction_count(len(score_rows), fraction))
    return [item[1] for item in ranked_rows[:count]]


def precompute_target_contributions(
    *,
    prompt_frame: pd.DataFrame,
    selector_models: dict[str, SelectorModel],
    candidate_keys: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    contributions: dict[str, dict[str, np.ndarray]] = {}
    for target_name, model in selector_models.items():
        contributions[target_name] = {
            candidate_key: model.feature_delta_contribution(prompt_frame, candidate_key) for candidate_key in candidate_keys
        }
    return contributions


def build_tune_context(*, tune_frame: pd.DataFrame, generation_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    return {
        "frame": tune_frame,
        "pt_metrics": generation_metrics["PT"],
        "it_metrics": generation_metrics["IT_neutral"],
        "full_metrics": generation_metrics["PT_plus_FullDelta"],
        "qa_mask": tune_frame["slice"] == "QA",
        "math_mask": tune_frame["slice"] == "Math",
        "format_mask": tune_frame["slice"] == "Format",
        "brevity_mask": tune_frame["slice"] == "Brevity",
        "harmful_mask": tune_frame["slice"] == "Harmful",
        "benign_mask": tune_frame["slice"] == "BenignAdjacent",
        "length_mask": tune_frame["slice"].isin(["QA", "Math", "Format", "Brevity"]),
        "pt_passed": tune_frame["pt_passed"].to_numpy(dtype=np.float64),
        "full_passed": tune_frame["full_passed"].to_numpy(dtype=np.float64),
        "pt_token_len": tune_frame["pt_token_len"].to_numpy(dtype=np.float64),
        "full_token_len": tune_frame["full_token_len"].to_numpy(dtype=np.float64),
        "pt_brevity_excess_tokens": tune_frame["pt_brevity_excess_tokens"].to_numpy(dtype=np.float64),
        "full_brevity_excess_tokens": tune_frame["full_brevity_excess_tokens"].to_numpy(dtype=np.float64),
        "pt_refused": tune_frame["pt_refused"].to_numpy(dtype=np.float64),
        "full_refused": tune_frame["full_refused"].to_numpy(dtype=np.float64),
    }


def capability_objective(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> float:
    metrics = predicted_capability_variant_metrics(delta_by_target=delta_by_target, context=context)
    benign_penalty = 0.20 * max(0.0, metrics["BenignRefusal"] - context["pt_metrics"]["BenignRefusal"])
    verbosity_penalty = 0.20 * max(0.0, metrics["VerbCarry"] - 0.25)
    return float(metrics["Cap"] - verbosity_penalty - benign_penalty)


def verbosity_objective(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> float:
    metrics = predicted_full_minus_variant_metrics(delta_by_target=delta_by_target, context=context)
    eps = 1e-9
    full_metrics = context["full_metrics"]
    pt_metrics = context["pt_metrics"]
    s_len = (full_metrics["Len"] - metrics["Len"]) / (full_metrics["Len"] - pt_metrics["Len"] + eps)
    s_brev = (full_metrics["BrevEx"] - metrics["BrevEx"]) / (full_metrics["BrevEx"] - pt_metrics["BrevEx"] + eps)
    cap_penalty = 0.25 * max(0.0, full_metrics["Cap"] - metrics["Cap"])
    return float(0.5 * s_len + 0.5 * s_brev - cap_penalty)


def refusal_objective(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> float:
    metrics = predicted_refusal_variant_metrics(delta_by_target=delta_by_target, context=context)
    capability_penalty = 0.25 * max(0.0, context["pt_metrics"]["Cap"] - metrics["Cap"])
    return float(metrics["Policy"] - capability_penalty)


def predicted_capability_variant_metrics(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> dict[str, float]:
    qa = mean_for_mask(context["qa_mask"], np.clip(context["pt_passed"] + delta_by_target["qa_correct_delta"], 0.0, 1.0))
    math_score = mean_for_mask(context["math_mask"], np.clip(context["pt_passed"] + delta_by_target["math_correct_delta"], 0.0, 1.0))
    format_score = mean_for_mask(
        context["format_mask"], np.clip(context["pt_passed"] + delta_by_target["format_pass_delta"], 0.0, 1.0)
    )
    mean_len = mean_for_mask(context["length_mask"], np.clip(context["pt_token_len"] + delta_by_target["token_len_delta"], 0.0, None))
    brevity_excess = mean_for_mask(
        context["brevity_mask"],
        np.clip(context["pt_brevity_excess_tokens"] + delta_by_target["brevity_excess_tokens_delta"], 0.0, None),
    )
    harmful_refusal = mean_for_mask(
        context["harmful_mask"], np.clip(context["pt_refused"] + delta_by_target["harmful_refused_delta"], 0.0, 1.0)
    )
    benign_refusal = mean_for_mask(
        context["benign_mask"], np.clip(context["pt_refused"] + delta_by_target["benign_refused_delta"], 0.0, 1.0)
    )
    cap = float((qa + math_score + format_score) / 3.0)
    return {
        "QA_EM": qa,
        "Math_EM": math_score,
        "Format_Pass": format_score,
        "Cap": cap,
        "Len": mean_len,
        "BrevEx": brevity_excess,
        "HarmfulRefusal": harmful_refusal,
        "BenignRefusal": benign_refusal,
        "Policy": harmful_refusal - benign_refusal,
        "VerbCarry": verb_carry(mean_len, brevity_excess, context["pt_metrics"], context["it_metrics"]),
    }


def predicted_full_minus_variant_metrics(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> dict[str, float]:
    qa = mean_for_mask(context["qa_mask"], np.clip(context["full_passed"] - delta_by_target["qa_correct_delta"], 0.0, 1.0))
    math_score = mean_for_mask(
        context["math_mask"], np.clip(context["full_passed"] - delta_by_target["math_correct_delta"], 0.0, 1.0)
    )
    format_score = mean_for_mask(
        context["format_mask"], np.clip(context["full_passed"] - delta_by_target["format_pass_delta"], 0.0, 1.0)
    )
    mean_len = mean_for_mask(context["length_mask"], np.clip(context["full_token_len"] - delta_by_target["token_len_delta"], 0.0, None))
    brevity_excess = mean_for_mask(
        context["brevity_mask"],
        np.clip(context["full_brevity_excess_tokens"] - delta_by_target["brevity_excess_tokens_delta"], 0.0, None),
    )
    harmful_refusal = mean_for_mask(
        context["harmful_mask"], np.clip(context["full_refused"] - delta_by_target["harmful_refused_delta"], 0.0, 1.0)
    )
    benign_refusal = mean_for_mask(
        context["benign_mask"], np.clip(context["full_refused"] - delta_by_target["benign_refused_delta"], 0.0, 1.0)
    )
    cap = float((qa + math_score + format_score) / 3.0)
    return {
        "QA_EM": qa,
        "Math_EM": math_score,
        "Format_Pass": format_score,
        "Cap": cap,
        "Len": mean_len,
        "BrevEx": brevity_excess,
        "HarmfulRefusal": harmful_refusal,
        "BenignRefusal": benign_refusal,
        "Policy": harmful_refusal - benign_refusal,
    }


def predicted_refusal_variant_metrics(delta_by_target: dict[str, np.ndarray], context: dict[str, Any]) -> dict[str, float]:
    return predicted_capability_variant_metrics(delta_by_target=delta_by_target, context=context)


def verb_carry(predicted_len: float, predicted_brevity_excess: float, pt_metrics: dict[str, float], it_metrics: dict[str, float]) -> float:
    eps = 1e-9
    return float(
        0.5
        * (
            abs(predicted_len - pt_metrics["Len"]) / (abs(it_metrics["Len"] - pt_metrics["Len"]) + eps)
            + abs(predicted_brevity_excess - pt_metrics["BrevEx"]) / (abs(it_metrics["BrevEx"] - pt_metrics["BrevEx"]) + eps)
        )
    )


def mean_for_mask(mask: pd.Series, values: np.ndarray) -> float:
    selected = values[np.asarray(mask.to_numpy(), dtype=bool)]
    if selected.size == 0:
        return 0.0
    return float(selected.mean())


def forward_select_mask(
    *,
    mask_name: str,
    candidate_keys: list[str],
    score_lookup: dict[str, dict[str, Any]],
    target_contributions: dict[str, dict[str, np.ndarray]],
    objective: Callable[[dict[str, np.ndarray]], float],
    max_size: int,
    min_gain: float,
    objective_name: str,
) -> dict[str, Any]:
    selected_keys: list[str] = []
    current_state = zero_delta_state(target_contributions)
    current_score = objective(current_state)
    log: list[dict[str, Any]] = []

    while len(selected_keys) < max_size:
        best_key = None
        best_score = current_score
        best_state = current_state
        for candidate_key in candidate_keys:
            if candidate_key in selected_keys:
                continue
            trial_state = {
                target_name: current_state[target_name] + target_contributions[target_name][candidate_key]
                for target_name in current_state
            }
            trial_score = objective(trial_state)
            if trial_score > best_score + 1e-12:
                best_key = candidate_key
                best_score = trial_score
                best_state = trial_state

        gain = best_score - current_score
        if best_key is None or gain < min_gain:
            break
        selected_keys.append(best_key)
        current_state = best_state
        log.append(
            {
                "step": len(selected_keys),
                "mask_name": mask_name,
                "objective_name": objective_name,
                "candidate_key": best_key,
                "layer": int(score_lookup[best_key]["layer"]),
                "feature_id": int(score_lookup[best_key]["feature_id"]),
                "objective_before": float(current_score),
                "objective_after": float(best_score),
                "gain": float(gain),
            }
        )
        current_score = best_score

    return {
        "mask_name": mask_name,
        "objective_name": objective_name,
        "candidate_count": len(candidate_keys),
        "members": [
            {"layer": int(score_lookup[candidate_key]["layer"]), "feature_id": int(score_lookup[candidate_key]["feature_id"])}
            for candidate_key in selected_keys
        ],
        "selected_keys": selected_keys,
        "objective_score": float(current_score),
        "log": log,
    }


def zero_delta_state(target_contributions: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    sample_target = next(iter(target_contributions))
    sample_key = next(iter(target_contributions[sample_target]))
    width = target_contributions[sample_target][sample_key].shape[0]
    return {target_name: np.zeros(width, dtype=np.float64) for target_name in target_contributions}


def build_primary_mask_payloads(
    *,
    capability_result: dict[str, Any],
    verbosity_result: dict[str, Any],
    refusal_result: dict[str, Any] | None,
    selection_run_id: str,
    feature_table_run_id: str,
    split_audit: dict[str, Any],
) -> list[dict[str, Any]]:
    payloads = [
        build_mask_payload(
            mask_name="capability_mask",
            selection_run_id=selection_run_id,
            feature_table_run_id=feature_table_run_id,
            members=capability_result["members"],
            split_audit=split_audit,
            objective_name="J_cap",
            predicted_objective=capability_result["objective_score"],
            construction_log=capability_result["log"],
            reference_mask_name=None,
            reference_type="primary",
        ),
        build_mask_payload(
            mask_name="verbosity_mask",
            selection_run_id=selection_run_id,
            feature_table_run_id=feature_table_run_id,
            members=verbosity_result["members"],
            split_audit=split_audit,
            objective_name="J_vsub",
            predicted_objective=verbosity_result["objective_score"],
            construction_log=verbosity_result["log"],
            reference_mask_name=None,
            reference_type="primary",
        ),
    ]
    if refusal_result is not None:
        payloads.append(
            build_mask_payload(
                mask_name="refusal_mask",
                selection_run_id=selection_run_id,
                feature_table_run_id=feature_table_run_id,
                members=refusal_result["members"],
                split_audit=split_audit,
                objective_name="J_ref",
                predicted_objective=refusal_result["objective_score"],
                construction_log=refusal_result["log"],
                reference_mask_name=None,
                reference_type="primary",
            )
        )
    return payloads


def build_baseline_mask_payloads(
    *,
    capability_result: dict[str, Any],
    verbosity_result: dict[str, Any],
    refusal_result: dict[str, Any] | None,
    score_rows: list[dict[str, Any]],
    selection_run_id: str,
    feature_table_run_id: str,
    split_audit: dict[str, Any],
    random_seed: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for baseline_name, reference_result in (
        ("capability", capability_result),
        ("verbosity", verbosity_result),
    ):
        payloads.append(
            build_mask_payload(
                mask_name=f"random_mask_{baseline_name}",
                selection_run_id=selection_run_id,
                feature_table_run_id=feature_table_run_id,
                members=build_random_baseline_members(
                    reference_members=reference_result["members"],
                    available_rows=score_rows,
                    seed=random_seed + len(baseline_name),
                ),
                split_audit=split_audit,
                objective_name=f"matched_{baseline_name}_random_baseline",
                predicted_objective=None,
                construction_log=[],
                reference_mask_name=f"{baseline_name}_mask",
                reference_type="random_baseline",
            )
        )
        payloads.append(
            build_mask_payload(
                mask_name=f"activation_mass_mask_{baseline_name}",
                selection_run_id=selection_run_id,
                feature_table_run_id=feature_table_run_id,
                members=build_activation_mass_baseline_members(
                    reference_members=reference_result["members"],
                    available_rows=score_rows,
                ),
                split_audit=split_audit,
                objective_name=f"matched_{baseline_name}_activation_mass_baseline",
                predicted_objective=None,
                construction_log=[],
                reference_mask_name=f"{baseline_name}_mask",
                reference_type="activation_mass_baseline",
            )
        )
    if refusal_result is not None:
        payloads.append(
            build_mask_payload(
                mask_name="random_mask_refusal",
                selection_run_id=selection_run_id,
                feature_table_run_id=feature_table_run_id,
                members=build_random_baseline_members(
                    reference_members=refusal_result["members"],
                    available_rows=score_rows,
                    seed=random_seed + 97,
                ),
                split_audit=split_audit,
                objective_name="matched_refusal_random_baseline",
                predicted_objective=None,
                construction_log=[],
                reference_mask_name="refusal_mask",
                reference_type="random_baseline",
            )
        )
        payloads.append(
            build_mask_payload(
                mask_name="activation_mass_mask_refusal",
                selection_run_id=selection_run_id,
                feature_table_run_id=feature_table_run_id,
                members=build_activation_mass_baseline_members(
                    reference_members=refusal_result["members"],
                    available_rows=score_rows,
                ),
                split_audit=split_audit,
                objective_name="matched_refusal_activation_mass_baseline",
                predicted_objective=None,
                construction_log=[],
                reference_mask_name="refusal_mask",
                reference_type="activation_mass_baseline",
            )
        )
    return payloads


def build_mask_payload(
    *,
    mask_name: str,
    selection_run_id: str,
    feature_table_run_id: str,
    members: list[dict[str, int]],
    split_audit: dict[str, Any],
    objective_name: str,
    predicted_objective: float | None,
    construction_log: list[dict[str, Any]],
    reference_mask_name: str | None,
    reference_type: str,
) -> dict[str, Any]:
    sorted_members = sorted_mask_members(members)
    return {
        "run_id": selection_run_id,
        "stage_name": "select_feature_masks",
        "mask_name": mask_name,
        "reference_type": reference_type,
        "reference_mask_name": reference_mask_name,
        "feature_table_run_id": feature_table_run_id,
        "selection_split": "select_tune",
        "selector_fit_split": "select_train",
        "objective_name": objective_name,
        "predicted_objective": predicted_objective,
        "members": sorted_members,
        "size": len(sorted_members),
        "layer_counts": {str(layer): count for layer, count in layer_count_map(sorted_members).items()},
        "split_audit": split_audit,
        "construction_log": construction_log,
    }


def build_random_baseline_members(
    *,
    reference_members: list[dict[str, int]],
    available_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, int]]:
    reference_counts = layer_count_map(reference_members)
    reference_keys = {candidate_feature_key(member["layer"], member["feature_id"]) for member in reference_members}
    available_by_layer: dict[int, list[dict[str, int]]] = {}
    for row in available_rows:
        layer = int(row["layer"])
        available_by_layer.setdefault(layer, []).append({"layer": layer, "feature_id": int(row["feature_id"])})
    rng = random.Random(seed)
    sampled: list[dict[str, int]] = []
    for layer, required_count in sorted(reference_counts.items()):
        layer_candidates = [
            member
            for member in available_by_layer.get(layer, [])
            if candidate_feature_key(member["layer"], member["feature_id"]) not in reference_keys
        ]
        if len(layer_candidates) < required_count:
            raise ValueError(f"Not enough layer-matched candidates for random baseline on layer {layer}")
        rng.shuffle(layer_candidates)
        sampled.extend(layer_candidates[:required_count])
    return sampled


def build_activation_mass_baseline_members(
    *,
    reference_members: list[dict[str, int]],
    available_rows: list[dict[str, Any]],
) -> list[dict[str, int]]:
    reference_counts = layer_count_map(reference_members)
    reference_keys = {candidate_feature_key(member["layer"], member["feature_id"]) for member in reference_members}
    selected: list[dict[str, int]] = []
    for layer, required_count in sorted(reference_counts.items()):
        layer_rows = [
            row
            for row in available_rows
            if int(row["layer"]) == layer and candidate_feature_key(int(row["layer"]), int(row["feature_id"])) not in reference_keys
        ]
        layer_rows.sort(key=lambda item: (-float(item["mass"]), int(item["feature_id"])))
        if len(layer_rows) < required_count:
            raise ValueError(f"Not enough layer-matched candidates for activation-mass baseline on layer {layer}")
        selected.extend({"layer": layer, "feature_id": int(row["feature_id"])} for row in layer_rows[:required_count])
    return selected


if __name__ == "__main__":
    raise SystemExit(main())
