from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_ROOT = REPO_ROOT / "paper"
FIG_ROOT = PAPER_ROOT / "figures"
GEN_ROOT = PAPER_ROOT / "generated"


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    GEN_ROOT.mkdir(parents=True, exist_ok=True)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def build_main_results_table(main_metrics: dict) -> None:
    generation = main_metrics["generation_metrics"]
    teacher = main_metrics["teacher_forced"]

    rows = [
        ("PT", "PT"),
        ("IT target", "IT_neutral"),
        ("FullDelta", "PT_plus_FullDelta"),
        ("CapMask", "PT_plus_CapMask"),
        ("FullDelta - VMask", "PT_plus_FullDelta_minus_VerbosityMask"),
        ("RandomMask", "PT_plus_RandomMask"),
        ("ActivationMassMask", "PT_plus_ActivationMassMask"),
        ("MeanDiff", "PT_plus_MeanDiff"),
    ]

    body_rows = []
    for label, key in rows:
        prefix = ""
        if label == "FullDelta":
            prefix = "\\rowcolor{accentlight}\n"
        if label == "CapMask":
            prefix = "\\rowcolor{failurelight}\n"
        kl = teacher[key]["KL_ans_to_IT"]
        cap = generation[key]["Cap"]
        cap_recovery = generation[key]["Cap_Recovery"]
        length = generation[key]["Len"]
        brevity = generation[key]["BrevEx"]
        verbcarry = generation[key]["VerbCarry"]
        body_rows.append(
            prefix
            + f"{label} & {fmt(kl)} & {fmt(cap)} & {fmt(cap_recovery)} & "
            + f"{fmt(length, 2)} & {fmt(brevity, 2)} & {fmt(verbcarry)} \\\\"
        )

    table = rf"""
\begin{{table}}[t]
\centering
\caption{{Main held-out results on the neutral-template test split. Lower is better for KL, length, brevity excess, and verbosity carryover. Higher is better for capability and capability recovery.}}
\label{{tab:main-results}}
\small
\setlength{{\tabcolsep}}{{6pt}}
\begin{{threeparttable}}
\begin{{tabular}}{{lrrrrrr}}
\toprule
Variant & KL$_{{ans \to IT}}$ & Cap & CapRec & Len & BrevEx & VerbCarry \\
\midrule
{chr(10).join(body_rows)}
\bottomrule
\end{{tabular}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Cap is the mean of QA exact match, Math exact match, and format pass rate. CapRec is capability recovery relative to PT and IT.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()

    write_text(GEN_ROOT / "main_results_table.tex", table + "\n")


def build_key_comparisons_table(bootstrap: dict) -> None:
    comparison_map = {row["name"]: row for row in bootstrap["comparisons"]}

    rows = [
        ("FullDelta vs PT", "full_delta_vs_pt", "Cap", "Capability gain"),
        ("FullDelta vs PT", "full_delta_vs_pt", "Len", "Much shorter than PT"),
        ("CapMask vs PT", "capmask_vs_pt", "Cap", "Null held-out result"),
        ("VMask subtraction vs FullDelta", "verbosity_subtraction_vs_fulldelta", "Cap", "Recovered capability preserved"),
        (
            "VMask subtraction vs FullDelta",
            "verbosity_subtraction_vs_fulldelta",
            "Len",
            "Gets longer, not shorter",
        ),
        (
            "VMask subtraction vs FullDelta",
            "verbosity_subtraction_vs_fulldelta",
            "VerbCarry",
            "Slightly lower verbosity carry metric",
        ),
        ("MeanDiff vs FullDelta", "mean_diff_vs_fulldelta", "Cap", "Much weaker than the learned surrogate"),
    ]

    formatted_rows = []
    for label, comp_name, metric_name, reading in rows:
        metric = comparison_map[comp_name]["metrics"][metric_name]
        delta = fmt(metric["delta"])
        ci = f"[{fmt(metric['ci_lower'])}, {fmt(metric['ci_upper'])}]"
        formatted_rows.append(
            f"{label} & {metric_name} & {delta} & {ci} & {reading} \\\\"
        )

    table = rf"""
\begin{{table}}[t]
\centering
\caption{{Key paired bootstrap comparisons on the held-out test split (10{{,}}000 prompt-level resamples, 95\% confidence intervals).}}
\label{{tab:key-comparisons}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\begin{{threeparttable}}
\begin{{tabularx}}{{\linewidth}}{{l l r r X}}
\toprule
Comparison & Metric & Delta & 95\% CI & Reading \\
\midrule
{chr(10).join(formatted_rows)}
\bottomrule
\end{{tabularx}}
\begin{{tablenotes}}[flushleft]
\footnotesize
\item Deltas are computed in the direction shown in the first column. Negative deltas are better for KL, length, and verbosity carry; positive deltas are better for capability.
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
""".strip()

    write_text(GEN_ROOT / "key_comparisons_table.tex", table + "\n")


def build_metrics_macros(main_metrics: dict, ablation_metrics: dict, prompt_summary: dict) -> None:
    generation = main_metrics["generation_metrics"]
    teacher = main_metrics["teacher_forced"]
    ablation_generation = ablation_metrics["generation_metrics"]
    ablation_teacher = ablation_metrics["teacher_forced"]
    assessment = main_metrics["c1_assessment"]

    macro_lines = [
        rf"\newcommand{{\NumTestPrompts}}{{{prompt_summary['splits']['test']}}}",
        rf"\newcommand{{\FullDeltaKL}}{{{fmt(teacher['PT_plus_FullDelta']['KL_ans_to_IT'])}}}",
        rf"\newcommand{{\PTKL}}{{{fmt(teacher['PT']['KL_ans_to_IT'])}}}",
        rf"\newcommand{{\KLReductionPct}}{{{fmt(100.0 * assessment['kl_reduction_fraction'], 1)}}}",
        rf"\newcommand{{\FullDeltaCap}}{{{fmt(generation['PT_plus_FullDelta']['Cap'])}}}",
        rf"\newcommand{{\FullDeltaCapRecovery}}{{{fmt(generation['PT_plus_FullDelta']['Cap_Recovery'])}}}",
        rf"\newcommand{{\ITCap}}{{{fmt(generation['IT_neutral']['Cap'])}}}",
        rf"\newcommand{{\CapMaskCap}}{{{fmt(generation['PT_plus_CapMask']['Cap'])}}}",
        rf"\newcommand{{\VSubLen}}{{{fmt(generation['PT_plus_FullDelta_minus_VerbosityMask']['Len'], 2)}}}",
        rf"\newcommand{{\FullDeltaLen}}{{{fmt(generation['PT_plus_FullDelta']['Len'], 2)}}}",
        rf"\newcommand{{\VSubVerbCarry}}{{{fmt(generation['PT_plus_FullDelta_minus_VerbosityMask']['VerbCarry'])}}}",
        rf"\newcommand{{\FullDeltaVerbCarry}}{{{fmt(generation['PT_plus_FullDelta']['VerbCarry'])}}}",
        rf"\newcommand{{\TwoLayerCap}}{{{fmt(generation['PT_plus_FullDelta']['Cap'])}}}",
        rf"\newcommand{{\OneLayerCap}}{{{fmt(ablation_generation['PT_plus_FullDelta']['Cap'])}}}",
        rf"\newcommand{{\TwoLayerKL}}{{{fmt(teacher['PT_plus_FullDelta']['KL_ans_to_IT'])}}}",
        rf"\newcommand{{\OneLayerKL}}{{{fmt(ablation_teacher['PT_plus_FullDelta']['KL_ans_to_IT'])}}}",
    ]

    write_text(GEN_ROOT / "metrics_macros.tex", "\n".join(macro_lines) + "\n")


def build_main_results_figure(main_metrics: dict) -> None:
    generation = main_metrics["generation_metrics"]
    teacher = main_metrics["teacher_forced"]

    variants = [
        ("PT", "PT"),
        ("IT", "IT_neutral"),
        ("FullDelta", "PT_plus_FullDelta"),
        ("CapMask", "PT_plus_CapMask"),
        ("FullDelta - VMask", "PT_plus_FullDelta_minus_VerbosityMask"),
        ("MeanDiff", "PT_plus_MeanDiff"),
    ]
    colors = {
        "PT": "#8C8C8C",
        "IT": "#1F1F1F",
        "FullDelta": "#1F77B4",
        "CapMask": "#D55E00",
        "FullDelta - VMask": "#009E73",
        "MeanDiff": "#7F7FBD",
    }

    metrics = [
        ("Capability composite", [generation[key]["Cap"] for _, key in variants], "higher"),
        ("KL to IT", [teacher[key]["KL_ans_to_IT"] for _, key in variants], "lower"),
        ("Mean output length", [generation[key]["Len"] for _, key in variants], "lower"),
        ("Verbosity carryover", [generation[key]["VerbCarry"] for _, key in variants], "lower"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.2))
    axes = axes.ravel()
    labels = [name for name, _ in variants]
    x = np.arange(len(labels))

    for ax, (title, values, direction) in zip(axes, metrics, strict=True):
        ax.bar(
            x,
            values,
            color=[colors[label] for label in labels],
            edgecolor="#2B2B2B",
            linewidth=0.6,
        )
        ax.set_title(
            f"{title} ({'up' if direction == 'higher' else 'down'} is better)",
            fontsize=11.0,
            pad=8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9.6)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Main held-out intervention results",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_ROOT / "main_results.pdf", bbox_inches="tight")
    plt.close(fig)


def build_robustness_figure(
    main_metrics: dict,
    ablation_metrics: dict,
    threshold_metrics: dict,
) -> None:
    generation = main_metrics["generation_metrics"]
    teacher = main_metrics["teacher_forced"]
    ablation_generation = ablation_metrics["generation_metrics"]
    ablation_teacher = ablation_metrics["teacher_forced"]

    pt = generation["PT"]
    ablation_panels = [
        (
            "KL reduction vs PT (%)",
            [
                100.0
                * (
                    teacher["PT"]["KL_ans_to_IT"]
                    - teacher["PT_plus_FullDelta"]["KL_ans_to_IT"]
                )
                / teacher["PT"]["KL_ans_to_IT"],
                100.0
                * (
                    ablation_teacher["PT"]["KL_ans_to_IT"]
                    - ablation_teacher["PT_plus_FullDelta"]["KL_ans_to_IT"]
                )
                / ablation_teacher["PT"]["KL_ans_to_IT"],
            ],
        ),
        (
            "Capability composite",
            [
                generation["PT_plus_FullDelta"]["Cap"],
                ablation_generation["PT_plus_FullDelta"]["Cap"],
            ],
        ),
        (
            "Length reduction vs PT",
            [
                pt["Len"] - generation["PT_plus_FullDelta"]["Len"],
                pt["Len"] - ablation_generation["PT_plus_FullDelta"]["Len"],
            ],
        ),
    ]

    cap_variants = [
        row
        for row in threshold_metrics["variants"]
        if row["target"] == "capability"
    ]
    cap_labels = [row["mask_name"].replace("capability_mask_", "") for row in cap_variants]
    cap_objectives = [row["predicted_objective"] for row in cap_variants]
    cap_colors = [
        "#1F77B4" if row["matches_locked_mask"] else "#D55E00"
        for row in cap_variants
    ]

    fig = plt.figure(figsize=(11.8, 4.0))
    grid = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 1.35])
    axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]

    ablation_labels = ["2-layer", "1-layer"]
    ablation_colors = ["#1F77B4", "#7F7FBD"]

    for ax, (title, values) in zip(axes[:3], ablation_panels, strict=True):
        ax.bar(
            np.arange(2),
            values,
            color=ablation_colors,
            edgecolor="#2B2B2B",
            linewidth=0.6,
        )
        ax.set_title(title, fontsize=10.6, pad=8)
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(ablation_labels, rotation=14, ha="right", fontsize=9.4)
        ax.grid(axis="y", color="#E7E7E7", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    ax = axes[3]
    y_pos = np.arange(len(cap_labels))[::-1]
    ax.scatter(
        cap_objectives,
        y_pos,
        s=72,
        c=cap_colors,
        edgecolors="#2B2B2B",
        linewidth=0.6,
        zorder=3,
    )
    ax.hlines(
        y_pos,
        xmin=min(cap_objectives) - 0.003,
        xmax=cap_objectives,
        color="#CFCFCF",
        linewidth=1.0,
        zorder=1,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cap_labels, fontsize=9.3)
    ax.set_title("Capability-mask threshold sweep", fontsize=10.6, pad=8)
    ax.set_xlabel("Predicted objective on select_tune", fontsize=9.4)
    ax.grid(axis="x", color="#E7E7E7", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.suptitle(
        "Ablation and selector-side robustness",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_ROOT / "robustness_results.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()

    main_metrics = load_json(REPO_ROOT / "results/metrics/fidelity_run_eval-4418a8a8ede9.json")
    ablation_metrics = load_json(REPO_ROOT / "results/metrics/fidelity_run_eval-ef0e63facf62.json")
    bootstrap = load_json(REPO_ROOT / "results/metrics/bootstrap_cis_run_eval-4418a8a8ede9.json")
    threshold_metrics = load_json(
        REPO_ROOT
        / "results/metrics/threshold_sensitivity_build_threshold_sensitivity-ee261143421c.json"
    )
    prompt_summary = load_json(
        REPO_ROOT / "data/processed/prompt_suite_summary_build_prompt_suite-3cab836a02bd.json"
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
        }
    )

    build_main_results_table(main_metrics)
    build_key_comparisons_table(bootstrap)
    build_metrics_macros(main_metrics, ablation_metrics, prompt_summary)
    build_main_results_figure(main_metrics)
    build_robustness_figure(main_metrics, ablation_metrics, threshold_metrics)


if __name__ == "__main__":
    main()
