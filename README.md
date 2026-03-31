# Surgical Post-Training Diffing

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aliuyar1234/surgical-posttraining-diffing)
[![Paper PDF](https://img.shields.io/badge/PDF-Download%20Paper-B30B00?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://raw.githubusercontent.com/aliuyar1234/surgical-posttraining-diffing/main/paper/surgical-posttraining-diffing-ali-uyar.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-Manuscript-008080?style=flat-square&logo=latex&logoColor=white)](./paper/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](./pyproject.toml)

Code and manuscript for studying sparse PT-to-IT diffing, feature-mask interventions, and evaluation of instruction-tuning behavior.

This repository asks a focused mechanistic question: how much of the behavioral shift from a pretrained model to its instruction-tuned sibling can be recovered by a small sparse surrogate, and how cleanly can targeted feature masks separate capability from verbosity.

## Highlights

- Sparse PT-to-IT surrogate learning over answer-phase residual-stream activations.
- Causal intervention tooling for full-delta edits, sparse masks, and subtraction-style edits.
- Evaluation code for capability, fidelity, verbosity, baselines, ablations, and bootstrap comparisons.
- Included manuscript source and final PDF for the current paper draft.

## Paper

- Final PDF: [`paper/surgical-posttraining-diffing-ali-uyar.pdf`](./paper/surgical-posttraining-diffing-ali-uyar.pdf)
- LaTeX source: [`paper/main.tex`](./paper/main.tex)
- Bibliography: [`paper/references.bib`](./paper/references.bib)

## Repository Layout

- `src/`: implementation for data construction, caching, sparse-delta training, analysis, and evaluation
- `configs/`: experiment and pipeline configs
- `tests/`: unit tests and contract checks
- `paper/`: manuscript source, bibliography, asset builder, and tracked final PDF
- `pyproject.toml`: project metadata and pytest defaults

## What Is Intentionally Omitted

This is a slim public release, not a full artifact dump.

- Internal planning files and private workflow notes
- Generated datasets, cached activations, checkpoints, metrics, and runtime outputs
- Large intermediate artifacts and local scratch directories
- Generated paper assets beyond the tracked final PDF

## Quick Start

1. Create a Python 3.11 environment.
2. Install the runtime stack you want to use for your hardware setup.
3. Use Hugging Face model IDs directly, or switch config entries to explicit local paths with `./` or `../` prefixes.
4. Run the tests with `python -m pytest -q`.

Representative entrypoints:

- `python -m src.data.build_prompt_suite --config configs/data.yaml`
- `python -m src.data.generate_it_completions --config configs/model_pair.yaml`
- `python -m src.cache.cache_paired_activations --config configs/cache.yaml`
- `python -m src.train.train_sparse_delta --config configs/delta_module.yaml`
- `python -m src.analysis.select_feature_masks --config configs/selectors.yaml`
- `python -m src.eval.run_eval --config configs/eval.yaml`

## Config Notes

- The config loader preserves Hugging Face model IDs such as `google/gemma-3-4b-it`.
- Explicit relative filesystem paths are resolved relative to the config file that declares them.
- Some configs reference run-specific artifacts under `artifacts/` or `results/`; those are intentionally not versioned in this public repo and need to be regenerated locally.

## Scope

The public repo includes the implementation, configs, tests, manuscript source, and final PDF needed to inspect the method and reproduce the software structure. It does not attempt to bundle every experimental byproduct that existed in the original research workspace.
