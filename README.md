# Surgical Post-Training Diffing

[![Paper PDF](https://img.shields.io/badge/PDF-Download%20Paper-B30B00?style=flat-square&logo=adobeacrobatreader&logoColor=white)](https://raw.githubusercontent.com/aliuyar1234/surgical-posttraining-diffing/main/paper/surgical-posttraining-diffing-ali-uyar.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-Manuscript-008080?style=flat-square&logo=latex&logoColor=white)](./paper/main.tex)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](./pyproject.toml)

Code for sparse PT-to-IT diffing, intervention masks, and evaluation of instruction-tuning behavior.

This public repo is intentionally slim. It keeps the implementation, configs, tests, and manuscript source, while leaving out internal planning docs, generated artifacts, datasets, cached activations, checkpoints, and compiled paper outputs.

## Included

- `src/`: data, caching, training, analysis, and evaluation code
- `configs/`: experiment and pipeline configs
- `tests/`: invariants and regression checks
- `paper/main.tex`: manuscript source
- `paper/references.bib`: bibliography
- `paper/scripts/build_assets.py`: figure/table asset generator
- `pyproject.toml`: package and test metadata

## Intentionally omitted

- `AGENTS.md` and `docs/`
- `data/`
- `artifacts/`
- `results/`
- generated paper assets beyond the final PDF
- local caches and temporary files

## What this repo studies

The project asks a focused mechanistic question: how much of the behavioral gap between a pretrained model and its instruction-tuned counterpart can be recovered by a small sparse residual-stream surrogate, and how well do targeted feature masks separate different behavioral effects.

The codebase includes:

- prompt-suite construction and rendering utilities
- paired activation caching
- sparse delta training and gate calibration
- feature-table construction and mask selection
- evaluation, bootstrapping, and runtime accounting

## Quick start

1. Create a Python 3.11 environment.
2. Install the project and test dependencies.
3. Use Hugging Face model IDs directly or switch to explicit local paths with `./` or `../` prefixes.
4. Run the test suite with `pytest -q`.

Representative entrypoints:

- `python -m src.data.build_prompt_suite --config configs/data.yaml`
- `python -m src.data.generate_it_completions --config configs/model_pair.yaml`
- `python -m src.cache.cache_paired_activations --config configs/cache.yaml`
- `python -m src.train.train_sparse_delta --config configs/delta_module.yaml`
- `python -m src.analysis.select_feature_masks --config configs/selectors.yaml`
- `python -m src.eval.run_eval --config configs/eval.yaml`

## Notes on configs

Some configs in this repo are archived experiment configs and still reference run-specific artifacts that are intentionally ignored from version control. The loader accepts Hugging Face model IDs as-is, and it resolves explicit relative filesystem paths for local assets. Before running end-to-end experiments, regenerate the ignored artifacts locally or update those config entries to match your environment.

## Manuscript source

The manuscript source is included under `paper/`. Generated figures, tables, previews, and compiled PDFs are intentionally excluded from version control in this slim public release.
