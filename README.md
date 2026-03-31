# Surgical Post-Training Diffing

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
- compiled LaTeX outputs and generated paper assets
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
3. Update model and artifact paths in the configs for your environment.
4. Run the test suite with `pytest -q`.

Representative entrypoints:

- `python -m src.data.build_prompt_suite --config configs/data.yaml`
- `python -m src.data.generate_it_completions --config configs/model_pair.yaml`
- `python -m src.cache.cache_paired_activations --config configs/cache.yaml`
- `python -m src.train.train_sparse_delta --config configs/delta_module.yaml`
- `python -m src.analysis.select_feature_masks --config configs/selectors.yaml`
- `python -m src.eval.run_eval --config configs/eval.yaml`

## Notes on configs

Some configs in this repo are archived experiment configs and still contain environment-specific model paths or references to locally generated artifacts. Treat them as source material, not as guaranteed plug-and-play defaults. Before running end-to-end experiments, replace those paths with values that match your machine and regenerate any ignored artifacts locally.

## Manuscript source

The manuscript source is included under `paper/`. Generated figures, tables, previews, and compiled PDFs are intentionally excluded from version control in this slim public release.
