# Surgical Post-Training Diffing: Partial Recovery Without Clean Small-Mask Separation

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.19347477-0A7BBB?style=flat-square&logo=doi&logoColor=white)](https://doi.org/10.5281/zenodo.19347477)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF-B31B1B?style=flat-square&logo=adobeacrobatreader&logoColor=white)](paper/surgical-posttraining-diffing-ali-uyar.pdf)
[![Manuscript Source](https://img.shields.io/badge/LaTeX-Source-1D4ED8?style=flat-square&logo=latex&logoColor=white)](paper/main.tex)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/aliuyar1234/surgical-posttraining-diffing)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![Scope](https://img.shields.io/badge/Scope-Paired%20PT%2FIT%20Study-5B4B8A?style=flat-square)](#scope)

Ali Uyar
Independent Researcher

**Paper title:** *Surgical Post-Training Diffing: Partial Recovery Without Clean Small-Mask Separation*

This repository accompanies a focused mechanistic study of instruction tuning as a sparse activation difference. It pairs Gemma 3 4B pretrained (PT) and instruction-tuned (IT) siblings under a shared neutral template, trains sparse modules to predict answer-phase residual-stream deltas from PT hidden states, and then intervenes on subsets of the resulting features. The question is not only whether a compact surrogate can recover part of the PT-to-IT shift, but whether tiny frozen feature masks can selectively preserve capability while discarding assistant-style verbosity.

## Abstract

Instruction-tuned models differ from their pretrained siblings in ways that are behaviorally useful but internally hard to localize. We study this gap with a paired answer-phase sparse surrogate built from two residual-stream layers under a shared neutral template. On Gemma 3 4B PT/IT, the learned surrogate lowers answer-token KL to the instruction-tuned model, raises a held-out capability composite from 0, and sharply reduces the pretrained model's long repeated continuations. However, the more ambitious separation story fails on held-out data. A frozen 5-feature capability mask recovers no capability at all, matching trivial baselines, and subtracting a learned verbosity mask preserves recovered capability but lengthens outputs as the carryover-based verbosity score falls. A late-only one-layer ablation nearly matches the two-layer surrogate, and a threshold sweep shows the null capability-mask result is robust to nearby selector cutoffs. The resulting picture is more precise than a simple positive or negative headline: a small answer-phase surrogate can capture a real part of instruction-tuning behavior, but tiny frozen masks do not cleanly split capability from verbosity in this compact setting.

## Main Finding

The result has two halves that must be reported together, not one. The learned surrogate is a real held-out intervention; the small-mask separation story is not.

| Variant                          | Answer-token KL to IT | Capability composite | Capability recovery | Mean output length | Verbosity carryover |
| -------------------------------- | --------------------- | -------------------- | ------------------- | ------------------ | ------------------- |
| PT (baseline)                    | high                  | 0.00                 | 0%                  | long (repeats)     | ---                 |
| IT (neutral template target)     | 0 (reference)         | reference            | 100% (reference)    | reference          | reference           |
| **PT + FullDelta** (surrogate)   | **substantially reduced** | **nontrivial, > 0** | **partial**         | much shorter       | lower               |
| PT + CapMask (5-feature frozen)  | PT-like               | 0.00                 | 0%                  | PT-like            | PT-like             |
| PT + FullDelta - VerbosityMask   | similar to FullDelta  | preserved            | preserved           | **longer** than FullDelta | lower        |
| MeanDiff / Random / ActMass      | PT-like               | ~0                   | ~0%                 | PT-like            | PT-like             |

The two-layer answer-phase surrogate (`FullDelta`) is the clearest positive result: it is the only frozen intervention that moves teacher-forced fidelity and held-out capability meaningfully away from the pretrained baseline, and it is not matched by mean-delta, random-mask, or activation-mass-mask baselines. The recovery is partial, not majority, and the late-layer teacher-forced reconstruction `R^2` remains negative, so the surrogate is real but incomplete. The clean separation story, however, fails on the held-out test split: a 5-feature frozen capability mask lands on the null, and verbosity subtraction lowers the carryover score while making raw outputs *longer* rather than shorter. A one-layer late-only ablation nearly matches the two-layer model, and a threshold sweep reproduces the same null capability-mask result across nearby selector cutoffs.

## Contributions

1. A paired sparse answer-phase surrogate for the PT-to-IT shift on Gemma 3 4B. The surrogate caches answer-token residual-stream deltas at two layers under a shared neutral template, learns sparse modules to predict those deltas from PT hidden states, and intervenes by adding decoded deltas back into the PT forward pass with per-layer calibrated gates.
2. A held-out evaluation-first protocol that reports KL to IT, a QA/Math/Format capability composite, capability recovery relative to PT and IT, mean length and brevity excess, and a verbosity carryover metric, with prompt-level paired bootstrap intervals.
3. A direct demonstration that tiny frozen masks do not cleanly separate capability from verbosity on held-out data: the frozen capability mask fails outright, and verbosity subtraction is mixed rather than clean.
4. Support from matched trivial baselines (mean-delta, random mask, activation-mass mask), a one-layer late-only ablation, and a threshold-sensitivity sweep that together rule out several "selector was just unlucky" explanations of the null.

## Scope

This release is intentionally narrow and claim-safe.

- One PT/IT model pair: Gemma 3 4B pretrained and instruction-tuned siblings
- One prompt condition: a shared neutral template with answer-phase scope
- Evaluation over six slices (QA, Math, Format, Brevity, Harmful, BenignAdjacent), with capability restricted to QA/Math/Format and refusal treated as secondary
- Two residual-stream layers in the main surrogate; one-layer late-only tested as an ablation
- No training of the underlying models; interventions modify PT forward passes at answer-token positions only
- Capability recovery is partial and is driven more by short-format execution (Math and Format) than by open-domain QA exact match

The contribution is not a disentanglement claim. It is a bounded mechanistic study: a real held-out surrogate for part of the PT-to-IT shift, and a principled negative result on tiny-frozen-mask capability/verbosity separation.

## Paper

- Compiled PDF: [`paper/surgical-posttraining-diffing-ali-uyar.pdf`](paper/surgical-posttraining-diffing-ali-uyar.pdf)
- LaTeX source: [`paper/main.tex`](paper/main.tex)
- Bibliography: [`paper/references.bib`](paper/references.bib)
- Generated tables and metric macros: [`paper/generated/`](paper/generated/)

## Repository Layout

- [`src/`](src/) — implementation for data construction, caching, sparse-delta training, feature scoring, intervention, and evaluation
- [`configs/`](configs/) — experiment and pipeline configurations for data, model pair, caching, delta training, selectors, interventions, and evaluation
- [`tests/`](tests/) — unit tests and contract checks
- [`paper/`](paper/) — manuscript source, bibliography, asset builder, and tracked final PDF
- [`docs/`](docs/) — public methods and operational documentation
- [`artifacts/`](artifacts/) and [`results/`](results/) — run-specific outputs produced locally (not all versioned in the public repo)

Representative entrypoints:

```bash
python -m src.data.build_prompt_suite       --config configs/data.yaml
python -m src.data.generate_it_completions  --config configs/model_pair.yaml
python -m src.cache.cache_paired_activations --config configs/cache.yaml
python -m src.train.train_sparse_delta      --config configs/delta_module.yaml
python -m src.analysis.select_feature_masks --config configs/selectors.yaml
python -m src.eval.run_eval                 --config configs/eval.yaml
```

## Reproducibility

The repository includes resolved configs, frozen run identifiers, artifact manifests, prompt-level outputs, and stage-level runtime logs. Large intermediate artifacts (cached activations, checkpoints, generated datasets) are not tracked in the public repo and need to be regenerated locally. The config loader preserves Hugging Face model IDs such as `google/gemma-3-4b-it`; explicit relative paths are resolved against the config file that declares them.

## Citation

```bibtex
@unpublished{uyar2026surgicalposttraining,
  author = {Uyar, Ali},
  title  = {Surgical Post-Training Diffing: Partial Recovery Without Clean Small-Mask Separation},
  year   = {2026},
  doi    = {10.5281/zenodo.19347477},
  note   = {Independent research}
}
```
