# gemma-lasso-router

Two small, self-funded AI research projects, run on a laptop (RTX 3060, 6 GB) and on CPU.
Experiment code in `lasso_tests/` and `continual_learning/` was written with Claude (Anthropic) as a coding
assistant; the ideas and direction are mine. Each folder has a detailed `RESULTS.md` with every number,
seed counts, and caveats. All results are small-scale; negative results are reported as they came out.

## 1. Lasso router ("navigate to the neighborhood, then compute")
Idea: run many cheap "tracings" to collect candidate specialists, use a lasso penalty to keep only the
few that explain the prompt, and load just those.

- `01_*.py` – `17_*.py` (June 2026): Gemma 3 1B (`google/gemma-3-1b-it`) with 7 LoRA adapters
  (r = 8, q/k/v/o), synthetic training data, and an L1 logistic-regression router on MiniLM embeddings.
- `lasso_tests/` (October 2026): synthetic selection and navigation tests, Gate 1 (real embeddings),
  Gate 2 (Gemma 3 1B + 6 adapters, 80 auto-graded prompts).

Key results (details in `lasso_tests/RESULTS.md`, `lasso_tests/NAVIGATION_RESULTS.md`):
- Gate 1: lasso picked exactly the right adapter set 63.5% ± 3.0 of the time vs 23.9% for top-1 routing
  (10 train/test splits of one prompt set).
- Gate 2 (one deterministic run): **negative**. Plain Gemma 32.5% vs lasso 12.5% vs top-1 router 7.5%;
  even perfect routing (17.5%) lost, so the adapters themselves hurt. One subgroup (SQL + percentage):
  lasso 7/15 vs router 1/15 (sign test p = 0.016).
- Synthetic navigation (3 seeds): on a lasso-learned sparse graph, greedy search found the best specialist
  in 83.8% of cases at 1,024 specialists (kNN graph 67.4%) using about 41 comparisons instead of 1,024.

## 2. Continual learning without forgetting (`continual_learning/`)
Idea: instead of overwriting shared weights, grow new memory and switch it on only in the right
situation (a lasso as the switch); under a memory cap, store new facts as sparse lasso "recipes" of
existing slots.

Toy results (tiny synthetic transformer, 3 seeds each; see `continual_learning/RESULTS.md`):
- Lasso-based slot selection ties KL selection and beats TF-IDF for sparse memory finetuning; all sparse
  methods beat full finetuning about 6× on a learning-vs-retention measure.
- Side finding: with AdamW, "sparse" updates leak — 20–36× more memory rows change than were selected.
- Growth + gated switch: about 99% of new facts learned with ~0.999 retention (close to GRACE / NeuralDB,
  i.e. not new).
- Capped memory: lasso recipes stored 0.720 / 0.436 / 0.214 of 400 facts at 50 / 25 / 12.5% budgets, vs
  0.549 / 0.315 / 0.128 for a MEMOIR-style toy baseline (random features, not real MEMOIR).

Real-model benchmark (in progress): zsRE, 1,000 sequential edits, Qwen2.5-0.5B-Instruct, RTX 3060.
- MEMOIR (my port of the public implementation, single seed): reliability 0.686, generalization 0.589,
  locality 1.000, average 0.758 at threshold 0.5. Lower than the paper reports for 7–8B models; the cause
  is not yet diagnosed.
- My methods on this benchmark: **not yet measured** (`run_edit_bench.ps1`).

## Not included
- LoRA adapter weights (`adapters/`, ~270 MB) and the Python environment.
- `zsre_mend_eval.json` (downloaded by `run_edit_bench.ps1` from rome.baulab.info).
- Model checkpoints (`*.pt`); scripts regenerate them.

## Setup
Python 3.11, PyTorch 2.6 (CUDA 12.4), transformers 4.51.3, peft, sentence-transformers, scikit-learn, numpy.
