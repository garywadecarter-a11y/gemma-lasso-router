# Lasso multi-adapter routing: test results (2026-10-12)

Idea tested: run many cheap "tracings" to collect candidate specialists, use a
lasso penalty to keep only the few that explain the prompt, load just those.

## Step 1: synthetic (lasso_select.py, curves.py)
- Hierarchical bank of 256 or 1,024 specialists; prompts blend 1–3; 3 seeds.
- At 256 specialists, clean prompts, about 3 kept: both skills found in 65.5% (lasso)
  vs 61.1% (OMP) vs 14.1% (similarity top-k).
- Tracings collected 91% of the needed specialists at 256 but only 52–60% at 1,024,
  so navigation is the weak link at scale.

## Gate 1: real MiniLM embeddings, your 7 routes (gate1*.py) — PASSED
165 mixed + 70 single prompts (written by Claude; ground truth is a judgment).
Tuned on half, scored on the other half, 10 splits. See gate1_summary.txt.

| Selector | Exactly right set | Mixed: both kept | Single: exact | Avg kept (mix/single) |
|---|---|---|---|---|
| Your router (top-1) | 23.9% | 0% | 84.0% | 1 / 1 |
| Similarity top-2 (centered) | 50.5% | 70.6% | 0% | 2 / 2 |
| Lasso | 63.5% ± 3.0 | 77.4% | 78.0% | 2.13 / 1.24 |

## Gate 2: Gemma 3 1B + your 6 LoRA adapters, CPU (gate2.py) — FAILED vs plain Gemma
80 auto-graded prompts (60 mixed, 20 single). See gate2_summary.txt; raw answers in
gate2_results.jsonl and gate2b_results.jsonl.

| Setup | All | Mixed |
|---|---|---|
| Plain Gemma (no adapter) | 32.5% | 40.0% |
| Your router (1 adapter) | 7.5% | 6.7% |
| Lasso set, full strength | 12.5% | 13.3% |
| Lasso, averaged | 13.8% | 15.0% |
| Perfect routing (cheat), full strength | 12.5% | 11.7% |
| Perfect routing (cheat), averaged | 17.5% | 18.3% |

- Even perfect routing loses to plain Gemma, so the adapters are the problem, not the
  selection. The harder_math adapter, for example, makes Gemma answer with a bare
  number, dropping JSON and code.
- Bright spot, SQL + percentage: lasso 7/15 vs your router 1/15 and plain Gemma 1/15
  (sign test p = 0.016), matching perfect routing (7/15).
- Blend strength matters by task: full strength helps SQL+math (47%) but breaks
  code+math (7%); averaging restores code+math (53%) but loses SQL+math (7%).

## Grading notes
- SQL is executed in SQLite; code is run against unit tests; numbers allow ±0.01.
- Not-enough-info and numeric grading were fixed after the run and re-graded from
  saved answers (the analyzer does this).
