# Real-model test: MEMOIR vs your lasso recipes (zsRE, Qwen2.5-0.5B, RTX 3060)

## Run it (one command)
Open PowerShell and paste:
```powershell
powershell -ExecutionPolicy Bypass -File "$HOME\gemma-lasso-router\continual_learning\run_edit_bench.ps1"
```
What it does:
1. Downloads the standard zsRE evaluation file (7.7 MB) the first time.
2. Checks the GPU. You should see `CUDA: True NVIDIA GeForce RTX 3060 ...`.
3. Downloads Qwen2.5-0.5B-Instruct (~1 GB) the first time, then runs 6 experiments of 1,000 sequential
   edits each. My estimate is 20–40 minutes each, so about 2–4 hours total. It is safe to stop and
   rerun: finished experiments are skipped.
4. Prints a results table and saves it to `continual_learning\edit_results\SUMMARY.md`.

When it finishes, tell me and I'll read the results from your folder.

## The 6 experiments
| Run | What it is |
|---|---|
| memoir | Official MEMOIR mechanism (NeurIPS 2025), ported faithfully; ~4.4M memory numbers |
| slot_lasso_B500 | Your method: lasso recipes + per-edit addresses, capped at 500 slots for 1,000 edits |
| slot_unbounded | Same slot design, no cap (ceiling of the design) |
| slot_evict_B500 | Same cap, but overwrite the oldest slot when full (baseline) |
| slot_lasso_B250 | Your method at a 250-slot cap |
| slot_evict_B250 | Eviction baseline at 250 |

Scores (same as MEMOIR's code): **reliability** (edit learned), **generalization** (works when the question
is reworded), **locality** (unrelated answers unchanged), and their average.

## If something goes wrong
- *Out of GPU memory:* add `--dtype bf16` to the runs (edit the `$runs` list in the .ps1), or close other GPU apps.
- *Download blocked:* get `zsre_mend_eval.json` from https://rome.baulab.info/data/dsets/ in a browser and put
  it in the `continual_learning` folder.
- Logs for each run are in `edit_results\*.log`.
