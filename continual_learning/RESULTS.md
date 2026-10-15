# Continual learning, round 1: slot-selection rules for sparse memory finetuning

## Setup (tiny CPU testbed, `cl_proxy.py`)
- 4-layer transformer (d = 128); layer 2's FFN is a product-key memory with 4,096 slots, top-8 lookup.
- Pretrained on 1,000 synthetic people (4,000 facts, 5 phrasings each) plus the addition table 0–20.
  All 3 seed models reach 100% on both.
- Continual phase: stream 400 new facts one at a time (16 paraphrases × 3 steps per fact).
- Measured at the end:
  - **new** = accuracy on streamed facts
  - **retention** = mean of (old-fact accuracy, addition accuracy) relative to before (1.00 = no forgetting)
- 7 methods × 4–20 settings × 3 seeds = 231 runs. Each method is judged by its best tradeoff curve.

## Main result: Gary's group-lasso rule ties KL; both beat TF-IDF; all sparse methods beat full finetuning by ~6×

Area under the learning-vs-retention curve (retention above 0.80); higher is better:

| Method | Seed 0 | Seed 1 | Seed 2 | Mean ± sd |
|---|---|---|---|---|
| Full finetuning (AdamW) | 0.016 | 0.018 | 0.022 | 0.019 ± 0.002 |
| Dense memory (all accessed rows) | 0.067 | 0.106 | 0.128 | 0.100 ± 0.026 |
| Gradient top-t | 0.068 | 0.107 | 0.129 | 0.101 ± 0.025 |
| Gradient ÷ interference top-t | 0.079 | 0.108 | 0.130 | 0.106 ± 0.021 |
| TF-IDF (Lin et al. 2025) | 0.094 | 0.112 | 0.133 | 0.113 ± 0.016 |
| KL (2026 follow-up) | 0.097 | 0.114 | 0.140 | 0.117 ± 0.018 |
| **Group lasso (Gary)** | 0.096 | 0.115 | 0.140 | **0.117 ± 0.018** |

Paired by seed:
- glasso − KL = −0.0003 / +0.0006 / +0.0004 (a tie)
- glasso − TF-IDF = +0.0025 / +0.0024 / +0.0069 (small but consistent)

What this means:
- **Weighting by background usage (interference) is the ingredient that matters.** Gradient-only selection is no
  better than updating everything (0.101 vs 0.100). Dividing by interference helps (0.106), and the full
  group-lasso shrink helps more (0.117).
- KL already contains an interference term (batch usage vs background usage), which is why it ties.
- Full finetuning learned at most 9.6% of new facts while keeping retention ≥ 0.90. Sparse methods
  learned 48–61% at the same retention.
- No method got above about 0.77 new-fact accuracy: later facts overwrite earlier ones within the stream.

## Second finding: Adam silently breaks sparsity (relevant to the 2026 Qwen code)
The 2026 follow-up runs its "sparse" phase with AdamW. Adam keeps per-parameter momentum, so rows masked
out this step still move if they were updated recently.

| Rule | Rows selected per step | Rows actually changed with Adam | With SGD |
|---|---|---|---|
| KL (top-64) | 64 | **~1,350 (21×)** | 56 |
| TF-IDF (top-64) | 64 | **~1,250 (20×)** | 60 |
| Group lasso | ~18 | **~640 (36×)** | 18 |

Effect on forgetting at matched learning (Adam lr 0.01 vs best SGD setting at least as good on new facts):
- KL: retention 0.909 with Adam vs 0.928 with SGD
- TF-IDF: 0.941 vs 0.961
- glasso: 0.916 vs 0.890

So the leak is large in mechanics but its cost to forgetting is modest and mixed in this tiny model. Whether
it matters at Qwen scale is an open, cheap-to-test question (see RUN_ON_3060.md).

## Caveats
- Tiny synthetic world. Results may not carry over to real models.
- Learning-rate grid is coarse (×3 steps); a finer grid could separate KL and glasso.
- Forgetting probes are old facts plus arithmetic only.

## Files
`cl_proxy.py` (testbed), `sweep.py`, `analyze.py`, `adam_leak.py`, `sweep.jsonl`, `adam.jsonl`,
`sweep_summary.txt`, `smf_glasso_sgd.patch` (adds `--sparse-scoring glasso`, `--sparse-lam`, `--optim sgd`
to the SMF-ICML-FG code).

# Round 2: Gary's "grow" + "remember and switch" ideas (`grow_switch.py`, `gs.jsonl`)
Each new fact gets 4 fresh memory slots (one "situation key" per training phrasing); old weights never change.
A switch decides which grown slots turn on for each token. Trained on phrasings 0–3, also tested on an unseen 5th.
3 seeds. Best setting shown per arm (full table in gs.jsonl).

| Arm | New facts (seen phrasings) | New facts (unseen phrasing) | Retention | Memory added |
|---|---|---|---|---|
| A. No growth (KL sparse, lr 3) | 0.502 ± 0.035 | 0.460 ± 0.024 | 0.971 | 0 |
| A. No growth (KL sparse, lr 10) | 0.775 ± 0.044 | 0.735 ± 0.045 | 0.863 | 0 |
| B. Growth + perfect switch | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 | +1,600 slots (+39%) |
| C. Growth + nearest-match (cosine ≥ 0.9) | 0.995–1.000 | 0.992–1.000 | 0.999 | +1,600 |
| D. Growth + lasso switch (alpha 0.05, lr 30) | 0.990 ± 0.005 | 0.979 ± 0.007 | 0.997 | +1,600 |

- Growth + a switch essentially solves overwriting here: about 99% vs at most 78% (and that 78% only at 0.86 retention).
- The lasso switch works, but simple nearest-match is slightly better. With one fact per question there is
  nothing for a sparse multi-pick to add.
- Caveat: this is close to retrieval (store per item, look it up), a known approach. The open questions are
  (1) questions needing several stored pieces at once, (2) using stored facts indirectly (reasoning, reverse
  questions), and (3) bounded growth (merging slots so memory doesn't grow forever).

# Round 3 (overnight): the three harder tests
All numbers are means over 3 seeds; "retention" = old facts + arithmetic kept (1.00 = no forgetting).
Raw data: t1.jsonl, t2.jsonl, t3.jsonl, gated.jsonl, recipe_results.jsonl; summaries: night_summary.txt, recipe_summary.txt.

## Test 1 — knowledge spread over two partial pieces (`tests13.py 1`)
Each fact mentioned twice (phrasings 0–1, later 2–3); each mention grows 2 slots and is learned only partially.
Write lr = 3 (where one piece alone is clearly partial), retention ≥ 0.99:

| Switch | New facts (seen) | New facts (unseen phrasing) |
|---|---|---|
| Perfect switch, one piece only | 0.405 ± 0.031 | 0.399 ± 0.031 |
| Perfect switch, both pieces | 0.708 ± 0.025 | 0.706 ± 0.022 |
| Nearest single match | 0.163 ± 0.021 | 0.165 ± 0.017 |
| **All matches above similarity 0.9** | **0.711 ± 0.029** | **0.706 ± 0.034** |
| Lasso | 0.150 ± 0.021 | 0.147 ± 0.017 |

**Correction (next morning):** my first explanation, that lasso "stops at one piece", was wrong. Diagnostics
(`t1_diag.py`): lasso turns on 3.95–3.98 of the right fact's 4 slots. The failure was a **scaling mismatch**: the
pieces were written to be switched on at weight 1 each, but lasso weights describe how to rebuild the *question*,
and with 4 near-identical keys (cosine 0.98) it splits about 1.0 of total weight across them (~0.24 each).
Fix: let lasso choose WHICH slots, then switch them fully on, and keep the off-switch (act only when some key has
cosine ≥ 0.9). Rerun, 3 seeds (`t1_gate.py`, `t1gate.jsonl`):

| Write lr | Switch | New (seen) | New (unseen phrasing) | Retention |
|---|---|---|---|---|
| 3 | Perfect, both pieces | 0.708 ± 0.025 | 0.706 ± 0.022 | 1.000 |
| 3 | Threshold 0.9 | 0.711 ± 0.029 | 0.706 ± 0.034 | 1.000 |
| 3 | **Gated lasso-select (0.9, alpha 0.2)** | **0.704 ± 0.029** | **0.704 ± 0.037** | **0.999** |
| 10 | Perfect, both pieces | 0.887 ± 0.011 | 0.892 ± 0.009 | 1.000 |
| 10 | Threshold 0.9 | 0.900 ± 0.005 | 0.901 ± 0.008 | 0.999 |
| 10 | **Gated lasso-select (0.9, alpha 0.2)** | **0.903 ± 0.006** | **0.899 ± 0.008** | **0.999** |

Without the gate, lasso-select matches on new facts but wrecks old ones (retention 0.75–0.82 at lr 3, 0.11–0.17 at
lr 10), because it always finds *some* slots to switch on. Conclusion: used correctly, lasso finds all the relevant
pieces and ties the perfect switch. It ties (not beats) a simple threshold here because the pieces are near-duplicates,
so both methods find the same set.

## Test 2 — indirect use: taught only a new person's CITY, asked their COUNTRY (`world2.py`)
New world: every city belongs to one of 10 countries; the model answers "the country of P is via [city] [country]"
(thinks out loud). After pretraining, it composes correctly for 99.0% of people it was never told the country of.
Silent (no think-out-loud) composition did NOT emerge (14–15%, chance 10%), so I switched formats.

| Method | New city | New country (indirect) | Old city | Old composed | Arithmetic |
|---|---|---|---|---|---|
| Full finetuning (lr 1e-4) | 0.077 | 0.165 | 0.470 | 0.918 | 0.974 |
| KL sparse memory (lr 10) | 0.768 | 0.610 | 0.796 | 0.696 | 0.998 |
| **Growth + nearest (cos ≥ 0.9, lr 100)** | **0.995** | **0.993** | **0.993** | **0.990** | **1.000** |
| Growth + lasso (alpha 0.05, lr 30) | 0.992 | 0.907 | 1.000 | 0.960 | 1.000 |

New knowledge stored in grown slots was **used indirectly**, combined with old knowledge (city → country), at 99%.
Caveat: with think-out-loud, the indirect question becomes "recall the city in a new context, then use old knowledge".
Silent multi-step use was not tested because even pretraining couldn't learn it at this size.

## Test 3 — bounded growth: memory capped below one slot per fact (`tests13.py 3`, `recipe.py`)
400 new facts, 1 slot each when unlimited (99.9% learned). Budgets 200 / 100 / 50 slots. When full:
evict oldest, merge with most similar slot, or **lasso reuse** (the lasso expresses the new fact's situation as a
sparse recipe of existing slots and trains those). Readers: nearest key, or **address → recipe** (each fact keeps a
small address pointing to the recipe it was written with). Retention ≥ 0.99 in every row below.

| Budget | Evict oldest | Merge nearest | **Lasso reuse + address→recipe** | Lasso reuse, perfect switch |
|---|---|---|---|---|
| 200 (50%) | 0.502 ± 0.002 | 0.517 ± 0.008 | **0.720 ± 0.001** | 0.714 ± 0.007 |
| 100 (25%) | 0.256 ± 0.003 | 0.277 ± 0.004 | **0.436 ± 0.006** | 0.427 ± 0.010 |
| 50 (12.5%) | 0.132 ± 0.002 | 0.151 ± 0.002 | **0.214 ± 0.025** | 0.215 ± 0.024 |

- Evict/merge store about **1 fact per slot** (50% budget → ~50%). Lasso reuse stores about **1.44 facts per slot**
  (0.72 × 400 / 200), i.e. +40% at half budget and +57–70% at a quarter.
- The address→recipe reader matches the perfect switch, so nothing is lost in reading.
- Without addresses, lasso-reuse memory reads back poorly (0.383 nearest-key; 0.386 gated lasso), because the
  reader has to re-guess each fact's recipe.
- An ungated lasso reader damages old facts (retention ~0.78); gating it (only act when a key is ≥ 0.8–0.9 similar)
  restores retention.

**Why it could matter:** the expensive part of memory is the values (in real memory layers ~1,024+ numbers each);
addresses can be small. Here keys and values are both 128 numbers, so the toy shows the *capacity* gain, not a
memory saving. Real savings depend on the address-to-value size ratio (e.g. 256 vs 1,024 in Lin et al.).
**Novelty is unchecked:** this resembles sparse coding / dictionary memories and sparse distributed memory (Kanerva,
1988). A literature search is needed before claiming anything.

# Literature check (Oct 13–14, 2026): the closest prior work is "lifelong model editing"
| Our result | Closest published method | How close |
|---|---|---|
| Growth + nearest-match + off-switch (round 2, test 2) | **GRACE** (NeurIPS 2023): codebook of keys + values at one layer; a key fires only if the query is within its "deferral radius". **NeuralDB** (ICLR 2026): key → residual value store with a cosine gate (0.65); 100,000 facts on Llama-3-8B | Essentially the same idea |
| Off-switch is essential (tests 1 and 3) | Every method above has a gate or threshold; **WISE** (NeurIPS 2024) learns a routing threshold | Known |
| **Lasso recipes + per-fact address in a capped memory (test 3)** | **MEMOIR** (NeurIPS 2025): ONE fixed-size shared residual memory; each edit writes into a sparse subset of columns (chosen by "TopHash" of activations); a database stores each edit's mask; at inference the closest stored mask is used if overlap ≥ τ, else memory is off. 15,000 sequential edits | Very close: bounded shared memory + sparse per-item pattern + stored pattern lookup + gate |
| Memory per edit reduction | **LadderEdit** (2026): low-rank compression per edit; memory still grows linearly | Different (no sharing) |

What remains different in test 3: MEMOIR picks each edit's slots with a fixed hash of activations (binary mask);
ours picks them with a lasso, i.e. chosen to *reuse existing slots that already explain the situation*, with
learned weights. That is an incremental variant of MEMOIR, not a new paradigm. A real claim would need
"lasso-chosen recipes beat TopHash masks" on MEMOIR's own benchmark (ZsRE, thousands of edits, 7B-class models).

# Head-to-head: lasso recipes vs MEMOIR-style TopHash masks (`memoir_vs_lasso.py`, `mvl_*.jsonl`, `mvl_summary.txt`)
Equal value-memory budget, 400 new facts, 3 seeds, retention ≥ 0.99, best tuned config per method.
MEMOIR-style got the larger tuning grid (54 configs × 4 thresholds vs 4 × 3 for lasso).
MEMOIR features: fixed random expansion of the frozen query (stand-in for real FFN activations); TopHash masks;
stored-mask lookup with an overlap threshold. (An early version had a feature-scaling bug that crippled it; fixed
before the real run.)

| Budget | Delete oldest (earlier) | MEMOIR-style | **Lasso recipes** | Lasso − MEMOIR per seed |
|---|---|---|---|---|
| 200 (50%) | 0.502 | 0.549 ± 0.005 | **0.720 ± 0.001** | +0.164, +0.176, +0.173 |
| 100 (25%) | 0.256 | 0.315 ± 0.015 | **0.436 ± 0.006** | +0.096, +0.121, +0.146 |
| 50 (12.5%) | 0.132 | 0.128 ± 0.005 | **0.214 ± 0.025** | +0.115, +0.048, +0.095 |

Lasso recipes win at every budget, in all 9 seed-by-budget comparisons (+5 to +18 points).
Main caveat: MEMOIR's real version selects columns from thousands of actual FFN activations; here its features
are a small random expansion whose size equals the budget. The real test is against real MEMOIR on its benchmark.
