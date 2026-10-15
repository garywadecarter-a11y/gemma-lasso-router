# Synthetic navigation test (nav_experiment.py)

Greedy navigation from the best of 16 portals on a sparse graph (lasso-learned vs k-nearest-neighbor vs random, similar average degree),
synthetic hierarchical specialist keys, dimension 64. 'hit' = stop node equals the exhaustive-search best match. Mean over seeds.

| Beam | Bank size | Seeds | Exhaustive | Lasso graph | kNN graph | Random graph | Lasso evals/prompt | kNN evals/prompt |
|---|---|---|---|---|---|---|---|---|
| 1 | 64 | 3 | 1.000 | 0.993 | 1.000 | 0.414 | 28.0 | 24.2 |
| 1 | 256 | 3 | 0.993 | 0.933 | 0.948 | 0.138 | 38.5 | 30.3 |
| 1 | 1024 | 3 | 0.970 | 0.838 | 0.674 | 0.041 | 41.2 | 37.6 |
| 1 | 4096 | 3 | 0.912 | 0.479 | 0.096 | 0.008 | 46.5 | 33.4 |
| 4 | 64 | 3 | 1.000 | 1.000 | 1.000 | 0.853 | 46.1 | 36.3 |
| 4 | 256 | 3 | 0.993 | 0.992 | 0.951 | 0.456 | 83.3 | 69.7 |
| 4 | 1024 | 3 | 0.970 | 0.968 | 0.888 | 0.145 | 96.0 | 93.2 |
| 4 | 4096 | 1 | 0.906 | 0.845 | 0.111 | 0.034 | 123.9 | 82.2 |

Beam 4 at 4,096 has 1 seed only (run did not finish). Raw per-seed lines: nav_beam1.log, nav_beam4.log.
