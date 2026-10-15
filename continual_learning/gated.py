import json, sys, tests13 as T
T.READS3 = [("oracle", None), ("nearest1", 0.8), ("nearest1", 0.9)] + \
           [("gated_lasso", (tau, a)) for tau in [0.7, 0.8, 0.9] for a in [0.02, 0.05, 0.1]]
seeds = [int(s) for s in sys.argv[1].split(",")]
with open(sys.argv[2], "a") as f:
    for s in seeds:
        for strat in ["lasso_reuse", "evict", "merge"]:
            for B in [200, 100]:
                for r in T.test3(s, 30, B, strat):
                    f.write(json.dumps(r) + "\n")
                f.flush()
        print("seed", s, "done", flush=True)
