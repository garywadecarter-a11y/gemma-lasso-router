import json, sys, tests13 as T
T.READS3 = [("oracle", None), ("nearest1", 0.8), ("nearest1", 0.9)] + [("recipe", t) for t in [0.7, 0.8, 0.9]]
with open(sys.argv[2], "a") as f:
    for s in [int(x) for x in sys.argv[1].split(",")]:
        for strat in ["lasso_reuse", "merge", "evict"]:
            for B in [200, 100, 50]:
                for r in T.test3(s, 30, B, strat): f.write(json.dumps(r) + "\n")
                f.flush()
        print("seed", s, "done", flush=True)
