import json, sys, tests13 as T, collections, numpy as np
orig = T.test1
def reads_override(seed, lr):
    import grow_switch as GS
    return orig(seed, lr)
# patch the read list inside test1 by wrapping read_eval selection
src = open("tests13.py").read()
T_reads = [("oracle_both", None), ("nearest_sum", 0.9)] + \
          [("gated_lasso_set", (tau, a)) for tau in [0.8, 0.9] for a in [0.05, 0.1, 0.2]]
import types
code = T.test1.__code__
# simplest: re-define test1 with new reads by exec of modified source
s = src[src.index("def test1"):src.index("def test3")]
s = s.replace('''    reads = [("oracle_one", None), ("oracle_both", None)] + \\
            [("nearest1", t) for t in [0.6, 0.7, 0.8, 0.9]] + \\
            [("nearest_sum", t) for t in [0.6, 0.7, 0.8, 0.9]] + \\
            [("lasso", a) for a in [0.02, 0.05, 0.1, 0.2, 0.3]]''', "    reads = T_READS")
ns = dict(T.__dict__); ns["T_READS"] = T_reads
exec(s, ns)
with open(sys.argv[2], "a") as f:
    for sd in [int(x) for x in sys.argv[1].split(",")]:
        for lr in [3, 10]:
            for r in ns["test1"](sd, lr): f.write(json.dumps(r) + "\n")
            f.flush()
        print("seed", sd, "done", flush=True)
