import json, sys, cl_proxy as C
shard, n = (int(x) for x in sys.argv[1].split("/"))
cfgs = [(m, lr, t, lam, s) for s in [0, 1, 2]
        for (m, t, lam) in [("kl", 64, 0), ("tfidf", 64, 0), ("glasso", 0, 0.5)]
        for lr in [1e-3, 3e-3, 1e-2]]
cfgs += [(m, 3, t, lam, s) for s in [0, 1, 2] for (m, t, lam) in [("kl", 64, 0), ("tfidf", 64, 0), ("glasso", 0, 0.5)]]
with open(f"adam_{shard}.jsonl", "a") as f:
    for i, (m, lr, t, lam, s) in enumerate(cfgs):
        if i % n != shard: continue
        meth = m if lr == 3 else m + "_adam"
        r = C.continual(f"ckpt_seed{s}.pt", meth, lr, t=t, lam=lam, seed=s)
        f.write(json.dumps(r) + "\n"); f.flush()
        print(meth, lr, s, round(r["new_acc"],3), round(r["old_acc"],3), r["rows_per_step"], r["rows_changed_per_step"], flush=True)
