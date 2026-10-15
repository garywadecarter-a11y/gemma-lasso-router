"""Grid sweep over methods x hyperparameters x seeds; resumable; writes sweep.jsonl."""
import json, os, sys, itertools, time
import cl_proxy as C

GRID = []
for lr in [1e-5, 3e-5, 1e-4, 3e-4]:
    GRID.append(("full_ft", lr, 0, 0))
for lr in [0.3, 1, 3, 10, 30]:
    GRID.append(("mem_dense", lr, 0, 0))
for m in ["tfidf", "kl", "grad", "ratio"]:
    for t in [16, 64, 256]:
        for lr in [1, 3, 10, 30]:
            GRID.append((m, lr, t, 0))
for lam in [0.25, 0.5, 1, 2, 4]:
    for lr in [1, 3, 10, 30]:
        GRID.append(("glasso", lr, 0, lam))

out = sys.argv[1] if len(sys.argv) > 1 else "sweep.jsonl"
seeds = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["0", "1", "2"])]
shard, nshard = (int(x) for x in (sys.argv[3].split("/") if len(sys.argv) > 3 else ["0", "1"]))
done = set()
if os.path.exists(out):
    for l in open(out):
        r = json.loads(l); done.add((r["method"], r["lr"], r["t"], r["lam"], r["seed"]))
t0 = time.time(); n = 0
with open(out, "a") as f:
    for seed in seeds:
        for gi, (m, lr, t, lam) in enumerate(GRID):
            if gi % nshard != shard:
                continue
            key = (m, lr, t, lam, seed)
            if key in done:
                continue
            r = C.continual(f"ckpt_seed{seed}.pt", m, lr, t=t, lam=lam, seed=seed)
            f.write(json.dumps(r) + "\n"); f.flush(); n += 1
            print(f"{n} {time.time()-t0:.0f}s {m} lr={lr} t={t} lam={lam} seed={seed} "
                  f"new={r['new_acc']:.3f} old={r['old_acc']:.3f} add={r['add_acc']:.3f}", flush=True)
print("SWEEP DONE", flush=True)
