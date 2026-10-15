"""
Step 1 of Gary's idea: "many cheap tracings, then a lasso keeps the few (2-5)
neighborhoods that actually explain the problem."

Synthetic setup
  - A bank of K specialists, each with a unit "key" vector. The bank is hierarchical
    (16 topics > 8 subtopics > leaves), so many specialists have near-duplicate
    siblings -- the situation where naive top-k wastes slots.
  - A prompt is a blend of n = 1, 2 or 3 specialists from different topics:
        x = normalize( sum_i a_i * key_i  +  noise )
    with blend shares a ~ Dirichlet(2) and noise std sigma / sqrt(d) per dimension.
  - Candidates: T greedy "tracings" on a sparse kNN map, each starting from a
    different one of the best-scoring portals; the union of visited nodes is the
    candidate pool (typically 20-60 nodes, versus K in the bank).

Selectors (each returns a set S, at most 5 specialists)
  - top1   : best single specialist (what the current gemma-lasso-router does)
  - topk   : k most similar candidates, k tuned on validation
  - thresh : all candidates within a tuned margin of the best score
  - omp    : orthogonal matching pursuit (classic greedy sparse coding),
             stopping residual tuned on validation
  - lasso  : non-negative lasso   min ||x - C w||^2/(2m) + alpha*sum|w|, w >= 0
             over candidate keys C; alpha tuned on validation; keep nonzero w

Every selector's single knob is tuned on a validation set to maximize F1, then
scored on a separate test set. Metrics: all-found (every true component kept),
recall, precision, mean |S|.
"""
import argparse, json
import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.cluster import KMeans

MAX_KEEP = 5


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_bank(K, d, rng, n_top=16, n_sub=8, s_sub=0.55, s_leaf=0.40):
    tops = unit(rng.normal(size=(n_top, d)))
    per_sub = max(1, K // (n_top * n_sub))
    keys, top_id = [], []
    for t in range(n_top):
        for _ in range(n_sub):
            sub = unit(tops[t] + s_sub * unit(rng.normal(size=d)))
            for _ in range(per_sub):
                keys.append(unit(sub + s_leaf * unit(rng.normal(size=d))))
                top_id.append(t)
    return np.array(keys[:K]), np.array(top_id[:K])


def make_prompts(keys, top_id, n, sigma, rng):
    K, d = keys.shape
    X, Y = [], []
    for _ in range(n):
        m = rng.choice([1, 2, 3])
        topics = rng.choice(16, m, replace=False)          # different topics
        comps = [rng.choice(np.flatnonzero(top_id == t)) for t in topics]
        a = rng.dirichlet([2.0] * m) if m > 1 else np.array([1.0])
        x = (a[:, None] * keys[comps]).sum(0) + sigma * rng.normal(size=d) / np.sqrt(d)
        X.append(unit(x)); Y.append(set(int(c) for c in comps))
    return np.array(X), Y


def knn_graph(keys, deg):
    S = keys @ keys.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :deg]
    out = [set(r.tolist()) for r in nn]
    for i, r in enumerate(nn):
        for j in r:
            out[j].add(i)
    return [np.array(sorted(s)) for s in out]


def tracings(x, keys, nbrs, portals, T, max_hops=40):
    """T greedy walks, each from a different top-scoring portal. Returns the
    union of visited nodes (candidate pool) and the number of key evaluations."""
    evals = len(portals)
    ps = keys[portals] @ x
    starts = portals[np.argsort(-ps)[:T]]
    score = {int(p): float(s) for p, s in zip(portals, ps)}
    pool = set()
    for st in starts:
        cur = int(st)
        for _ in range(max_hops):
            nb = [j for j in nbrs[cur] if j not in score]
            if nb:
                s = keys[nb] @ x; evals += len(nb)
                score.update({int(j): float(v) for j, v in zip(nb, s)})
            pool.add(cur); pool.update(nbrs[cur].tolist())
            best = max(nbrs[cur].tolist() + [cur], key=lambda j: score[j])
            if best == cur:
                break
            cur = best
    pool = np.array(sorted(pool))
    return pool, np.array([score[j] for j in pool]), evals


# ---------------- selectors ----------------
def sel_top1(x, C, s, knob):
    return [int(np.argmax(s))]


def sel_topk(x, C, s, k):
    return list(np.argsort(-s)[:int(k)])


def sel_thresh(x, C, s, margin):
    o = np.argsort(-s)
    keep = [i for i in o if s[i] >= s[o[0]] - margin]
    return keep[:MAX_KEEP]


def sel_omp(x, C, s, tol):
    m = OrthogonalMatchingPursuit(n_nonzero_coefs=None, tol=tol, fit_intercept=False)
    # restrict to the 30 most similar candidates (same as lasso) for fairness
    o = np.argsort(-s)[:30]
    try:
        w = m.fit(C[o].T, x).coef_
    except Exception:
        return [int(o[0])]
    nz = np.flatnonzero(w > 1e-6)
    if len(nz) == 0:
        return [int(o[0])]
    nz = nz[np.argsort(-w[nz])][:MAX_KEEP]
    return [int(o[i]) for i in nz]


def sel_lasso(x, C, s, alpha):
    o = np.argsort(-s)[:30]
    w = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=5000).fit(C[o].T, x).coef_
    nz = np.flatnonzero(w > 1e-6)
    if len(nz) == 0:
        return [int(o[0])]
    nz = nz[np.argsort(-w[nz])][:MAX_KEEP]
    return [int(o[i]) for i in nz]


SELECTORS = {
    "top1":   (sel_top1,   [0]),
    "topk":   (sel_topk,   [2, 3, 4, 5]),
    "thresh": (sel_thresh, [0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3]),
    "omp":    (sel_omp,    [0.9, 0.75, 0.6, 0.45, 0.3, 0.2, 0.12, 0.06, 0.03]),
    "lasso":  (sel_lasso,  [3e-2, 2e-2, 1.4e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4]),
}


def score_sets(preds, truths):
    allf = np.mean([t <= p for p, t in zip(preds, truths)])
    rec = np.mean([len(p & t) / len(t) for p, t in zip(preds, truths)])
    prec = np.mean([len(p & t) / max(1, len(p)) for p, t in zip(preds, truths)])
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    size = np.mean([len(p) for p in preds])
    by_m = {}
    for m in (1, 2, 3):
        idx = [i for i, t in enumerate(truths) if len(t) == m]
        by_m[m] = float(np.mean([truths[i] <= preds[i] for i in idx]))
    return dict(all_found=float(allf), recall=float(rec), precision=float(prec),
                f1=float(f1), size=float(size), all_found_by_n=by_m)


def run(K, sigma, seed, d=64, deg=8, n_portals=16, T=8, n_val=600, n_test=1500):
    rng = np.random.default_rng(seed)
    keys, top_id = make_bank(K, d, rng)
    nbrs = knn_graph(keys, deg)
    km = KMeans(n_portals, n_init=3, random_state=seed).fit(keys)
    portals = np.unique([np.argmax(keys @ unit(c)) for c in km.cluster_centers_])

    def prep(n):
        X, Y = make_prompts(keys, top_id, n, sigma, rng)
        P = [tracings(x, keys, nbrs, portals, T) for x in X]
        return X, Y, P

    Xv, Yv, Pv = prep(n_val)
    Xt, Yt, Pt = prep(n_test)

    def apply(fn, knob, X, P):
        out = []
        for x, (pool, s, _) in zip(X, P):
            loc = fn(x, keys[pool], s, knob)
            out.append(set(int(pool[i]) for i in loc))
        return out

    res = {"K": K, "sigma": sigma, "seed": seed,
           "pool_size": float(np.mean([len(p[0]) for p in Pt])),
           "key_evals": float(np.mean([p[2] for p in Pt])),
           "pool_recall": float(np.mean([len(set(p[0].tolist()) & y) / len(y) for p, y in zip(Pt, Yt)]))}
    for name, (fn, grid) in SELECTORS.items():
        best = max(grid, key=lambda k: score_sets(apply(fn, k, Xv, Pv), Yv)["f1"])
        r = score_sets(apply(fn, best, Xt, Pt), Yt)
        r["knob"] = best
        res[name] = r
        # budgeted: best all_found with mean |S| <= 3 on validation
        vs = {k: score_sets(apply(fn, k, Xv, Pv), Yv) for k in grid}
        ok = [k for k in grid if vs[k]["size"] <= 3.0] or [min(grid, key=lambda k: vs[k]["size"])]
        bk = max(ok, key=lambda k: vs[k]["all_found"])
        rb = score_sets(apply(fn, bk, Xt, Pt), Yt); rb["knob"] = bk
        res[name + "_budget3"] = rb
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="lasso_select_results.json")
    a = ap.parse_args()
    allres = []
    for K in a.K:
        for sg in a.sigma:
            for s in a.seeds:
                r = run(K, sg, s)
                allres.append(r)
                print(json.dumps({k: (v if not isinstance(v, dict) else
                      {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
                      for k, v in r.items()}), flush=True)
    json.dump(allres, open(a.out, "w"), indent=1)
