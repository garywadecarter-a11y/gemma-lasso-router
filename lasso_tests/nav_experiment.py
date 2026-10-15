"""
Gary's maze idea, version A:
  1) learn a SPARSE graph over specialists with the lasso (neighborhood selection)
  2) a prompt enters at the best of P portals, then greedily hops to better-matching
     neighbors until no neighbor beats the current node ("arrive in the neighborhood")
  3) the normal LM would then answer with that specialist (not simulated here)

This script tests step 2 at increasing bank sizes K and compares:
  - global   : score the prompt against ALL K specialists (exhaustive, the ceiling)
  - lasso    : greedy navigation on the lasso-learned sparse graph
  - knn      : same navigation on a plain k-nearest-neighbor graph, same avg degree
  - random   : same navigation on a random graph, same avg degree (sanity floor)
Metrics: exact hit rate, "neighborhood" hit (true specialist is the stop node or one
of its neighbors), and number of similarity evaluations per prompt (compute).

Data is synthetic and hierarchical (16 topics > subtopics > specialists) so that the
bank can be scaled to thousands of specialists on a CPU. Assumptions are printed.
"""
import argparse, json, time
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_bank(K, d, rng, n_top=16, n_sub=8, s_sub=0.55, s_leaf=0.40):
    """Hierarchical specialist 'keys': topic -> subtopic -> leaf."""
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


def make_prompts(keys, n_per, sigma, rng):
    K, d = keys.shape
    y = np.repeat(np.arange(K), n_per)
    X = unit(keys[y] + sigma * rng.normal(size=(len(y), d)) / np.sqrt(d))
    return X, y


def lasso_graph(keys, Xtr, ytr, target_deg, n_cand, rng):
    """Neighborhood selection: for node i, regress its activation sim(x, key_i)
    on candidate nodes' activations sim(x, key_j) over prompts that land near i.
    Nonzero lasso coefficients become edges.  Alpha is tuned per node so the
    average out-degree lands near target_deg (tie-break by |beta|)."""
    K = len(keys)
    S_keys = keys @ keys.T
    np.fill_diagonal(S_keys, -np.inf)
    cand = np.argsort(-S_keys, axis=1)[:, :n_cand]
    by_class = [np.where(ytr == k)[0] for k in range(K)]
    nbrs = []
    for i in range(K):
        c = cand[i]
        idx = np.concatenate([by_class[i]] + [by_class[j] for j in c])
        A = Xtr[idx] @ keys[np.r_[i, c]].T          # activations, col 0 = node i
        yv, F = A[:, 0], A[:, 1:]
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)
        yv = yv - yv.mean()
        chosen = None
        for alpha in [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]:
            beta = Lasso(alpha=alpha, max_iter=2000, tol=1e-3).fit(F, yv).coef_
            nz = np.flatnonzero(np.abs(beta) > 1e-8)
            if len(nz) >= target_deg:
                chosen = nz[np.argsort(-np.abs(beta[nz]))][:target_deg]
                break
            chosen = nz
        if len(chosen) == 0:
            chosen = np.arange(min(2, len(c)))
        nbrs.append(set(c[chosen].tolist()))
    return symmetrize(nbrs)


def knn_graph(keys, deg):
    S = keys @ keys.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, :deg]
    return symmetrize([set(r.tolist()) for r in nn])


def random_graph(K, deg, rng):
    return symmetrize([set(rng.choice(np.delete(np.arange(K), i), deg, replace=False).tolist())
                       for i in range(K)])


def symmetrize(nbrs):
    out = [set(s) for s in nbrs]
    for i, s in enumerate(nbrs):
        for j in s:
            out[j].add(i)
    return [np.array(sorted(s)) for s in out]


def navigate(x, keys, nbrs, portals, beam=1, max_hops=50):
    """Enter at best portal(s), then greedy / beam hop to better neighbors."""
    evals = len(portals)
    ps = keys[portals] @ x
    frontier = list(portals[np.argsort(-ps)[:beam]])
    score = {int(p): float(s) for p, s in zip(portals, ps)}
    visited = set(frontier)
    for _ in range(max_hops):
        new = []
        for n in frontier:
            cand = [j for j in nbrs[n] if j not in score]
            if cand:
                s = keys[cand] @ x
                evals += len(cand)
                score.update({int(j): float(v) for j, v in zip(cand, s)})
            new.extend(nbrs[n].tolist())
        pool = sorted(set(frontier) | set(new), key=lambda j: -score[j])[:beam]
        if set(pool) <= visited and set(pool) == set(frontier):
            break
        visited |= set(pool)
        frontier = pool
    best = max(score, key=score.get)
    return best, evals


def run(K, seed, d, sigma, deg, n_portals, beam, n_per_train, n_per_test, n_cand):
    rng = np.random.default_rng(seed)
    keys, _ = make_bank(K, d, rng)
    Xtr, ytr = make_prompts(keys, n_per_train, sigma, rng)
    Xte, yte = make_prompts(keys, n_per_test, sigma, rng)
    if len(yte) > 4000:
        sel = rng.choice(len(yte), 4000, replace=False)
        Xte, yte = Xte[sel], yte[sel]

    km = KMeans(n_portals, n_init=3, random_state=seed).fit(keys)
    portals = np.array([np.argmax(keys @ unit(c)) for c in km.cluster_centers_])

    res = {"K": K, "seed": seed}
    glob = np.argmax(Xte @ keys.T, axis=1)
    res["global"] = {"hit": float((glob == yte).mean()), "evals": K}

    t0 = time.time()
    graphs = {"lasso": lasso_graph(keys, Xtr, ytr, deg, n_cand, rng)}
    res["lasso_build_s"] = time.time() - t0
    lasso_deg = float(np.mean([len(n) for n in graphs["lasso"]]))
    knn_k = max(1, int(round(deg)))
    graphs["knn"] = knn_graph(keys, knn_k)
    graphs["random"] = random_graph(K, knn_k, rng)

    for name, nbrs in graphs.items():
        hits, nbhd, evs, agree = [], [], [], []
        for x, t, g in zip(Xte, yte, glob):
            b, e = navigate(x, keys, nbrs, portals, beam=beam)
            hits.append(b == t)
            nbhd.append(b == t or t in nbrs[b])
            agree.append(b == g)
            evs.append(e)
        res[name] = {"hit": float(np.mean(hits)), "nbhd_hit": float(np.mean(nbhd)),
                     "agree_global": float(np.mean(agree)),
                     "evals": float(np.mean(evs)),
                     "avg_degree": float(np.mean([len(n) for n in nbrs]))}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[64, 256, 1024, 4096])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--deg", type=int, default=8)
    ap.add_argument("--portals", type=int, default=16)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--n_cand", type=int, default=48)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    allres = []
    for K in a.K:
        for s in a.seeds:
            r = run(K, s, a.d, a.sigma, a.deg, a.portals, a.beam, 20, 4, a.n_cand)
            allres.append(r)
            print(json.dumps(r), flush=True)
    json.dump({"args": vars(a), "results": allres}, open(a.out, "w"), indent=1)
