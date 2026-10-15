"""Gate 1: on REAL MiniLM embeddings, does a lasso over route keys recover both
routes a mixed prompt needs, better than similarity top-k and Gary's L1 router?

Route keys = mean MiniLM embedding of each route's training prompts, regenerated
exactly from Gary's 10_train_multi_router.py (random.seed(42)).
Selectors are tuned on a validation half and scored on the other half,
repeated over 10 random splits (stratified by prompt type) to get spreads.
"""
import json, re, random, warnings
import numpy as np
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression, Lasso, OrthogonalMatchingPursuit
import gate1_prompts as GP

MINILM = "/mnt/user-data/uploads/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ROUTER_SRC = "/mnt/user-data/uploads/gemma-lasso-router/10_train_multi_router.py"


def router_training_data():
    src = open(ROUTER_SRC).read()
    src = src[:src.index('print(f"Total router examples')]
    src = src.replace('Path("router").mkdir(exist_ok=True)', '')
    ns = {}
    exec(src, ns)
    return ns["prompts"], ns["labels"]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main():
    emb = SentenceTransformer(MINILM, device="cpu")
    tr_p, tr_y = router_training_data()
    Xtr = emb.encode(tr_p, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    tr_y = np.array(tr_y)
    routes = sorted(set(tr_y))
    assert routes == GP.ROUTES, routes
    keys = unit(np.stack([Xtr[tr_y == r].mean(0) for r in routes]))
    lr = LogisticRegression(penalty="l1", solver="saga", C=0.5, max_iter=5000, random_state=42).fit(Xtr, tr_y)

    items = GP.mixed_prompts() + GP.single_prompts()
    X = emb.encode([it["prompt"] for it in items], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    truth = [set(it["routes"]) for it in items]
    kinds = np.array([it["kind"] for it in items])
    pairs = ["+".join(it["routes"]) for it in items]
    S = X @ keys.T
    P = lr.predict_proba(X)
    ridx = {r: i for i, r in enumerate(routes)}

    # ---------- additivity diagnostics on mixed prompts ----------
    diag = {"second_route_rank": [], "pair_key_vs_best_single": []}
    for x, s, it in zip(X, S, items):
        if it["kind"] != "mixed":
            continue
        a, b = (ridx[r] for r in it["routes"])
        order = list(np.argsort(-s))
        diag["second_route_rank"].append(max(order.index(a), order.index(b)) + 1)
        pair = unit(keys[a] + keys[b])
        diag["pair_key_vs_best_single"].append(float(x @ pair - max(s[a], s[b])))
    ranks = np.array(diag["second_route_rank"])
    additivity = {
        "both_routes_in_top2_by_similarity": float(np.mean(ranks <= 2)),
        "both_in_top3": float(np.mean(ranks <= 3)),
        "mean_rank_of_weaker_route": float(ranks.mean()),
        "share_where_sum_of_keys_fits_better_than_best_single": float(np.mean(np.array(diag["pair_key_vs_best_single"]) > 0)),
    }

    # ---------- selectors ----------
    def sel(name, knob, i):
        x, s, p = X[i], S[i], P[i]
        if name == "lr_top1":
            return {routes[int(np.argmax(p))]}
        if name == "cos_top1":
            return {routes[int(np.argmax(s))]}
        if name == "cos_topk":
            return {routes[j] for j in np.argsort(-s)[:knob]}
        if name == "lr_topk":
            return {routes[j] for j in np.argsort(-p)[:knob]}
        if name == "lr_thresh":
            keep = {routes[j] for j in np.flatnonzero(p >= knob)}
            return keep or {routes[int(np.argmax(p))]}
        if name == "cos_thresh":
            keep = {routes[j] for j in np.flatnonzero(s >= s.max() - knob)}
            return keep
        if name == "omp":
            w = OrthogonalMatchingPursuit(tol=knob, fit_intercept=False).fit(keys.T, x).coef_
            nz = np.flatnonzero(w > 1e-6)
            return {routes[j] for j in nz} or {routes[int(np.argmax(s))]}
        if name == "lasso":
            w = Lasso(alpha=knob, positive=True, fit_intercept=False, max_iter=10000).fit(keys.T, x).coef_
            nz = np.flatnonzero(w > 1e-6)
            return {routes[j] for j in nz} or {routes[int(np.argmax(s))]}

    grids = {"lr_top1": [0], "cos_top1": [0], "cos_topk": [2, 3], "lr_topk": [2, 3],
             "lr_thresh": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4], "cos_thresh": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.15],
             "omp": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
             "lasso": [0.03, 0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002, 0.001]}

    def metrics(idx, preds):
        mixed = [k for k in idx if kinds[k] == "mixed"]
        single = [k for k in idx if kinds[k] == "single"]
        both = np.mean([truth[k] <= preds[k] for k in mixed])
        one = np.mean([len(truth[k] & preds[k]) / 2 for k in mixed])
        s_hit = np.mean([truth[k] <= preds[k] for k in single])
        s_exact = np.mean([truth[k] == preds[k] for k in single])
        size = np.mean([len(preds[k]) for k in idx])
        prec = np.mean([len(truth[k] & preds[k]) / len(preds[k]) for k in idx])
        rec = np.mean([len(truth[k] & preds[k]) / len(truth[k]) for k in idx])
        f1 = 2 * prec * rec / (prec + rec)
        return dict(mixed_both=both, mixed_recall=one, single_hit=s_hit, single_exact=s_exact,
                    size=size, f1=f1)

    all_preds = {(n, k): {i: sel(n, k, i) for i in range(len(items))} for n, g in grids.items() for k in g}

    strata = {}
    for i, pr in enumerate(pairs):
        strata.setdefault(pr, []).append(i)
    results = {n: [] for n in grids}
    for split in range(10):
        rng = random.Random(split)
        val, test = [], []
        for idx in strata.values():
            idx = idx[:]; rng.shuffle(idx)
            h = len(idx) // 2
            val += idx[:h]; test += idx[h:]
        for n, g in grids.items():
            best = max(g, key=lambda k: metrics(val, all_preds[(n, k)])["f1"])
            m = metrics(test, all_preds[(n, best)]); m["knob"] = best
            results[n].append(m)

    summary = {}
    for n, rs in results.items():
        summary[n] = {k: [round(float(np.mean([r[k] for r in rs])), 3), round(float(np.std([r[k] for r in rs])), 3)]
                      for k in ["mixed_both", "mixed_recall", "single_hit", "single_exact", "size", "f1"]}
        summary[n]["knobs"] = sorted(set(r["knob"] for r in rs))

    # per-pair breakdown for lasso vs cos_topk vs lr_topk using full data, modal knob
    per_pair = {}
    for n in ["lr_top1", "cos_topk", "lr_topk", "omp", "lasso"]:
        knob = max(set(r["knob"] for r in results[n]), key=[r["knob"] for r in results[n]].count)
        preds = all_preds[(n, knob)]
        for pr, idx in strata.items():
            if "+" in pr:
                per_pair.setdefault(pr, {})[n] = round(float(np.mean([truth[k] <= preds[k] for k in idx])), 2)

    out = {"n_mixed": int((kinds == "mixed").sum()), "n_single": int((kinds == "single").sum()),
           "additivity": additivity, "summary": summary, "per_pair_both_found": per_pair}
    json.dump(out, open("gate1_results.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
