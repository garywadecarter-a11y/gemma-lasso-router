"""Matched-size comparison. pool='oracle' -> candidates = 60 most similar keys
(exhaustive); pool='trace' -> candidates from 8 greedy tracings. For each selector
sweep its knob, record (mean set size, all-found rate on 2-skill prompts, on all
prompts), then interpolate to set size 2.0 and 3.0 so methods are compared at
equal budget."""
import json, sys, numpy as np, warnings
warnings.filterwarnings("ignore")
import lasso_select as L
from sklearn.cluster import KMeans

def evaluate(K, sigma, seed, pool_mode, n_test=1200):
    rng = np.random.default_rng(seed)
    keys, top_id = L.make_bank(K, 64, rng)
    X, Y = L.make_prompts(keys, top_id, n_test, sigma, rng)
    if pool_mode == "oracle":
        S = X @ keys.T
        P = []
        for s in S:
            o = np.argsort(-s)[:60]; P.append((o, s[o]))
    else:
        nbrs = L.knn_graph(keys, 8)
        km = KMeans(16, n_init=3, random_state=seed).fit(keys)
        portals = np.unique([np.argmax(keys @ L.unit(c)) for c in km.cluster_centers_])
        P = [L.tracings(x, keys, nbrs, portals, 8)[:2] for x in X]
    pool_rec = float(np.mean([len(set(np.asarray(p[0]).tolist()) & y)/len(y) for p, y in zip(P, Y)]))
    grids = {"topk":[1,2,3,4,5], "thresh":[0,.02,.04,.06,.08,.1,.13,.16,.2,.25,.3,.4],
             "omp":[1.5,1.2,.9,.75,.6,.45,.3,.2,.12,.06,.03,.01],
             "lasso":[.06,.04,.03,.02,.014,.01,.007,.005,.003,.002,.001,.0005,.0002]}
    fns = {"topk":L.sel_topk,"thresh":L.sel_thresh,"omp":L.sel_omp,"lasso":L.sel_lasso}
    out = {"K":K,"sigma":sigma,"seed":seed,"pool":pool_mode,"pool_recall":pool_rec}
    for name, grid in grids.items():
        curve=[]
        for k in grid:
            preds=[set(int(p[0][i]) for i in fns[name](x, keys[p[0]], p[1], k)) for x,p in zip(X,P)]
            r=L.score_sets(preds,Y)
            curve.append((r["size"], r["all_found_by_n"][2], r["all_found"], r["all_found_by_n"][1], r["precision"]))
        curve.sort()
        c=np.array(curve)
        res={}
        for tgt in (2.0,3.0):
            if c[:,0].min()<=tgt<=c[:,0].max():
                res[str(tgt)]={"n2":float(np.interp(tgt,c[:,0],c[:,1])),"all":float(np.interp(tgt,c[:,0],c[:,2])),"n1":float(np.interp(tgt,c[:,0],c[:,3]))}
        out[name]=res
    return out

if __name__=="__main__":
    rows=[]
    for pool in ["oracle","trace"]:
        for K in [256,1024]:
            for sg in [0.5,1.0]:
                for s in [0,1,2]:
                    r=evaluate(K,sg,s,pool); rows.append(r); print(json.dumps(r),flush=True)
    json.dump(rows,open("curves_results.json","w"),indent=1)
