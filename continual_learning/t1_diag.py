"""Why does lasso pick ~1 piece in test 1? Measure (a) similarity between a fact's two pieces,
(b) how many slots lasso turns on at the answer position, (c) where its weight goes."""
import torch, numpy as np, torch.nn.functional as F, cl_proxy as C, grow_switch as GS, tests13 as T
torch.manual_seed(0)
model, people, stream, rng, adds, old = T.base_setup(0)
g = T.Grown2(model.mem.q.out_features, model.mem.values.weight.shape[1]); GS.attach(model, g)
fact_of = []
for fi in range(100):                       # first 100 facts, both episodes, keys only
    p, at = stream[fi]
    for phr in ([0, 1], [2, 3]):
        k = GS.answer_queries(model, [C.fact_sentence(p, at, t) for t in phr])
        g.K = torch.cat([g.K, k]); g.V = torch.cat([g.V, torch.zeros(2, g.V.shape[1])]); fact_of += [fi, fi]
fact_of = torch.tensor(fact_of)
K = g.K
# (a) similarity within a fact vs between facts
same = [float(K[i] @ K[j]) for i in range(len(K)) for j in range(i+1, len(K)) if fact_of[i] == fact_of[j]]
diff = [float(K[i] @ K[j]) for i in range(0, len(K), 7) for j in range(i+1, len(K), 5) if fact_of[i] != fact_of[j]]
print(f"(a) cosine between the 4 keys of the SAME fact: mean {np.mean(same):.3f} (min {np.min(same):.3f})")
print(f"    cosine between keys of DIFFERENT facts:   mean {np.mean(diff):.3f} (95th pct {np.percentile(diff,95):.3f})")
# (b,c) lasso on held-out phrasing queries
Q = torch.cat([GS.answer_queries(model, [C.fact_sentence(*stream[fi], 4)]) for fi in range(100)])
for alpha in [0.005, 0.02, 0.05, 0.1]:
    g.mode, g.param = "lasso", alpha
    W = g.weights(Q)
    nz = (W > 1e-6).sum(1).float()
    own = torch.stack([W[i][fact_of == i].sum() for i in range(100)]); tot = W.sum(1)
    ownnz = torch.stack([(W[i][fact_of == i] > 1e-6).sum() for i in range(100)]).float()
    print(f"(b) lasso alpha={alpha}: slots on per question {nz.mean():.2f}; of the right fact's 4 slots on: {ownnz.mean():.2f}; "
          f"total weight {tot.mean():.2f} (right fact {own.mean():.2f})")
g.mode, g.param = "nearest_sum", 0.9
W = g.weights(Q); print(f"(b) threshold 0.9: slots on {(W>0).sum(1).float().mean():.2f}, right-fact slots on "
      f"{torch.stack([(W[i][fact_of==i]>0).sum() for i in range(100)]).float().mean():.2f}")
