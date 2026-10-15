"""
Head-to-head at equal value-memory budget B (number of d-dim value vectors):

MEMOIR-style (Wang et al., NeurIPS 2025), adapted to the testbed:
  residual memory W (B x d), zero init, added to layer-2 output:  out += (m * a(x)) @ W
  a(x) = ReLU(R q + b) in R^B   (fixed random feature expansion of the frozen query q;
         stands in for MEMOIR's FFN activations; centered by the mean over 100 background prompts)
  TopHash: mask = fixed permutation of the top-k |centered a| indices
  edit: mask from the mean of the fact's 4 phrasing activations; SGD on W rows in the mask
  inference (per token): mask of the query; nearest stored edit mask by Hamming distance;
         if overlap |m_q & m_match| / k >= tau use m_match, else memory off
Lasso recipes (ours): tests13.test3(strategy="lasso_reuse") with the address->recipe reader.
"""
import json, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import cl_proxy as C, grow_switch as GS, tests13 as T


class Memoir:
    def __init__(self, dq, d, B, k, seed):
        g = torch.Generator().manual_seed(1234 + seed)
        self.R = torch.randn(B, dq, generator=g)          # unit-variance features (q is unit-norm)
        self.b = torch.randn(B, generator=g) * 0.1
        self.perm = torch.randperm(B, generator=g)
        self.W = torch.zeros(B, d); self.B, self.k = B, k
        self.mu = torch.zeros(B); self.masks = []          # stored edit masks (bool B)
        self.mode = "off"; self.fixed = None; self.tau = 0.5

    def act(self, q):
        return F.relu(q @ self.R.T + self.b)

    def mask_of(self, a):                                   # a: (N,B)
        idx = (a - self.mu).abs().topk(self.k, dim=1).indices
        m = torch.zeros_like(a, dtype=torch.bool)
        m.scatter_(1, self.perm[idx], True)
        return m

    def out(self, q):
        a = self.act(q)
        if self.mode == "off" or not self.masks:
            return torch.zeros(q.shape[0], self.W.shape[1])
        if self.mode == "fixed":
            M = self.fixed.unsqueeze(0).expand_as(a)
        else:
            mq = self.mask_of(a).float()                       # N,B
            S = torch.stack(self.masks).float()                # E,B
            ov = mq @ S.T                                      # N,E  overlap counts (= k - hamming/2)
            best = ov.max(1)
            M = S[best.indices] * (best.values / self.k >= self.tau).float().unsqueeze(1)
        return (M * a) @ self.W


def attach(model, mem):
    pkm = model.mem; orig = pkm.forward
    def fwd(x):
        o = orig(x)
        if mem.mode == "off":
            return o
        B_, T_, D_ = x.shape
        q = F.normalize(pkm.q(x).reshape(B_ * T_, -1), dim=1)
        return o + mem.out(q).reshape(B_, T_, D_)
    pkm.forward = fwd


def run_memoir(seed, B, k, lr, steps, taus):
    model, people, stream, rng, adds, old = T.base_setup(seed)
    mem = Memoir(model.mem.q.out_features, model.mem.values.weight.shape[1], B, k, seed); attach(model, mem)
    # centering mean from 100 background prompts
    bg = [C.fact_sentence(rng.choice(people), rng.choice(list(C.ATTR)), rng.randrange(5)) for _ in range(100)]
    mem.mu = mem.act(GS.answer_queries(model, bg)).mean(0)
    for (p, at) in stream:
        a = mem.act(GS.answer_queries(model, [C.fact_sentence(p, at, t) for t in range(4)])).mean(0, keepdim=True)
        m = mem.mask_of(a)[0]; mem.masks.append(m); mem.mode, mem.fixed = "fixed", m.float()
        for _ in range(steps):
            Wp = mem.W.clone().requires_grad_(True); mem.W = Wp
            ids = torch.tensor([C.encode(C.fact_sentence(p, at, rng.choice([0, 1, 2, 3]))) for _ in range(16)])
            C.lm_loss(model, ids).backward()
            with torch.no_grad(): mem.W = (Wp - lr * Wp.grad).detach()
    out = []
    for tau in taus:
        mem.mode, mem.tau = "route", tau
        r = dict(old_acc=C.accuracy(model, old), add_acc=C.accuracy(model, adds),
                 new_seen=C.accuracy(model, [C.fact_sentence(p, at, t) for (p, at) in stream for t in range(4)]),
                 new_heldout=C.accuracy(model, [C.fact_sentence(p, at, 4) for (p, at) in stream]))
        out.append({"method": "memoir", "seed": seed, "budget": B, "k": k, "lr": lr, "steps": steps, "tau": tau, **r})
    return out


def run_lasso(seed, B, lr, steps, taus):
    T.READS3 = [("recipe", t) for t in taus]
    orig_write = T.write
    T.write = lambda model, g, fn, slots, lr_, st, rng, phr: orig_write(model, g, fn, slots, lr_, steps, rng, phr)
    try:
        rows = T.test3(seed, lr, B, "lasso_reuse")
    finally:
        T.write = orig_write
    return [{"method": "lasso_recipe", "seed": seed, "budget": B, "k": None, "lr": lr, "steps": steps,
             "tau": r["param"], **{k: r[k] for k in ("old_acc", "add_acc", "new_seen", "new_heldout")}} for r in rows]


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]; outf = sys.argv[2]; part = sys.argv[3]
    with open(outf, "a") as f:
        for s in seeds:
            t0 = time.time()
            for B in [200, 100, 50]:
                if part in ("lasso", "all"):
                    for lr in [30, 100]:
                        for st in [3, 10]:
                            for r in run_lasso(s, B, lr, st, [0.7, 0.8, 0.9]): f.write(json.dumps(r) + "\n")
                            f.flush()
                if part in ("memoir", "all"):
                    for k in sorted({max(2, B // 16), max(2, B // 8), max(2, B // 4)}):
                        for lr in [1, 3, 10]:
                            for st in [3, 10]:
                                for r in run_memoir(s, B, k, lr, st, [0.6, 0.7, 0.8, 0.9]): f.write(json.dumps(r) + "\n")
                                f.flush()
            print(f"{part} seed {s} done {time.time()-t0:.0f}s", flush=True)
