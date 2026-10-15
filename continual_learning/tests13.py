"""
Test 1 (knowledge spread across pieces) and Test 3 (bounded growth), using the
original pretrained checkpoints (ckpt_seed*.pt) and the grown-memory machinery.

Grown slot weights here are ADDITIVE: a slot that is on contributes w_j * V_j with
w_j = 1 for "on" (no averaging), so two partial pieces can add up.

TEST 1: each of 400 new facts is mentioned in two separate episodes at random points
  in the stream: episode A uses phrasings {0,1}, episode B phrasings {2,3}. Each
  episode grows 2 slots (keys = queries of its 2 phrasings) and trains them for ONE
  step, so a single episode is only a partial memory. Reads:
    oracle_one  : only episode A's slots           (what one piece alone gives)
    oracle_both : both episodes' slots              (ceiling for combining)
    nearest1    : single best key, if cosine >= tau
    nearest_sum : every key with cosine >= tau, weight 1 each
    lasso       : non-negative lasso over top-16 keys (alpha), weights used directly
  Evaluated on all phrasings 0-3 and the never-trained phrasing 4.

TEST 3: one slot per fact (key = mean of its 4 phrasing queries). Memory budget
  B in {400 (= no limit), 200, 100, 50} slots. When full:
    evict  : overwrite the oldest slot
    merge  : share the most similar existing slot (key -> normalized mean, keep training its value)
    lasso_reuse : lasso decides which existing slots already explain the new key and trains
                  those (weighted by lasso coefficients); falls back to merge if lasso picks none
  Reads: nearest1 (tau grid) and lasso (alpha grid). Plus no-growth KL baseline numbers
  come from earlier runs.
"""
import json, random, sys, time
import numpy as np, torch, torch.nn.functional as F
import cl_proxy as C, grow_switch as GS


class Grown2(GS.Grown):
    def weights(self, q):
        n = self.K.shape[0]
        W = torch.zeros(q.shape[0], n)
        if n == 0 or self.mode == "off":
            return W
        if self.mode == "fixed":
            for j, wt in self.active.items():
                W[:, j] = wt
            return W
        S = q @ self.K.T
        if self.mode == "nearest1":
            v, i = S.max(1)
            on = v >= self.param
            W[torch.arange(len(q))[on], i[on]] = 1.0
            return W
        if self.mode == "nearest_sum":
            return (S >= self.param).float()
        if self.mode == "lasso":
            return GS.Grown.weights(self, q)
        if self.mode == "gated_lasso_set":       # off unless a key is a close match; then lasso picks, full strength
            tau, alpha = self.param
            gate = (S.max(1).values >= tau).float().unsqueeze(1)
            self.mode, self.param = "lasso", alpha
            W = (GS.Grown.weights(self, q) > 1e-6).float() * gate
            self.mode, self.param = "gated_lasso_set", (tau, alpha)
            return W
        if self.mode in ("lasso_set", "lasso_max"):     # lasso chooses WHICH slots; then switch them fully on
            self.mode, mode = "lasso", self.mode
            W = GS.Grown.weights(self, q)
            self.mode = mode
            if mode == "lasso_set":
                return (W > 1e-6).float()
            return W / W.max(1, keepdim=True).values.clamp_min(1e-12)
        if self.mode == "recipe":               # nearest fact ADDRESS -> that fact's stored recipe
            tau = self.param
            Sa = q @ self.addr.T
            v, i = Sa.max(1)
            for row in torch.nonzero(v >= tau).flatten().tolist():
                for j, wt in self.recipes[int(i[row])].items():
                    W[row, j] = wt
            return W
        if self.mode == "gated_lasso":          # lasso only where some key is a close match
            tau, alpha = self.param
            gate = (S.max(1).values >= tau).float().unsqueeze(1)
            self.mode, self.param = "lasso", alpha
            W = GS.Grown.weights(self, q) * gate
            self.mode, self.param = "gated_lasso", (tau, alpha)
            return W
        raise ValueError(self.mode)


def lasso_vec(q, K, alpha, m=16):
    """non-negative lasso of one unit vector q over rows of K (top-m candidates)."""
    S = K @ q
    m = min(m, K.shape[0]); v, i = S.topk(m)
    Kc = K[i]; Gm = Kc @ Kc.T; b = Kc @ q; L = Gm.diagonal().sum().clamp_min(1e-6)
    w = torch.zeros(m)
    for _ in range(300):
        w = (w - (Gm @ w - b) / L - alpha / L).clamp_min(0)
    return {int(i[j]): float(w[j]) for j in range(m) if w[j] > 1e-6}


def write(model, g, sents_fn, slots_w, lr, steps, rng, phr):
    """train values of the slots in slots_w (dict slot->weight) on the fact's paraphrases."""
    g.mode, g.active = "fixed", slots_w
    for _ in range(steps):
        V = g.V.clone().requires_grad_(True); g.V = V
        ids = torch.tensor([C.encode(sents_fn(rng.choice(phr))) for _ in range(16)])
        C.lm_loss(model, ids).backward()
        with torch.no_grad():
            g.V = (V - lr * V.grad).detach()


def base_setup(seed):
    model, people, stream, rng = GS.setup(seed)
    adds, old, _, _ = GS.evals(model, people, stream)
    return model, people, stream, rng, adds, old


def read_eval(model, g, stream, adds, old, mode, param, owners=None):
    if mode.startswith("oracle"):
        g.mode = "off"
        o, a = C.accuracy(model, old), C.accuracy(model, adds)
        g.mode = "fixed"; seen = held = 0
        for fi, (p, at) in enumerate(stream):
            g.active = owners(fi)
            seen += C.accuracy(model, [C.fact_sentence(p, at, t) for t in range(4)]) * 4
            held += C.accuracy(model, [C.fact_sentence(p, at, 4)])
        return dict(old_acc=o, add_acc=a, new_seen=seen / (4 * len(stream)), new_heldout=held / len(stream))
    g.mode, g.param = mode, param
    return dict(old_acc=C.accuracy(model, old), add_acc=C.accuracy(model, adds),
                new_seen=C.accuracy(model, [C.fact_sentence(p, at, t) for (p, at) in stream for t in range(4)]),
                new_heldout=C.accuracy(model, [C.fact_sentence(p, at, 4) for (p, at) in stream]))


def test1(seed, lr):
    model, people, stream, rng, adds, old = base_setup(seed)
    g = Grown2(model.mem.q.out_features, model.mem.values.weight.shape[1]); GS.attach(model, g)
    episodes = [(fi, ep) for fi in range(len(stream)) for ep in (0, 1)]
    rng.shuffle(episodes)
    slot_of = {}
    for fi, ep in episodes:
        p, at = stream[fi]
        phr = [0, 1] if ep == 0 else [2, 3]
        keys = GS.answer_queries(model, [C.fact_sentence(p, at, t) for t in phr])
        st = g.K.shape[0]
        g.K = torch.cat([g.K, keys]); g.V = torch.cat([g.V, torch.zeros(2, g.V.shape[1])])
        slot_of[(fi, ep)] = [st, st + 1]
        write(model, g, lambda t: C.fact_sentence(p, at, t), {st: 1.0, st + 1: 1.0}, lr, 1, rng, phr)
    out = []
    reads = [("oracle_one", None), ("oracle_both", None)] + \
            [("nearest1", t) for t in [0.6, 0.7, 0.8, 0.9]] + \
            [("nearest_sum", t) for t in [0.6, 0.7, 0.8, 0.9]] + \
            [("lasso", a) for a in [0.02, 0.05, 0.1, 0.2, 0.3]]
    for mode, param in reads:
        if mode == "oracle_one":
            own = lambda fi: {s: 1.0 for s in slot_of[(fi, 0)]}
        elif mode == "oracle_both":
            own = lambda fi: {s: 1.0 for s in slot_of[(fi, 0)] + slot_of[(fi, 1)]}
        else:
            own = None
        r = read_eval(model, g, stream, adds, old, mode, param, own)
        out.append({"test": 1, "seed": seed, "lr": lr, "read": mode, "param": param, **r})
    return out


READS3 = [("oracle", None)] + [("nearest1", t) for t in [0.6, 0.7, 0.8, 0.9]] + \
         [("lasso", a) for a in [0.02, 0.05, 0.1, 0.2]]


def test3(seed, lr, budget, strategy, alpha_reuse=0.05):
    model, people, stream, rng, adds, old = base_setup(seed)
    g = Grown2(model.mem.q.out_features, model.mem.values.weight.shape[1]); GS.attach(model, g)
    age = []          # write time of each slot
    owner = {}        # fact -> {slot: weight} used when writing (for oracle read)
    for fi, (p, at) in enumerate(stream):
        key = F.normalize(GS.answer_queries(model, [C.fact_sentence(p, at, t) for t in range(4)]).mean(0), dim=0)
        n = g.K.shape[0]
        if n < budget:
            g.K = torch.cat([g.K, key[None]]); g.V = torch.cat([g.V, torch.zeros(1, g.V.shape[1])])
            age.append(fi); slots = {n: 1.0}
        elif strategy == "evict":
            j = int(np.argmin(age))
            g.K[j] = key; g.V[j] = 0; age[j] = fi; slots = {j: 1.0}
        elif strategy == "merge":
            j = int((g.K @ key).argmax())
            g.K[j] = F.normalize(g.K[j] + key, dim=0); age[j] = fi; slots = {j: 1.0}
        elif strategy == "lasso_reuse":
            w = lasso_vec(key, g.K, alpha_reuse)
            if not w:
                j = int((g.K @ key).argmax()); w = {j: 1.0}
            mx = max(w.values()); slots = {j: v / mx for j, v in w.items()}
            for j in slots: age[j] = fi
        owner[fi] = slots
        g.addr = torch.cat([getattr(g, "addr", torch.zeros(0, key.shape[0])), key[None]])
        g.recipes = getattr(g, "recipes", []) + [slots]
        write(model, g, lambda t: C.fact_sentence(p, at, t), slots, lr, 3, rng, [0, 1, 2, 3])
    out = []
    reads = READS3
    for mode, param in reads:
        r = read_eval(model, g, stream, adds, old, mode, param, (lambda fi: owner[fi]) if mode == "oracle" else None)
        out.append({"test": 3, "seed": seed, "lr": lr, "budget": budget, "strategy": strategy,
                    "read": mode, "param": param, "slots": int(g.K.shape[0]), **r})
    return out


if __name__ == "__main__":
    which, seeds, outf = sys.argv[1], [int(s) for s in sys.argv[2].split(",")], sys.argv[3]
    with open(outf, "a") as f:
        for s in seeds:
            t0 = time.time()
            if which == "1":
                for lr in [1, 3, 10]:
                    for r in test1(s, lr): f.write(json.dumps(r) + "\n")
                    f.flush()
            else:
                for lr in [30, 100]:
                    for strat in ["evict", "merge", "lasso_reuse"]:
                        for B in [400, 200, 100, 50]:
                            if B == 400 and strat != "evict":
                                continue   # identical to evict when nothing is ever full
                            for r in test3(s, lr, B, strat): f.write(json.dumps(r) + "\n")
                            f.flush()
            print(f"test {which} seed {s} done {time.time()-t0:.0f}s", flush=True)
