"""
Growth + switch experiment (Gary's two ideas), in the same tiny testbed.

Grown memory G sits beside the frozen product-key memory in layer 2:
    layer output = PKM(x) + sum_j w_j(q) * V_j
  q    = the frozen PKM query of the current token (unit-normalized, 128-d)
  K_j  = situation key of grown slot j (unit vector), V_j = its value (trainable)
  w(q) = the SWITCH: which grown slots turn on for this token, and how strongly.

Writing a new fact (stream uses phrasings 0-3 only):
  allocate 4 slots, one per phrasing; key = q at the answer-predicting position
  for that phrasing; values start at 0 and are trained with SGD on the fact's
  paraphrase batch (3 steps). During writing only the fact's own slots are on
  (the writer knows what it is writing). Old weights are never touched.

Reading (evaluation) switches:
  oracle   : new-fact questions use exactly that fact's slots; everything else uses none (ceiling)
  nearest  : top-k keys by cosine, on only if cosine >= tau, softmax over the kept ones
  lasso    : non-negative lasso   min_w 1/2||q - K_C^T w||^2 + alpha*||w||_1 ,  w >= 0
             over the 16 most similar keys C; w is used directly as the switch.
             If nothing explains q well enough, w = 0 and the model is unchanged.
Baseline A (no growth): KL sparse memory finetuning on the same 4-phrasing stream.

Metrics: new_seen (phrasings 0-3), new_heldout (phrasing 4, never trained),
old_acc and add_acc (forgetting), and slots added.
"""
import json, random, sys, time, os
import numpy as np, torch, torch.nn.functional as F
import cl_proxy as C

TRAIN_T = [0, 1, 2, 3]; HELD_T = 4; SLOTS_PER_FACT = 4


class Grown:
    def __init__(self, d_q, d):
        self.K = torch.zeros(0, d_q); self.V = torch.zeros(0, d); self.owner = []
        self.mode = "off"; self.param = None; self.active = None   # active: set of slot ids (oracle/write)

    def weights(self, q):                       # q: (N, d_q) unit vectors -> (N, n_slots)
        n = self.K.shape[0]
        W = torch.zeros(q.shape[0], n)
        if n == 0 or self.mode == "off":
            return W
        if self.mode == "fixed":                # oracle / write: given slot set, equal weights
            if self.active:
                idx = torch.tensor(sorted(self.active)); W[:, idx] = 1.0 / len(idx)
            return W
        S = q @ self.K.T                         # cosine similarity
        if self.mode == "nearest":
            tau, k = self.param
            v, i = S.topk(min(k, n), dim=1)
            keep = v >= tau
            logits = torch.where(keep, v / 0.05, torch.full_like(v, -1e9))
            p = torch.softmax(logits, 1) * keep.any(1, keepdim=True)
            p = p * keep
            W.scatter_(1, i, p)
            return W
        if self.mode == "lasso":
            alpha = self.param
            m = min(16, n)
            v, i = S.topk(m, dim=1)                         # candidates
            Kc = self.K[i]                                  # N, m, d_q
            G = Kc @ Kc.transpose(1, 2)                     # N, m, m
            b = (Kc @ q.unsqueeze(-1)).squeeze(-1)          # N, m   (= K q)
            L = G.diagonal(dim1=1, dim2=2).sum(1, keepdim=True).clamp_min(1e-6)  # trace >= max eig
            w = torch.zeros_like(b)
            for _ in range(200):                            # projected ISTA
                grad = (G @ w.unsqueeze(-1)).squeeze(-1) - b
                w = (w - grad / L - alpha / L).clamp_min(0)
            W.scatter_(1, i, w)
            return W
        raise ValueError(self.mode)


def attach(model, grown):
    pkm = model.mem
    orig = pkm.forward

    def fwd(x):
        out = orig(x)
        if grown.K.shape[0] == 0 or grown.mode == "off":
            return out
        B, T, D = x.shape
        q = F.normalize(pkm.q(x).reshape(B * T, -1), dim=1)
        W = grown.weights(q)
        return out + (W @ grown.V).reshape(B, T, D)
    pkm.forward = fwd


def answer_queries(model, sents):
    """q at the answer-predicting position for each sentence."""
    P, _ = C.answer_prompts(sents)
    cap = {}
    h = model.mem.q.register_forward_hook(lambda m, i, o: cap.__setitem__("q", o))
    out = []
    with torch.no_grad():
        for p in P:
            ids = torch.tensor([p])
            model(ids)
            out.append(F.normalize(cap["q"][0, len(p) - 1], dim=0))
    h.remove()
    return torch.stack(out)


def setup(seed):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    ck = torch.load(f"ckpt_seed{seed}.pt", weights_only=False)
    model = C.TinyLM(len(C.VOCAB)); model.load_state_dict(ck["model"]); model.eval()
    people = ck["people"]
    newp = C.make_people(100, rng, exclude={tuple(int(x[1:]) for x in p["name"].split()) for p in people})
    stream = [(p, a) for p in newp for a in C.ATTR]; rng.shuffle(stream)
    for p_ in model.parameters(): p_.requires_grad = False
    return model, people, stream, rng


def evals(model, people, stream):
    adds = [s for s, _, _ in C.add_sentences()]
    old = [C.fact_sentence(p, a, 0) for p in people[:250] for a in C.ATTR]
    seen = [C.fact_sentence(p, a, t) for (p, a) in stream for t in TRAIN_T]
    held = [C.fact_sentence(p, a, HELD_T) for (p, a) in stream]
    return adds, old, seen, held


def run_growth(seed, lr, read_cfgs):
    model, people, stream, rng = setup(seed)
    g = Grown(model.mem.q.out_features, model.mem.values.weight.shape[1]); attach(model, g)
    adds, old, seen, held = evals(model, people, stream)
    base_old, base_add = C.accuracy(model, old), C.accuracy(model, adds)
    # ---- write each fact into its own fresh slots ----
    for fi, (p, a) in enumerate(stream):
        keys = answer_queries(model, [C.fact_sentence(p, a, t) for t in TRAIN_T])
        start = g.K.shape[0]
        g.K = torch.cat([g.K, keys]); g.V = torch.cat([g.V, torch.zeros(SLOTS_PER_FACT, g.V.shape[1])])
        g.owner += [fi] * SLOTS_PER_FACT
        g.mode, g.active = "fixed", set(range(start, start + SLOTS_PER_FACT))
        for _ in range(3):
            V = g.V.clone().requires_grad_(True); g.V = V
            ids = torch.tensor([C.encode(C.fact_sentence(p, a, rng.choice(TRAIN_T))) for _ in range(16)])
            loss = C.lm_loss(model, ids); loss.backward()
            with torch.no_grad():
                g.V = (V - lr * V.grad).detach()
    g.V = g.V.detach()
    # ---- read with each switch ----
    res = []
    owners = torch.tensor(g.owner)
    for (mode, param) in read_cfgs:
        r = {"arm": mode, "param": param, "lr": lr, "seed": seed, "slots_added": int(g.K.shape[0])}
        if mode == "oracle":
            g.mode = "off"
            r["old_acc"], r["add_acc"] = C.accuracy(model, old), C.accuracy(model, adds)
            g.mode = "fixed"; ns = nh = 0
            for fi, (p, a) in enumerate(stream):
                g.active = set(torch.nonzero(owners == fi).flatten().tolist())
                ns += C.accuracy(model, [C.fact_sentence(p, a, t) for t in TRAIN_T]) * len(TRAIN_T)
                nh += C.accuracy(model, [C.fact_sentence(p, a, HELD_T)])
            r["new_seen"], r["new_heldout"] = ns / (len(stream) * len(TRAIN_T)), nh / len(stream)
        else:
            g.mode, g.param = mode, param
            r["old_acc"], r["add_acc"] = C.accuracy(model, old), C.accuracy(model, adds)
            r["new_seen"], r["new_heldout"] = C.accuracy(model, seen), C.accuracy(model, held)
        r["base_old"], r["base_add"] = base_old, base_add
        res.append(r)
    return res


def run_baseline(seed, lr, t=64):
    model, people, stream, rng = setup(seed)
    model.train()
    for n_, p_ in model.named_parameters(): p_.requires_grad = n_.endswith("ff.values.weight")
    acc = C.Access(model, people, seed=seed); vals = model.mem.values.weight
    adds, old, seen, held = evals(model, people, stream)
    base_old, base_add = C.accuracy(model, old), C.accuracy(model, adds)
    for (p, a) in stream:
        for _ in range(3):
            ids = torch.tensor([C.encode(C.fact_sentence(p, a, rng.choice(TRAIN_T))) for _ in range(16)])
            loss = C.lm_loss(model, ids); vals.grad = None; loss.backward()
            mult = C.select_rows("kl", C.batch_counts(model, ids), vals.grad, acc, t, 0)
            with torch.no_grad(): vals -= lr * vals.grad * mult.unsqueeze(1)
    return [{"arm": "A_kl_baseline", "param": t, "lr": lr, "seed": seed, "slots_added": 0,
             "old_acc": C.accuracy(model, old), "add_acc": C.accuracy(model, adds),
             "new_seen": C.accuracy(model, seen), "new_heldout": C.accuracy(model, held),
             "base_old": base_old, "base_add": base_add}]


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1].split(",")]
    out = sys.argv[2]
    READ = [("oracle", None)] + \
           [("nearest", (tau, k)) for tau in [0.5, 0.7, 0.8, 0.9, 0.95] for k in [1, 4]] + \
           [("lasso", al) for al in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]]
    with open(out, "a") as f:
        for s in seeds:
            t0 = time.time()
            rows = [r for lr in [1, 3, 10] for r in run_baseline(s, lr)]
            rows += [r for lr in [30, 100] for r in run_growth(s, lr, READ)]
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush(); print(f"seed {s} done {time.time()-t0:.0f}s", flush=True)
