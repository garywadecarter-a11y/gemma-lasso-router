"""
Tiny continual-learning testbed for sparse memory finetuning (CPU-sized).

World:  synthetic people with 4 attributes (city, job, year, hobby), stated with
        5 paraphrase templates each, plus an arithmetic skill ("add a b equals s").
Model:  4-layer transformer; layer 2's FFN is a product-key memory (PKM) with
        n_keys^2 slots, top-k lookup, softmax-weighted values.
Phase 1 (pretrain): learn 1,000 people (4,000 facts) + all additions 0..20.
Phase 2 (continual): stream NEW people's facts one at a time (batch = paraphrases
        of one fact, a few steps each). Measure at the end:
          new_acc  = accuracy on the streamed facts       (learning)
          old_acc  = accuracy on pretrained facts          (forgetting, knowledge)
          add_acc  = accuracy on the addition table        (forgetting, skill)
Methods (phase 2):
  full_ft       all weights, AdamW
  mem_dense     every accessed memory value row, SGD
  tfidf / kl    top-t accessed rows by TF-IDF / KL score (Lin et al. 2025; 2026 follow-up)
  grad          top-t accessed rows by gradient norm ||g_i||  (ablation)
  ratio         top-t accessed rows by ||g_i|| / w_i            (benefit / interference)
  glasso        Gary's rule: u_i = g_i * max(0, 1 - lam * w_i * gbar / ||g_i||)
                (closed form of a weighted group lasso; w_i = background usage of
                 slot i normalized to mean 1, gbar = mean accessed-row grad norm)
"""
import argparse, json, math, os, random, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("NTHREADS", "2")))

# ----------------------------------------------------------------- world ----
ATTR = {"city": 100, "job": 100, "year": 50, "hobby": 100}
TEMPL = {
    "city":  ["{p} lives in {a} .", "{p} resides in {a} .", "the home of {p} is {a} .",
              "{p} is a resident of {a} .", "{p} moved to {a} ."],
    "job":   ["{p} works as {a} .", "the job of {p} is {a} .", "{p} is employed as {a} .",
              "{p} earns a living as {a} .", "by trade {p} is {a} ."],
    "year":  ["{p} was born in {a} .", "the birth year of {p} is {a} .", "{p} arrived in {a} .",
              "{p} entered the world in {a} .", "{p} is born {a} ."],
    "hobby": ["{p} enjoys {a} .", "the hobby of {p} is {a} .", "{p} spends weekends on {a} .",
              "in free time {p} likes {a} .", "{p} is a fan of {a} ."],
}
ADD_T = "add {x} {y} equals {s} ."
N_FIRST, N_LAST = 120, 120


def build_vocab():
    words = set()
    for ts in TEMPL.values():
        for t in ts:
            words.update(w for w in t.split() if not w.startswith("{"))
    words.update(ADD_T.replace("{x}", "").replace("{y}", "").replace("{s}", "").split())
    toks = ["<pad>"] + sorted(words)
    toks += [f"F{i}" for i in range(N_FIRST)] + [f"L{i}" for i in range(N_LAST)]
    for a, n in ATTR.items():
        toks += [f"{a[0].upper()}{a[1]}{i}" for i in range(n)]
    toks += [f"N{i}" for i in range(41)]
    return {t: i for i, t in enumerate(toks)}


VOCAB = build_vocab()
ITOS = {i: t for t, i in VOCAB.items()}
MAXLEN = 12


def attr_tok(a, v):
    return f"{a[0].upper()}{a[1]}{v}"


def make_people(n, rng, exclude=set()):
    out, seen = [], set(exclude)
    while len(out) < n:
        f, l = rng.randrange(N_FIRST), rng.randrange(N_LAST)
        if (f, l) in seen:
            continue
        seen.add((f, l))
        out.append({"name": f"F{f} L{l}", **{a: rng.randrange(n_) for a, n_ in ATTR.items()}})
    return out


def fact_sentence(person, a, t):
    return TEMPL[a][t].format(p=person["name"], a=attr_tok(a, person[a]))


def encode(s):
    ids = [VOCAB[w] for w in s.split()]
    return ids + [0] * (MAXLEN - len(ids))


def add_sentences():
    return [(ADD_T.format(x=f"N{x}", y=f"N{y}", s=f"N{x+y}"), x, y) for x in range(21) for y in range(21)]


# ----------------------------------------------------------------- model ----
class PKM(nn.Module):
    def __init__(self, d, n_keys=64, k=8, heads=2, dk=64):
        super().__init__()
        self.n, self.k, self.h, self.dk = n_keys, k, heads, dk
        self.q = nn.Linear(d, heads * dk)
        self.keys = nn.Parameter(torch.randn(heads, 2, n_keys, dk // 2) / math.sqrt(dk // 2))
        self.values = nn.Embedding(n_keys * n_keys, d)
        nn.init.normal_(self.values.weight, std=d ** -0.5)
        self.last_idx = None

    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, 2, self.dk // 2)
        s1 = torch.einsum("bthd,hnd->bthn", q[..., 0, :], self.keys[:, 0])
        s2 = torch.einsum("bthd,hnd->bthn", q[..., 1, :], self.keys[:, 1])
        v1, i1 = s1.topk(self.k, -1)
        v2, i2 = s2.topk(self.k, -1)
        all_s = (v1.unsqueeze(-1) + v2.unsqueeze(-2)).flatten(-2)
        all_i = (i1.unsqueeze(-1) * self.n + i2.unsqueeze(-2)).flatten(-2)
        sc, pos = all_s.topk(self.k, -1)
        idx = all_i.gather(-1, pos)                       # B,T,H,k
        w = F.softmax(sc, -1)
        out = (self.values(idx) * w.unsqueeze(-1)).sum((-2, -3))
        self.last_idx = idx
        return out


class Block(nn.Module):
    def __init__(self, d, nh, mem=False):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ff = PKM(d) if mem else nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.ln2(x))


class TinyLM(nn.Module):
    def __init__(self, V, d=128, nl=4, nh=4, mem_layer=2):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(MAXLEN, d)
        self.blocks = nn.ModuleList([Block(d, nh, mem=(i == mem_layer)) for i in range(nl)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
        self.mem = self.blocks[mem_layer].ff
        self.register_buffer("cmask", torch.triu(torch.full((MAXLEN, MAXLEN), float("-inf")), 1))

    def forward(self, ids):
        T = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(T))
        for b in self.blocks:
            x = b(x, self.cmask[:T, :T])
        return self.head(self.ln(x))


def lm_loss(model, ids):
    logits = model(ids[:, :-1])
    tgt = ids[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=0)


# ------------------------------------------------------------ evaluation ----
def answer_prompts(sents):
    """For each sentence, prompt = tokens before the answer (token before '.')."""
    P, A = [], []
    for s in sents:
        w = s.split()
        P.append([VOCAB[t] for t in w[:-2]]); A.append(VOCAB[w[-2]])
    return P, A


@torch.no_grad()
def accuracy(model, sents, bs=512):
    model.eval()
    P, A = answer_prompts(sents)
    correct = 0
    for i in range(0, len(P), bs):
        p, a = P[i:i + bs], A[i:i + bs]
        L = max(len(x) for x in p)
        ids = torch.tensor([x + [0] * (L - len(x)) for x in p])
        lens = torch.tensor([len(x) for x in p])
        logits = model(ids)
        last = logits[torch.arange(len(p)), lens - 1]
        correct += (last.argmax(-1) == torch.tensor(a)).sum().item()
    model.train()
    return correct / len(P)


# ------------------------------------------------------------- pretrain ----
def pretrain(seed, out, steps=30000, bs=128, n_people=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    people = make_people(n_people, rng)
    corpus = [fact_sentence(p, a, t) for p in people for a in ATTR for t in range(5)]
    adds = [s for s, _, _ in add_sentences()]
    corpus_ids = torch.tensor([encode(s) for s in corpus])
    add_ids = torch.tensor([encode(s) for s in adds])
    model = TinyLM(len(VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for step in range(steps):
        nf = int(bs * 0.85)
        b = torch.cat([corpus_ids[torch.randint(len(corpus_ids), (nf,))],
                       add_ids[torch.randint(len(add_ids), (bs - nf,))]])
        loss = lm_loss(model, b)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == steps - 1:
            ev = [fact_sentence(p, a, 0) for p in people[:500] for a in ATTR]
            print(f"seed {seed} step {step} loss {loss.item():.3f} fact_acc {accuracy(model, ev):.3f} "
                  f"add_acc {accuracy(model, adds):.3f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "people": people}, out)


# ------------------------------------------------------------- continual ----
class Access:
    """Background usage statistics of memory slots (from pretraining data)."""

    def __init__(self, model, people, n_batches=200, bs=32, seed=0):
        rng = random.Random(seed + 7)
        M = model.mem.values.weight.shape[0]
        self.count = torch.zeros(M); self.df = torch.zeros(M); self.n = n_batches
        adds = [s for s, _, _ in add_sentences()]
        with torch.no_grad():
            for _ in range(n_batches):
                sents = [fact_sentence(rng.choice(people), rng.choice(list(ATTR)), rng.randrange(5))
                         if rng.random() < 0.85 else rng.choice(adds) for _ in range(bs)]
                ids = torch.tensor([encode(s) for s in sents])
                model(ids[:, :-1])
                idx = model.mem.last_idx
                valid = (ids[:, :-1] != 0)
                c = torch.bincount(idx[valid].reshape(-1), minlength=M).float()
                self.count += c; self.df += (c > 0).float()
        self.w = (self.count + 1) / (self.count + 1).mean()          # interference weight, mean 1


def batch_counts(model, ids):
    M = model.mem.values.weight.shape[0]
    valid = (ids[:, :-1] != 0)
    return torch.bincount(model.mem.last_idx[valid].reshape(-1), minlength=M).float()


def select_rows(method, counts, g, acc, t, lam):
    """Return per-row multiplier (M,) applied to the value gradient."""
    accessed = counts > 0
    gn = g.norm(dim=1)
    mult = torch.zeros_like(counts)
    if method == "mem_dense":
        mult[accessed] = 1.0
        return mult
    if method == "glasso":
        gbar = gn[accessed].mean().clamp_min(1e-12)
        shrink = (1 - lam * acc.w * gbar / gn.clamp_min(1e-12)).clamp_min(0)
        mult[accessed] = shrink[accessed]
        return mult
    p = counts / counts.sum()
    if method == "tfidf":
        score = p * torch.log((acc.n + 1) / (acc.df + 1))
    elif method == "kl":
        pbg = (acc.count + 1) / (acc.count + 1).sum()
        score = p * torch.log((p + 1e-8) / (pbg + 1e-8))
    elif method == "grad":
        score = gn.clone()
    elif method == "ratio":
        score = gn / acc.w
    else:
        raise ValueError(method)
    score[~accessed] = -float("inf")
    k = min(int(t), int(accessed.sum()))
    mult[score.topk(k).indices] = 1.0
    return mult


def continual(ckpt, method, lr, t=64, lam=1.0, n_new=100, steps_per_fact=3, bs=16, seed=0,
              eval_old=1000):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    ck = torch.load(ckpt, weights_only=False)
    model = TinyLM(len(VOCAB)); model.load_state_dict(ck["model"]); model.train()
    people = ck["people"]
    new_people = make_people(n_new, rng, exclude={tuple(int(x[1:]) for x in p["name"].split()) for p in people})
    stream = [(p, a) for p in new_people for a in ATTR]
    rng.shuffle(stream)
    adds = [s for s, _, _ in add_sentences()]
    old_eval = [fact_sentence(p, a, 0) for p in people[:eval_old // 4] for a in ATTR]
    new_eval = [fact_sentence(p, a, tt) for (p, a) in stream for tt in range(5)]
    base = {"old_acc": accuracy(model, old_eval), "add_acc": accuracy(model, adds)}

    if method == "full_ft":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    else:
        for n_, p_ in model.named_parameters():
            p_.requires_grad = (n_ == "mem.values.weight" or n_.endswith("ff.values.weight"))
        acc = Access(model, people, seed=seed)
    vals = model.mem.values.weight
    rows_updated = []; rows_changed = []
    base_method, use_adam = (method[:-5], True) if method.endswith("_adam") else (method, False)
    if use_adam:
        vopt = torch.optim.AdamW([vals], lr=lr, weight_decay=0.0)
    for (p, a) in stream:
        for _ in range(steps_per_fact):
            sents = [fact_sentence(p, a, rng.randrange(5)) for _ in range(bs)]
            ids = torch.tensor([encode(s) for s in sents])
            loss = lm_loss(model, ids)
            if method == "full_ft":
                opt.zero_grad(); loss.backward(); opt.step(); continue
            vals.grad = None
            loss.backward()
            counts = batch_counts(model, ids)
            g = vals.grad
            mult = select_rows(base_method, counts, g, acc, t, lam)
            rows_updated.append(int((mult > 0).sum()))
            before = vals.detach().clone()
            if use_adam:
                vals.grad = g * mult.unsqueeze(1); vopt.step()
            else:
                with torch.no_grad():
                    vals -= lr * g * mult.unsqueeze(1)
            rows_changed.append(int(((vals.detach() - before).abs().sum(1) > 0).sum()))
    res = {"method": method, "lr": lr, "t": t, "lam": lam, "seed": seed,
           "new_acc": accuracy(model, new_eval), "old_acc": accuracy(model, old_eval),
           "add_acc": accuracy(model, adds), "base_old": base["old_acc"], "base_add": base["add_acc"],
           "rows_per_step": float(np.mean(rows_updated)) if rows_updated else None,
           "rows_changed_per_step": float(np.mean(rows_changed)) if rows_changed else None}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pretrain", "run"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--method"); ap.add_argument("--lr", type=float)
    ap.add_argument("--t", type=float, default=64); ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    ck = a.ckpt or f"ckpt_seed{a.seed}.pt"
    if a.cmd == "pretrain":
        pretrain(a.seed, ck, steps=a.steps)
    else:
        print(json.dumps(continual(ck, a.method, a.lr, a.t, a.lam, seed=a.seed)))
