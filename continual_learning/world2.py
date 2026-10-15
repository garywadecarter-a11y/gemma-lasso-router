"""
Test 2 world: every city belongs to one of 10 countries.
Pretraining corpus:
  - person facts (4 attributes x 5 phrasings), as in cl_proxy
  - city -> country facts:  "{c} is in {k} ."  and  "the country of {c} is {k} ."
  - person -> country questions for 70% of people: "the country of {p} is {k} ."
The other 30% of pretrained people never get a country sentence; accuracy on them
measures whether the model learned to COMPOSE person->city->country (2-hop).
Continual phase: new people are taught ONLY their city (phrasings 0-3), then asked
their country. Methods: full finetuning, KL sparse memory, growth+nearest, growth+lasso.
"""
import json, random, sys, time, os
import numpy as np, torch, torch.nn.functional as F
import cl_proxy as C

# ---- extend vocabulary (must happen before any model is built) ----
for w in ["country", "via"] + [f"K{i}" for i in range(10)]:
    if w not in C.VOCAB:
        C.VOCAB[w] = len(C.VOCAB)
C.ITOS.update({i: t for t, i in C.VOCAB.items()})
import grow_switch as G   # noqa: E402  (uses C.VOCAB at call time)

CITY_COUNTRY = [i % 10 for i in range(100)]
random.Random(123).shuffle(CITY_COUNTRY)


def country_of(person):
    return CITY_COUNTRY[person["city"]]


def country_q(person):
    # think-out-loud format: recall the city first, then its country
    return f"the country of {person['name']} is via Ci{person['city']} K{country_of(person)} ."


@torch.no_grad()
def cot_acc(model, persons, need_city=False):
    """Greedy-generate 2 tokens after 'the country of P is via'; correct if 2nd = country
    (and, if need_city, 1st = city)."""
    model.eval()
    ok = 0
    for i in range(0, len(persons), 256):
        ps = persons[i:i + 256]
        prompts = [[C.VOCAB[w] for w in f"the country of {p['name']} is via".split()] for p in ps]
        ids = torch.tensor(prompts)
        t1 = model(ids)[:, -1].argmax(-1)
        ids2 = torch.cat([ids, t1[:, None]], 1)
        t2 = model(ids2)[:, -1].argmax(-1)
        for p, a, b in zip(ps, t1.tolist(), t2.tolist()):
            good = b == C.VOCAB[f"K{country_of(p)}"]
            if need_city:
                good = good and a == C.VOCAB[f"Ci{p['city']}"]
            ok += good
    return ok / len(persons)


def city_country_sents():
    out = []
    for c in range(100):
        out += [f"Ci{c} is in K{CITY_COUNTRY[c]} .", f"the country of Ci{c} is K{CITY_COUNTRY[c]} ."]
    return out


def pretrain(seed, out, steps=8000, bs=128, n_people=1000):
    rng = random.Random(seed); torch.manual_seed(seed)
    people = C.make_people(n_people, rng)
    n_ask = int(0.7 * n_people)
    facts = [C.fact_sentence(p, a, t) for p in people for a in C.ATTR for t in range(5)]
    cc = city_country_sents()
    pq = [country_q(p) for p in people[:n_ask]]
    adds = [s for s, _, _ in C.add_sentences()]
    pools = [torch.tensor([C.encode(s) for s in x]) for x in (facts, cc, pq, adds)]
    mix = [0.60, 0.10, 0.20, 0.10]
    model = C.TinyLM(len(C.VOCAB))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.05)
    t0 = time.time()
    for step in range(steps):
        parts = []
        for pool, m in zip(pools, mix):
            n = max(1, int(round(bs * m)))
            parts.append(pool[torch.randint(len(pool), (n,))])
        loss = C.lm_loss(model, torch.cat(parts))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == steps - 1:
            heldp = people[n_ask:]
            print(f"seed {seed} step {step} loss {loss.item():.3f} "
                  f"facts {C.accuracy(model, [C.fact_sentence(p, 'city', 0) for p in people[:300]]):.3f} "
                  f"country_trained {cot_acc(model, people[:300]):.3f} "
                  f"country_COMPOSED(held-out people) {cot_acc(model, heldp):.3f} "
                  f"add {C.accuracy(model, adds):.3f} {time.time()-t0:.0f}s", flush=True)
    torch.save({"model": model.state_dict(), "people": people, "n_ask": n_ask}, out)


def setup(seed):
    torch.manual_seed(seed); rng = random.Random(2000 + seed)
    ck = torch.load(f"w2_seed{seed}.pt", weights_only=False)
    model = C.TinyLM(len(C.VOCAB)); model.load_state_dict(ck["model"])
    people = ck["people"]
    newp = C.make_people(200, rng, exclude={tuple(int(x[1:]) for x in p["name"].split()) for p in people})
    return model, people, ck["n_ask"], newp, rng


def evals(model, people, n_ask, newp):
    adds = [s for s, _, _ in C.add_sentences()]
    return {
        "new_city": C.accuracy(model, [C.fact_sentence(p, "city", t) for p in newp for t in range(5)]),
        "new_country": cot_acc(model, newp),
        "new_country_and_city": cot_acc(model, newp, need_city=True),
        "old_city": C.accuracy(model, [C.fact_sentence(p, "city", 0) for p in people[:300]]),
        "old_country_composed": cot_acc(model, people[n_ask:]),
        "add": C.accuracy(model, adds),
    }


def run(seed, method, lr, read=None):
    model, people, n_ask, newp, rng = setup(seed)
    before = evals(model, people, n_ask, newp)
    stream = list(newp); rng.shuffle(stream)
    if method in ("full_ft", "kl"):
        model.train()
        if method == "full_ft":
            opt = torch.optim.AdamW(model.parameters(), lr=lr)
        else:
            for n_, p_ in model.named_parameters(): p_.requires_grad = n_.endswith("ff.values.weight")
            acc = C.Access(model, people, seed=seed); vals = model.mem.values.weight
        for p in stream:
            for _ in range(3):
                ids = torch.tensor([C.encode(C.fact_sentence(p, "city", rng.choice([0, 1, 2, 3]))) for _ in range(16)])
                loss = C.lm_loss(model, ids)
                if method == "full_ft":
                    opt.zero_grad(); loss.backward(); opt.step()
                else:
                    vals.grad = None; loss.backward()
                    mult = C.select_rows("kl", C.batch_counts(model, ids), vals.grad, acc, 64, 0)
                    with torch.no_grad(): vals -= lr * vals.grad * mult.unsqueeze(1)
        model.eval()
        out = [{"method": method, "lr": lr, "read": None, "seed": seed, **evals(model, people, n_ask, newp)}]
    else:   # growth: 4 slots per person (one per city phrasing), then several read switches
        model.eval()
        for p_ in model.parameters(): p_.requires_grad = False
        g = G.Grown(model.mem.q.out_features, model.mem.values.weight.shape[1]); G.attach(model, g)
        for p in stream:
            keys = G.answer_queries(model, [C.fact_sentence(p, "city", t) for t in range(4)])
            st = g.K.shape[0]
            g.K = torch.cat([g.K, keys]); g.V = torch.cat([g.V, torch.zeros(4, g.V.shape[1])])
            g.mode, g.active = "fixed", set(range(st, st + 4))
            for _ in range(3):
                V = g.V.clone().requires_grad_(True); g.V = V
                ids = torch.tensor([C.encode(C.fact_sentence(p, "city", rng.choice([0, 1, 2, 3]))) for _ in range(16)])
                C.lm_loss(model, ids).backward()
                with torch.no_grad(): g.V = (V - lr * V.grad).detach()
        out = []
        for (mode, param) in read:
            g.mode, g.param = mode, param
            out.append({"method": "growth", "lr": lr, "read": [mode, param], "seed": seed,
                        **evals(model, people, n_ask, newp)})
    for r in out:
        r["before"] = before
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pretrain":
        for s in [int(x) for x in sys.argv[2].split(",")]:
            pretrain(s, f"w2_seed{s}.pt")
    else:
        seeds = [int(x) for x in sys.argv[2].split(",")]
        READ = [("nearest", (t, 1)) for t in [0.5, 0.6, 0.7, 0.8, 0.9]] + [("lasso", a) for a in [0.02, 0.05, 0.1, 0.2, 0.3]]
        with open(sys.argv[3], "a") as f:
            for s in seeds:
                rows = []
                for lr in [1e-4, 3e-4]: rows += run(s, "full_ft", lr)
                for lr in [3, 10, 30]: rows += run(s, "kl", lr)
                for lr in [30, 100]: rows += run(s, "growth", lr, READ)
                for r in rows: f.write(json.dumps(r) + "\n")
                f.flush(); print("seed", s, "done", flush=True)
