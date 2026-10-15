"""For one good config, measure (a) accuracy by position in the stream, (b) how often
later facts update the same memory slots as earlier facts (overwrite risk)."""
import random, torch, numpy as np, cl_proxy as C
def run(method, lr, t, lam, seed):
    torch.manual_seed(seed); rng = random.Random(1000 + seed)
    ck = torch.load(f"ckpt_seed{seed}.pt", weights_only=False)
    model = C.TinyLM(len(C.VOCAB)); model.load_state_dict(ck["model"]); model.train()
    people = ck["people"]
    newp = C.make_people(100, rng, exclude={tuple(int(x[1:]) for x in p["name"].split()) for p in people})
    stream = [(p, a) for p in newp for a in C.ATTR]; rng.shuffle(stream)
    for n_, p_ in model.named_parameters(): p_.requires_grad = n_.endswith("ff.values.weight")
    acc = C.Access(model, people, seed=seed); vals = model.mem.values.weight
    touched = []; just_after = []
    for k, (p, a) in enumerate(stream):
        rows = set()
        for _ in range(3):
            ids = torch.tensor([C.encode(C.fact_sentence(p, a, rng.randrange(5))) for _ in range(16)])
            loss = C.lm_loss(model, ids); vals.grad = None; loss.backward()
            mult = C.select_rows(method, C.batch_counts(model, ids), vals.grad, acc, t, lam)
            rows |= set(torch.nonzero(mult > 0).flatten().tolist())
            with torch.no_grad(): vals -= lr * vals.grad * mult.unsqueeze(1)
        touched.append(rows)
        just_after.append(C.accuracy(model, [C.fact_sentence(p, a, tt) for tt in range(5)]))
    final = [C.accuracy(model, [C.fact_sentence(p, a, tt) for tt in range(5)]) for (p, a) in stream]
    # overwrite exposure: how many LATER facts touched at least one of this fact's rows
    later_hits = [sum(1 for j in range(i + 1, len(stream)) if touched[i] & touched[j]) for i in range(len(stream))]
    return np.array(just_after), np.array(final), np.array(later_hits), touched
for seed in [0, 1, 2]:
    ja, fi, lh, tch = run("kl", 3, 64, 0, seed)
    q = np.array_split(np.arange(400), 4)
    print(f"seed {seed}: learned right after its own steps {ja.mean():.3f}; at end of stream {fi.mean():.3f}")
    print("   by stream quarter (first->last), end accuracy:", " ".join(f"{fi[i].mean():.2f}" for i in q))
    print("   by stream quarter, right-after accuracy:      ", " ".join(f"{ja[i].mean():.2f}" for i in q))
    lost = (ja > 0.5) & (fi < 0.5); kept = (ja > 0.5) & (fi >= 0.5)
    print(f"   facts learned then lost: {lost.sum()} of {(ja>0.5).sum()} learned ({lost.sum()/(ja>0.5).sum():.0%})")
    print(f"   later facts sharing >=1 updated slot: lost facts avg {lh[lost].mean():.1f}, kept facts avg {lh[kept].mean():.1f}")
    print(f"   distinct slots updated per fact ~{np.mean([len(s) for s in tch]):.0f}; slots touched by >=2 facts: "
          f"{sum(1 for v in np.bincount([r for s in tch for r in s]) if v>=2)} of {len(set(r for s in tch for r in s))}")
