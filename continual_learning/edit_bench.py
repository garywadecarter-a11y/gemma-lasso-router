"""
Lifelong model editing benchmark on zsRE: MEMOIR (NeurIPS 2025) vs. lasso-recipe slot memory.

Runs in an ordinary PyTorch + transformers environment (no EasyEdit needed). Both methods
edit the same layer (mlp.down_proj of one decoder layer), see the same features (the
down_proj input), use the same training recipe, and are scored with the same metrics as
MEMOIR's released code (EasyEdit teacher-forced token accuracy):
  reliability    token accuracy of target given the edit prompt
  generalization token accuracy of target given the rephrased prompt
  locality       fraction of tokens on an unrelated prompt left unchanged vs before editing

Methods
  memoir       faithful port of github.com/qym7/MEMOIR (MEMOIRAdapter): zero-init residual
               copy W of down_proj; each edit trains only the columns picked by TopHash
               (top-k |decentered mean prompt feature|, fixed permutation); stored masks;
               inference uses the best-overlapping stored mask if overlap >= tau, else off.
  slot_lasso   Gary's method: a bounded bank of B slots (key, value vector). Each edit gets an
               ADDRESS = normalized random projection (D -> 256) of the decentered mean prompt
               feature. While the bank has room, the edit gets a new slot. When full, a lasso
               expresses the new address as a sparse non-negative combination of existing slot
               keys (the edit's RECIPE) and trains those slots. Inference: nearest stored
               address by cosine; if >= tau apply that edit's recipe (sum w_j V_j added from the
               last prompt token on), else memory off.
  slot_evict   same bank, but when full the oldest slot is overwritten (baseline).
  slot_unbounded  same architecture with no budget (ceiling of the slot design).

Example (Qwen2.5-0.5B-Instruct, 1000 edits):
  python edit_bench.py --model Qwen/Qwen2.5-0.5B-Instruct --data zsre_mem.json --n_edits 1000 \
      --method memoir --out results_memoir.json
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------------ utils ----
class EarlyStopMeter:                              # identical to MEMOIR utils.EarlyStopMeter
    def __init__(self):
        self.pre, self.val = 0, 1e9

    def update(self, v):
        self.pre, self.val = self.val, v

    def stop(self):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02


def get_layer(model, idx):
    return model.model.layers[idx].mlp


def context_templates(model, tok, device):
    """Same as MEMOIR get_context_templates(length_params=[[5,5],[10,5]])."""
    p = tok(["I", "You", "Because", "Yes", "Q: "], padding=True, return_tensors="pt").to(device)
    out = []
    for length in (5, 10):
        g = model.generate(input_ids=p["input_ids"], attention_mask=p["attention_mask"], max_new_tokens=length,
                           num_beams=1, num_return_sequences=1, do_sample=False, pad_token_id=tok.pad_token_id)
        out += tok.batch_decode(g, skip_special_tokens=True)
    return ["{}"] + [o.replace("{", "").replace("}", "") + " {}" for o in out]


# ---------------------------------------------------------------- adapter ----
class Adapter(nn.Module):
    """Wraps down_proj: out = W0 x + memory(x).  The driver sets .mode, .boundary, etc."""

    def __init__(self, lin, args, device):
        super().__init__()
        self.lin, self.args, self.device = lin, args, device
        D_in, D_out = lin.in_features, lin.out_features
        self.D_in, self.D_out = D_in, D_out
        self.mode = "off"            # off | train | infer | collect
        self.boundary = None         # index of last prompt token (same for whole batch; left padding)
        self.collected = None
        self.bg_mean = torch.zeros(D_in, device=device)
        self.pad_count = 0
        g = torch.Generator().manual_seed(args.seed)
        if args.method == "memoir":
            self.W = torch.zeros(D_out, D_in, device=device)            # residual memory (fp32)
            self.perm = torch.randperm(D_in, generator=g).to(device)
            self.k = int(round(args.top_k_ratio * D_in))
            self.masks = torch.zeros(0, D_in, dtype=torch.bool, device=device)
            self.cur_mask = None
        else:
            self.P = (torch.randn(256, D_in, generator=g) / math.sqrt(D_in)).to(device)
            self.K = torch.zeros(0, 256, device=device)                  # slot keys
            self.V = torch.zeros(0, D_out, device=device)                # slot values
            self.age = []
            self.addr = torch.zeros(0, 256, device=device)               # per-edit addresses
            self.recipes = []                                            # per-edit {slot: weight}
            self.cur_recipe = None

    # features aggregated over prompt tokens, decentered (MEMOIR 'mean_decentered')
    def agg(self, x, start):
        return x[0, start:self.boundary + 1, :].float().mean(0) - self.bg_mean

    def address(self, f):
        return F.normalize(self.P @ f, dim=0)

    def forward(self, x):
        out = self.lin(x)
        if self.mode == "off":
            return out
        if self.mode == "collect":
            self.collected = x[0, int(self.pad_count):self.boundary + 1, :].float().mean(0).detach()
            return out
        start = int(self.pad_count) if self.mode == "train" else 0
        f = self.agg(x, start)
        if self.args.method == "memoir":
            idx = self.perm[f.abs().topk(self.k).indices]
            m = torch.zeros(self.D_in, dtype=torch.bool, device=x.device); m[idx] = True
            if self.mode == "train":
                self.cur_mask = m
                use = m
            else:
                if self.masks.shape[0] == 0:
                    return out
                ov = (m.float() @ self.masks.T.float()) / self.k
                best, j = ov.max(0)
                if best.item() < self.tau:
                    return out
                use = self.masks[j]
            xh = x.float() * use.float()
            return out + F.linear(xh, self.W).to(out.dtype)
        # slot memory
        if self.mode == "train":
            rec = self.cur_recipe
        else:
            if self.addr.shape[0] == 0:
                return out
            a = self.address(f)
            sims = self.addr @ a
            best, j = sims.max(0)
            if best.item() < self.tau:
                return out
            rec = self.recipes[j]
        idx = torch.tensor(list(rec.keys()), device=x.device)
        w = torch.tensor(list(rec.values()), device=x.device, dtype=torch.float32)
        delta = (w.unsqueeze(1) * self.V[idx]).sum(0)                     # D_out
        add = torch.zeros_like(out, dtype=torch.float32)
        add[:, self.boundary:, :] = delta
        return out + add.to(out.dtype)


def lasso_recipe(a, K, alpha, m=16, iters=300):
    s = K @ a
    m = min(m, K.shape[0]); v, i = s.topk(m)
    Kc = K[i]; G = Kc @ Kc.T; b = Kc @ a; L = G.diagonal().sum().clamp_min(1e-6)
    w = torch.zeros(m, device=a.device)
    for _ in range(iters):
        w = (w - (G @ w - b) / L - alpha / L).clamp_min(0)
    return {int(i[t]): float(w[t]) for t in range(m) if w[t] > 1e-6}


# ---------------------------------------------------------------- scoring ----
@torch.no_grad()
def token_scores(model, tok, ad, prompt, target, device):
    """EasyEdit test_prediction_acc for one sample: returns (acc, predicted token list)."""
    enc_pt = tok(prompt + " " + target, return_tensors="pt").to(device)
    n_prompt = len(tok(prompt)["input_ids"])
    ad.boundary = n_prompt - 1; ad.pad_count = 0
    logits = model(**enc_pt).logits[0]
    ans = logits.argmax(-1).tolist()[n_prompt - 1:-1]
    lab = enc_pt["input_ids"][0].tolist()[n_prompt:]
    acc = float(np.mean(np.equal(ans, lab))) if lab else float("nan")
    return acc, ans


# ------------------------------------------------------------- checkpoint ----
STATE_KEYS = ["W", "masks", "K", "V", "addr", "bg_mean"]


def save_ckpt(path, ad, n_done, pre_loc, t_elapsed):
    st = {k: getattr(ad, k).detach().cpu() for k in STATE_KEYS if hasattr(ad, k)}
    st.update(n_done=n_done, pre_loc=pre_loc, t_elapsed=t_elapsed,
              age=getattr(ad, "age", None), recipes=getattr(ad, "recipes", None))
    tmp = path + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, path)


def load_ckpt(path, ad, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    for k in STATE_KEYS:
        if k in st:
            setattr(ad, k, st[k].to(device))
    if st.get("age") is not None:
        ad.age = st["age"]
    if st.get("recipes") is not None:
        ad.recipes = st["recipes"]
    return st["n_done"], st["pre_loc"], st["t_elapsed"]


# ------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", required=True, help="zsre_mem.json (fields src, alt, rephrase, loc, loc_ans)")
    ap.add_argument("--n_edits", type=int, default=1000)
    ap.add_argument("--method", required=True, choices=["memoir", "slot_lasso", "slot_evict", "slot_unbounded"])
    ap.add_argument("--layer", type=int, default=None, help="default: round(0.84 * n_layers)")
    ap.add_argument("--budget", type=int, default=500, help="slot methods: max number of slots")
    ap.add_argument("--top_k_ratio", type=float, default=4096 / 14336, help="memoir: fraction of features per edit")
    ap.add_argument("--taus", default=None, help="comma list of inference thresholds to evaluate")
    ap.add_argument("--alpha", type=float, default=0.05, help="slot_lasso: lasso penalty")
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--n_iter", type=int, default=30)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_limit", type=int, default=None, help="score only the first N edits (faster)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_every", type=int, default=50, help="save progress every N edits")
    ap.add_argument("--stop_after", type=int, default=None, help=argparse.SUPPRESS)  # testing only
    args = ap.parse_args()
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    n_layers = len(model.model.layers)
    L = args.layer if args.layer is not None else int(round(0.84 * n_layers))
    mlp = get_layer(model, L)
    ad = Adapter(mlp.down_proj, args, device); mlp.down_proj = ad
    print(f"model {args.model}  layers {n_layers}  editing layer {L}  D_in {ad.D_in}  D_out {ad.D_out}  device {device}")

    data = json.load(open(args.data, encoding="utf-8"))
    edits = data[:args.n_edits]
    bg_items = data[args.n_edits:args.n_edits + 100]
    taus = [float(t) for t in args.taus.split(",")] if args.taus else ([0.3, 0.4, 0.5, 0.6] if args.method == "memoir" else [0.6, 0.7, 0.8, 0.9])

    # ---- background mean feature from 100 irrelevant prompts (MEMOIR mean_decentered) ----
    feats = []
    for it in bg_items:
        enc = tok(it["loc"], return_tensors="pt").to(device)
        ad.boundary = enc["input_ids"].shape[1] - 1; ad.pad_count = 0; ad.mode = "collect"
        with torch.no_grad(): model(**enc)
        feats.append(ad.collected)
    ad.bg_mean = torch.stack(feats).mean(0); ad.mode = "off"

    # ---- pre-edit locality outputs ----
    ev = edits if args.eval_limit is None else edits[:args.eval_limit]
    pre_loc = [token_scores(model, tok, ad, e["loc"], e["loc_ans"], device)[1] for e in ev]

    ckpt = args.out + ".ckpt.pt"
    start_n, t_prev = 0, 0.0
    if os.path.exists(ckpt):
        start_n, pre_loc, t_prev = load_ckpt(ckpt, ad, device)
        print(f"Resuming from checkpoint: {start_n} edits already done", flush=True)

    templates = context_templates(model, tok, device)
    print(f"{len(templates)} context templates, e.g. {templates[1]!r}")

    # ---- sequential editing ----
    t0 = time.time() - t_prev
    for n, e in enumerate(edits):
        if n < start_n:
            continue
        prompt, target = e["src"], e["alt"]
        full = [t.format(prompt + " " + target) for t in templates]
        enc = tok(full, return_tensors="pt", padding=True).to(device)
        S = enc["input_ids"].shape[1]
        lab_len = len(tok(prompt + " " + target)["input_ids"]) - len(tok(prompt)["input_ids"])
        boundary = S - lab_len - 1
        labels = enc["input_ids"].clone(); labels[:, :boundary + 1] = -100
        labels[enc["attention_mask"] == 0] = -100
        ad.boundary = boundary
        ad.pad_count = int((enc["attention_mask"][0] == 0).sum())

        if args.method == "memoir":
            params = None
        else:
            # address of this edit (same features MEMOIR uses)
            ad.mode = "collect"
            with torch.no_grad(): model(**{k: v[:1] for k, v in enc.items()})
            a = ad.address(ad.collected - ad.bg_mean)
            nslots = ad.K.shape[0]
            if args.method == "slot_unbounded" or nslots < args.budget:
                ad.K = torch.cat([ad.K, a[None]]); ad.V = torch.cat([ad.V, torch.zeros(1, ad.D_out, device=device)])
                ad.age.append(n); rec = {nslots: 1.0}
            elif args.method == "slot_evict":
                j = int(np.argmin(ad.age)); ad.K[j] = a; ad.V[j] = 0; ad.age[j] = n; rec = {j: 1.0}
            else:
                w = lasso_recipe(a, ad.K, args.alpha)
                if not w:
                    w = {int((ad.K @ a).argmax()): 1.0}
                mx = max(w.values()); rec = {j: v / mx for j, v in w.items()}
                for j in rec: ad.age[j] = n
            ad.addr = torch.cat([ad.addr, a[None]]); ad.recipes.append(rec); ad.cur_recipe = rec

        ad.mode = "train"
        meter = EarlyStopMeter()
        if args.method == "memoir":
            P = ad.W.clone().requires_grad_(True); ad.W = P
        else:
            P = ad.V.clone().requires_grad_(True); ad.V = P
        opt = torch.optim.SGD([P], lr=args.lr, weight_decay=1e-5)
        for it in range(args.n_iter):
            logits = model(**enc).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=-100)
            if meter.stop():
                break
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([P], max_norm=1.0)
            opt.step(); meter.update(loss.item())
        if args.method == "memoir":
            ad.W = P.detach()
            ad.masks = torch.cat([ad.masks, ad.cur_mask[None]])
        else:
            ad.V = P.detach()
        ad.mode = "off"
        if args.stop_after is not None and n + 1 == args.stop_after:
            save_ckpt(ckpt, ad, n + 1, pre_loc, time.time() - t0)
            print("stopping early (test)"); return
        if (n + 1) % args.ckpt_every == 0:
            el = time.time() - t0
            eta = el / (n + 1) * (len(edits) - n - 1)
            print(f"edit {n+1}/{len(edits)}  last loss {meter.val:.4f}  {el/60:.1f} min elapsed, "
                  f"~{eta/60:.0f} min of editing left (then ~5-10 min scoring)", flush=True)
            save_ckpt(ckpt, ad, n + 1, pre_loc, el)

    # ---- evaluation ----
    results = {"args": vars(args), "layer": L, "edit_seconds": time.time() - t0, "by_tau": {}}
    if args.method == "memoir":
        results["memory_floats"] = ad.W.numel(); results["mask_bits"] = int(ad.masks.numel())
    else:
        results["memory_floats"] = ad.K.numel() + ad.V.numel() + ad.addr.numel()
        results["slots"] = int(ad.K.shape[0])
    for tau in taus:
        ad.tau = tau; ad.mode = "infer"
        rel = [token_scores(model, tok, ad, e["src"], e["alt"], device)[0] for e in ev]
        gen = [token_scores(model, tok, ad, e["rephrase"], e["alt"], device)[0] for e in ev]
        post_loc = [token_scores(model, tok, ad, e["loc"], e["loc_ans"], device)[1] for e in ev]
        loc = [float(np.mean(np.equal(a, b))) for a, b in zip(post_loc, pre_loc)]
        r = {"reliability": float(np.nanmean(rel)), "generalization": float(np.nanmean(gen)), "locality": float(np.mean(loc))}
        r["average"] = float(np.mean([r["reliability"], r["generalization"], r["locality"]]))
        results["by_tau"][str(tau)] = r
        print(f"tau {tau}: rel {r['reliability']:.3f}  gen {r['generalization']:.3f}  loc {r['locality']:.3f}  avg {r['average']:.3f}", flush=True)
    ad.mode = "off"
    json.dump(results, open(args.out, "w"), indent=1)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print("saved", args.out)


if __name__ == "__main__":
    main()
