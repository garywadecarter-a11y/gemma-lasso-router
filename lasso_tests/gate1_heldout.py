"""Held-out evaluation. Metric that rewards ADAPTIVE selection:
exact-set accuracy = keep exactly the needed routes (2 for mixed, 1 for single).
Also report 'needed routes all kept' and avg size. Knob tuned on val half
(maximize exact-set acc), scored on test half; 10 stratified splits."""
import json, random, numpy as np, warnings; warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Lasso, LogisticRegression, OrthogonalMatchingPursuit
import gate1_prompts as GP, gate1 as G
emb=SentenceTransformer(G.MINILM,device="cpu")
tp,ty=G.router_training_data(); Xtr=emb.encode(tp,batch_size=64,normalize_embeddings=True); ty=np.array(ty)
routes=sorted(set(ty))
items=GP.mixed_prompts()+GP.single_prompts()
X=emb.encode([i["prompt"] for i in items],batch_size=64,normalize_embeddings=True)
truth=[set(i["routes"]) for i in items]; kinds=np.array([i["kind"] for i in items])
strata={}
for i,it in enumerate(items): strata.setdefault(it["kind"]+":"+"+".join(it["routes"]),[]).append(i)
lr=LogisticRegression(penalty="l1",solver="saga",C=0.5,max_iter=5000,random_state=42).fit(Xtr,ty); P=lr.predict_proba(X)
K=G.unit(np.stack([Xtr[ty==r].mean(0) for r in routes])); S=X@K.T
mu=Xtr.mean(0); Kc=G.unit(np.stack([(Xtr-mu)[ty==r].mean(0) for r in routes])); Xc=G.unit(X-mu); Sc=Xc@Kc.T
def lasso(Kk,Xx,Ss,a):
    out=[]
    for x,s in zip(Xx,Ss):
        w=Lasso(alpha=a,positive=True,fit_intercept=False,max_iter=20000).fit(Kk.T,x).coef_
        nz=np.flatnonzero(w>1e-6); out.append({routes[j] for j in nz} or {routes[int(np.argmax(s))]})
    return out
def omp(Kk,Xx,Ss,t):
    out=[]
    for x,s in zip(Xx,Ss):
        w=OrthogonalMatchingPursuit(tol=t,fit_intercept=False).fit(Kk.T,x).coef_
        nz=np.flatnonzero(w>1e-6); out.append({routes[j] for j in nz} or {routes[int(np.argmax(s))]})
    return out
cands={
 "gary_router_top1":{0:[{routes[int(np.argmax(p))]} for p in P]},
 "similarity_top1":{0:[{routes[int(np.argmax(s))]} for s in S]},
 "similarity_top2":{0:[{routes[j] for j in np.argsort(-s)[:2]} for s in S]},
 "similarity_top2_centered":{0:[{routes[j] for j in np.argsort(-s)[:2]} for s in Sc]},
 "similarity_margin":{m:[{routes[j] for j in np.flatnonzero(s>=s.max()-m)} for s in S] for m in [0.02,0.04,0.06,0.08,0.1,0.12,0.15,0.2,0.25]},
 "router_prob_threshold":{t:[{routes[j] for j in np.flatnonzero(p>=t)} or {routes[int(np.argmax(p))]} for p in P] for t in [0.05,0.08,0.1,0.12,0.15,0.2,0.3]},
 "omp":{t:omp(K,X,S,t) for t in [0.95,0.9,0.85,0.8,0.75,0.7,0.6]},
 "omp_centered":{t:omp(Kc,Xc,Sc,t) for t in [0.95,0.9,0.85,0.8,0.75,0.7,0.6]},
 "lasso":{a:lasso(K,X,S,a) for a in [1e-3,8e-4,7e-4,6e-4,5e-4,4e-4,3e-4,2e-4]},
 "lasso_centered":{a:lasso(Kc,Xc,Sc,a) for a in [7e-4,5e-4,4e-4,3e-4,2.5e-4,2e-4,1.5e-4,1e-4]},
}
def m(idx,preds):
    mix=[i for i in idx if kinds[i]=="mixed"]; sg=[i for i in idx if kinds[i]=="single"]
    return dict(exact_all=np.mean([truth[i]==preds[i] for i in idx]),
                mixed_both_kept=np.mean([truth[i]<=preds[i] for i in mix]),
                mixed_exact=np.mean([truth[i]==preds[i] for i in mix]),
                single_exact=np.mean([truth[i]==preds[i] for i in sg]),
                size_mixed=np.mean([len(preds[i]) for i in mix]),
                size_single=np.mean([len(preds[i]) for i in sg]))
res={n:[] for n in cands}
for sp in range(10):
    rng=random.Random(100+sp); val=[];test=[]
    for idx in strata.values():
        idx=idx[:]; rng.shuffle(idx); h=len(idx)//2; val+=idx[:h]; test+=idx[h:]
    for n,g in cands.items():
        k=max(g,key=lambda k:m(val,g[k])["exact_all"])
        r=m(test,g[k]); r["knob"]=k; res[n].append(r)
summ={}
print("%-26s %-15s %-15s %-15s %-15s %-9s %-9s %s"%("method","exact(all)","mixed:both kept","mixed:exact","single:exact","size mix","size sgl","knobs"))
for n,rs in res.items():
    f=lambda k:(np.mean([r[k] for r in rs]),np.std([r[k] for r in rs]))
    summ[n]={k:f(k) for k in ["exact_all","mixed_both_kept","mixed_exact","single_exact","size_mixed","size_single"]}
    summ[n]["knobs"]=sorted(set(float(r["knob"]) for r in rs))
    s=summ[n]
    print("%-26s %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.3f±%.3f    %.2f      %.2f      %s"%(n,*s["exact_all"],*s["mixed_both_kept"],*s["mixed_exact"],*s["single_exact"],s["size_mixed"][0],s["size_single"][0],s["knobs"]))
json.dump({n:{k:(list(v) if isinstance(v,tuple) else v) for k,v in s.items()} for n,s in summ.items()},open("gate1_heldout.json","w"),indent=1)
