import json, numpy as np, warnings; warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Lasso, LogisticRegression, OrthogonalMatchingPursuit
import gate1_prompts as GP, gate1 as G
emb=SentenceTransformer(G.MINILM,device="cpu")
tp,ty=G.router_training_data(); Xtr=emb.encode(tp,batch_size=64,normalize_embeddings=True); ty=np.array(ty)
routes=sorted(set(ty))
items=GP.mixed_prompts()+GP.single_prompts()
X=emb.encode([i["prompt"] for i in items],batch_size=64,normalize_embeddings=True)
truth=[set(i["routes"]) for i in items]; kinds=np.array([i["kind"] for i in items])
lr=LogisticRegression(penalty="l1",solver="saga",C=0.5,max_iter=5000,random_state=42).fit(Xtr,ty)
P=lr.predict_proba(X)
mu=Xtr.mean(0)
def keys_of(center):
    Z=Xtr-mu if center else Xtr
    return G.unit(np.stack([Z[ty==r].mean(0) for r in routes]))
def summarize(preds):
    m=kinds=="mixed"; s=~m
    both=np.mean([truth[i]<=preds[i] for i in np.flatnonzero(m)])
    sx=np.mean([truth[i]==preds[i] for i in np.flatnonzero(s)])
    mx=np.mean([truth[i]==preds[i] for i in np.flatnonzero(m)])
    size=np.mean([len(p) for p in preds])
    return both,mx,sx,size
rows=[]
for center in [False,True]:
    K=keys_of(center); Xq=G.unit(X-mu) if center else X; S=Xq@K.T
    for a in [3e-3,2e-3,1e-3,7e-4,5e-4,3e-4,2e-4,1e-4,5e-5,2e-5]:
        preds=[]
        for x,s in zip(Xq,S):
            w=Lasso(alpha=a,positive=True,fit_intercept=False,max_iter=20000).fit(K.T,x).coef_
            nz=np.flatnonzero(w>1e-6); preds.append({routes[j] for j in nz} or {routes[int(np.argmax(s))]})
        rows.append(("lasso"+("_centered" if center else ""),a)+summarize(preds))
    for t in [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]:
        preds=[]
        for x,s in zip(Xq,S):
            w=OrthogonalMatchingPursuit(tol=t,fit_intercept=False).fit(K.T,x).coef_
            nz=np.flatnonzero(w>1e-6); preds.append({routes[j] for j in nz} or {routes[int(np.argmax(s))]})
        rows.append(("omp"+("_centered" if center else ""),t)+summarize(preds))
    for m in [0.02,0.04,0.06,0.08,0.1,0.15,0.2]:
        preds=[{routes[j] for j in np.flatnonzero(s>=s.max()-m)} for s in S]
        rows.append(("cos_margin"+("_centered" if center else ""),m)+summarize(preds))
    for k in [1,2,3]:
        preds=[{routes[j] for j in np.argsort(-s)[:k]} for s in S]
        rows.append(("cos_top"+str(k)+("_centered" if center else ""),k)+summarize(preds))
for t in [0.05,0.1,0.15,0.2,0.3,0.4]:
    preds=[{routes[j] for j in np.flatnonzero(p>=t)} or {routes[int(np.argmax(p))]} for p in P]
    rows.append(("lr_prob_thresh",t)+summarize(preds))
for k in [1,2]:
    preds=[{routes[j] for j in np.argsort(-p)[:k]} for p in P]
    rows.append(("lr_top"+str(k),k)+summarize(preds))
print("%-22s %8s %7s %8s %8s %6s"%("method","knob","mixBoth","mixExact","sglExact","size"))
for r in rows: print("%-22s %8g %7.3f %8.3f %8.3f %6.2f"%r)
json.dump(rows,open("gate1_curves.json","w"))
