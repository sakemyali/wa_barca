import sys, time, glob, numpy as np, warnings, lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.'); sys.path.insert(0,'evaluation')
from possession import schema, features, matches
from joblib import Parallel, delayed
SIG=4.0
def feat(path, swap):
    m = schema.load(path)
    if swap: m = features.swap_teams(m)
    names, X, rows = features.build(m, ball_sigma=SIG)
    lab = m.label[rows]
    # distance (frames) to the nearest label change, for the breakdown
    ch = np.flatnonzero(lab[1:]!=lab[:-1])+1
    idx = np.arange(len(lab)); pos = np.searchsorted(ch, idx)
    dprev = idx - np.where(pos>0, ch[np.maximum(pos-1,0)], -10**6); dnext = np.where(pos<len(ch), ch[np.minimum(pos,len(ch)-1)], 10**7) - idx
    return m.name, names, X.astype(np.float16), lab, np.minimum(dprev,dnext)
paths = sorted(glob.glob('data_external/*.npz'))
t=time.time()
res = Parallel(n_jobs=4)(delayed(feat)(p, sw) for p in paths for sw in (False, True))
print("features %d in %.0fs"%(len(res), time.time()-t), flush=True)
names = res[0][1]; data = {r[0]: r[2:] for r in res}
def fit_eval(train, test, tag, cols=None):
    cols = np.arange(len(names)) if cols is None else cols
    Xs, ys = [], []
    for k in train:
        for kk in (k, k+'_swap'):
            X, lab, dist = data[kk]; m = lab<2
            Xs.append(X[m][::3][:,cols]); ys.append((lab[m][::3]==0).astype(int))
    Xtr = np.concatenate(Xs).astype(np.float32); ytr = np.concatenate(ys)
    mdl = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, min_child_samples=100, verbose=-1, n_jobs=8)
    mdl.fit(Xtr, ytr)
    for k in test:
        X, lab, dist = data[k]; m = lab<2
        p = mdl.predict_proba(X[m][:,cols].astype(np.float32))[:,1]; y = (lab[m]==0); d = dist[m]
        ok = (p>0.5)==y
        print("  [%s] %-18s acc %.4f | by frames to nearest change: <12: %.3f  12-50: %.3f  50-150: %.3f  >150: %.3f | baseline nearest-player acc %.3f"%(
            tag, k, ok.mean(), ok[d<12].mean(), ok[(d>=12)&(d<50)].mean(), ok[(d>=50)&(d<150)].mean(), ok[d>=150].mean(),
            ((X[m][:, names.index('b:d_diff')].astype(np.float32)<0)==y).mean()), flush=True)
    return mdl
keys = sorted(k for k in data if not k.endswith('_swap'))
dfl = [k for k in keys if k.startswith('dfl')]; met = [k for k in keys if k.startswith('metrica')]
print("train DFL -> test Metrica"); fit_eval(dfl, met, "dfl->met")
print("train Metrica -> test DFL"); fit_eval(met, dfl[:3], "met->dfl")
print("mixed: hold out 1 metrica + 2 dfl"); mdl = fit_eval(dfl[2:]+met[1:], dfl[:2]+met[:1], "mixed")
imp = sorted(zip(mdl.booster_.feature_importance('gain'), names), reverse=True)[:30]
print("top features:", [(n, int(g)) for g,n in imp])
pc = np.array([i for i,n in enumerate(names) if n.startswith('p:')])
print("players-only team model (no ball):"); fit_eval(dfl[2:]+met[1:], dfl[:2]+met[:1], "mixed-noball", cols=pc)
