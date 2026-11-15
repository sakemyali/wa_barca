import sys, time, numpy as np, warnings, lightgbm as lgb
warnings.filterwarnings('ignore')
sys.path.insert(0,'src'); sys.path.insert(0,'.'); sys.path.insert(0,'evaluation')
from possession import schema, features, matches
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
def feat(path):
    m = schema.load(path)
    names, X, rows = features.build(m, views=("players",))
    return m.name, names, X, m.label[rows], m.period[rows]
import glob
paths = sorted(glob.glob('data_external/*.npz'))
t=time.time()
res = Parallel(n_jobs=4)(delayed(feat)(p) for p in paths)
print("features for %d matches in %.0fs"%(len(res), time.time()-t), flush=True)
names = res[0][1]
data = {r[0]: (r[2], r[3], r[4]) for r in res}
def fit_eval(train, test, tag):
    Xtr = np.concatenate([data[k][0][::4] for k in train]); ytr = np.concatenate([(data[k][1][::4]==2).astype(int) for k in train])
    mdl = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, min_child_samples=200, verbose=-1, n_jobs=8)
    mdl.fit(Xtr, ytr)
    for k in test:
        X, y, _ = data[k]; p = mdl.predict_proba(X)[:,1]; yd = (y==2).astype(int)
        acc = ((p>0.5)==yd).mean()
        print("  [%s] test %-18s AUC %.4f acc %.4f  dead recall %.3f alive recall %.3f"%(tag, k, roc_auc_score(yd,p), acc, (p[yd==1]>0.5).mean(), (p[yd==0]<=0.5).mean()), flush=True)
    return mdl
keys = sorted(data)
dfl = [k for k in keys if k.startswith('dfl')]; met = [k for k in keys if k.startswith('metrica')]
print("train DFL -> test Metrica"); fit_eval(dfl, met, "dfl->met")
print("train Metrica -> test DFL"); fit_eval(met, dfl[:3], "met->dfl")
print("mixed: hold out 1 metrica + 2 dfl")
mdl = fit_eval(dfl[2:]+met[1:], dfl[:2]+met[:1], "mixed")
imp = sorted(zip(mdl.booster_.feature_importance('gain'), names), reverse=True)[:25]
print("top features by gain:", [(n, int(g)) for g,n in imp])
# apply the mixed model to the provided match: aggregate agreement with the ball-visibility channel only
a = matches.provided(); n2, Xa, rows = features.build(a, views=("players",))
p = mdl.predict_proba(Xa)[:,1]; miss = ~np.isfinite(a.ball_xy[rows,0])
print("ATD: mean P(dead) %.3f | P(dead) when ball missing: %.3f | when visible: %.3f | frac P>0.5: missing %.3f visible %.3f"%(p.mean(), p[miss].mean(), p[~miss].mean(), (p[miss]>0.5).mean(), (p[~miss]>0.5).mean()))
pass
