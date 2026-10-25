"""Permutation importance (AUC drop) of the gradient-boosting meta-model, fit on first 70% of pooled trades by time, scored on the last 30%."""
import os, sys
import numpy as np, pandas as pd
from sklearn.inspection import permutation_importance
ROOT = os.path.abspath(__file__)[: os.path.abspath(__file__).find("MAD_BTC") + len("MAD_BTC")]
sys.path.extend([ROOT, ROOT + "/src"])
from src.meta_labeling.run_meta import load, events, make_model, FEATS
for tf, lb in (("4h", 5), ("4h", 10)):
    ev = pd.concat([events(load(a, tf), lb).assign(a=a) for a in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"]]).sort_values("t_in")
    k = int(len(ev) * 0.7); tr, te = ev.iloc[:k], ev.iloc[k:]
    m = make_model("gboost").fit(tr[FEATS], tr["y"])
    r = permutation_importance(m, te[FEATS], te["y"], scoring="roc_auc", n_repeats=30, random_state=0)
    s = pd.Series(r.importances_mean, FEATS).sort_values(ascending=False)
    print(f"{tf} lb{lb}  AUC test={__import__('sklearn.metrics').metrics.roc_auc_score(te['y'], m.predict_proba(te[FEATS])[:,1]):.3f}")
    print(s.round(4).head(6).to_string())
