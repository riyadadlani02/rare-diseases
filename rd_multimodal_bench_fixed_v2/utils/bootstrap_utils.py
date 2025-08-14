
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def bootstrap_ci(y_true, y_prob, y_pred, average="macro", n_iter=1000, ci=95, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, aucs = [], [], []
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    num_classes = y_prob.shape[1] if y_prob.ndim == 2 else 2
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        try:
            auc = roc_auc_score(pd.get_dummies(yt), ypr, multi_class="ovr" if num_classes > 2 else "raise")
        except Exception:
            auc = np.nan
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average=average))
        aucs.append(auc)
    lo = (100 - ci) / 2.0
    hi = 100 - lo
    return (
        (float(np.nanpercentile(accs, lo)), float(np.nanpercentile(accs, hi))),
        (float(np.nanpercentile(f1s,  lo)), float(np.nanpercentile(f1s,  hi))),
        (float(np.nanpercentile(aucs, lo)), float(np.nanpercentile(aucs, hi)))
    )
