"""E2: loss-based Membership Inference Attack (Teacher-Student paper protocol, as in the project doc).

Positives ('in')  = forget samples; negatives ('out') = test samples from the SAME classes as the forget set.
Balanced by subsampling the larger side, losses clipped to [-400, 400], sklearn LogisticRegression, 5-fold CV.
Attack accuracy ~50% = attacker cannot separate forget from unseen (target: match Retrain).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def loss_mia(forget_loss, forget_label, test_loss, test_label, seed=0, n_folds=5):
    classes = np.unique(forget_label)
    test_loss = test_loss[np.isin(test_label, classes)]
    rng = np.random.RandomState(seed)
    n = min(len(forget_loss), len(test_loss))
    f = forget_loss[rng.choice(len(forget_loss), n, replace=False)]
    t = test_loss[rng.choice(len(test_loss), n, replace=False)]
    x = np.clip(np.nan_to_num(np.concatenate([f, t]), nan=400.0, posinf=400.0, neginf=-400.0), -400, 400)[:, None]
    y = np.concatenate([np.ones(n), np.zeros(n)])
    accs, scores = [], np.zeros(2 * n)
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(x, y):
        clf = LogisticRegression().fit(x[tr], y[tr])
        accs.append(clf.score(x[te], y[te]))
        scores[te] = clf.predict_proba(x[te])[:, 1]
    fpr, tpr, _ = roc_curve(y, scores)
    grid = np.linspace(0, 1, 201)
    return {
        "mia_acc": float(np.mean(accs) * 100),
        "mia_acc_std": float(np.std(accs) * 100),
        "mia_auc": float(roc_auc_score(y, scores) * 100),
        "mia_n_per_side": int(n),
        "roc_fpr_grid": grid.tolist(),
        "roc_tpr": np.interp(grid, fpr, tpr).tolist(),
    }
