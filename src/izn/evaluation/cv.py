"""Evaluation helpers: repeated CV, leave-one-population-out, permutation control, feature sets."""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold


def scores(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def cv_metrics(model, X, y, reps: int = 3, seed: int = 42, splits: int = 5) -> dict:
    """Per-fold AUC and average precision. Folds overlap between repeats, so sd is descriptive only."""
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=splits, n_repeats=reps, random_state=seed).split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        s = scores(m, X[te])
        auc.append(roc_auc_score(y[te], s))
        ap.append(average_precision_score(y[te], s))
    return {"auc_mean": float(np.mean(auc)), "auc_sd": float(np.std(auc)), "ap_mean": float(np.mean(ap))}


def lopo(model, X, y, pops: np.ndarray) -> dict[str, float]:
    """Train on four populations, test on the fifth. NaN when the test population lacks a class."""
    out = {}
    for p in sorted(set(pops)):
        te = pops == p
        if len(set(y[te])) < 2 or len(set(y[~te])) < 2:
            out[p] = float("nan")
            continue
        m = clone(model).fit(X[~te], y[~te])
        out[p] = float(roc_auc_score(y[te], scores(m, X[te])))
    return out


def permutation_auc(model, X, y, n_perm: int = 20, seed: int = 42) -> dict:
    """CV AUC with shuffled labels: should sit near 0.5. p95 is the line a real result must clear."""
    rng = np.random.default_rng(seed)
    vals = []
    for i in range(n_perm):
        yp = rng.permutation(y)
        vals.append(cv_metrics(model, X, yp, reps=1, seed=seed + i)["auc_mean"])
    return {"perm_auc_mean": float(np.mean(vals)), "perm_auc_p95": float(np.percentile(vals, 95))}


def feature_sets(features: list[str], rule_snps: list[str], pops: np.ndarray, geno: np.ndarray) -> dict[str, np.ndarray]:
    """all SNPs, only the SNPs in the label rule, SNPs without the rule SNPs, and population only."""
    idx = {f: i for i, f in enumerate(features)}
    rule = [idx[r] for r in rule_snps]
    rest = [i for i in range(len(features)) if i not in rule]
    onehot = np.stack([(pops == p).astype(float) for p in sorted(set(pops))], axis=1)
    return {"all": geno, "rule_only": geno[:, rule], "no_rule": geno[:, rest], "pop_only": onehot}
