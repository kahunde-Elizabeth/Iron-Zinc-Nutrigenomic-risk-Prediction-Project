"""Fair model comparison: every algorithm is tuned with the same effort, inside nested cross-validation.

Outer folds give the honest test score. Tuning (GridSearchCV, AUC) sees ONLY the outer training fold, so the
test fold never influences hyper-parameters. 'default' runs use the settings of izn.models.zoo.make_models.
The stacking ensemble uses the four default models as base learners and a logistic-regression meta-learner.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from izn.evaluation.cv import scores
from izn.models.zoo import make_models

ALGORITHMS = ["logreg", "random_forest", "gradient_boosting", "svm_rbf"]
# GridSearchCV addresses pipeline steps by name; make_pipeline names them in lower case
PARAM_PREFIX = {"logreg": "logisticregression__", "svm_rbf": "svc__", "random_forest": "", "gradient_boosting": ""}


def make_stack(seed: int = 42):
    return StackingClassifier(
        estimators=[("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                    ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=1)),
                    ("gb", GradientBoostingClassifier(random_state=seed)),
                    ("svm", make_pipeline(StandardScaler(), SVC(random_state=seed)))],
        final_estimator=LogisticRegression(max_iter=2000), cv=3, n_jobs=1)


def make_specs(grids: dict, seed: int = 42, which: str = "all") -> dict[str, tuple]:
    """name -> (estimator, grid or None). Names: <algo>@default, <algo>@tuned, stacking@default."""
    base = make_models(seed)
    for name, m in base.items():                       # one core per fit; the study parallelises across rules
        if hasattr(m, "n_jobs"):
            m.set_params(n_jobs=1)
    specs: dict[str, tuple] = {}
    for a in ALGORITHMS:
        if which in ("all", "default"):
            specs[f"{a}@default"] = (base[a], None)
        if which in ("all", "tuned"):
            grid = {PARAM_PREFIX[a] + k: v for k, v in grids[a].items()}
            specs[f"{a}@tuned"] = (base[a], grid)
    if which in ("all", "default"):
        specs["stacking@default"] = (make_stack(seed), None)
    return specs


def nested_evaluate(X: np.ndarray, y: np.ndarray, specs: dict[str, tuple], outer: int = 5, inner: int = 3,
                    seed: int = 42, keep_oof: bool = False) -> dict[str, dict]:
    """Per spec: mean/sd fold AUC, mean fold average-precision, optional out-of-fold scores."""
    res = {k: {"aucs": [], "aps": [], "oof": np.full(len(y), np.nan)} for k in specs}
    for tr, te in StratifiedKFold(outer, shuffle=True, random_state=seed).split(X, y):
        for name, (est, grid) in specs.items():
            if grid:
                m = GridSearchCV(clone(est), grid, scoring="roc_auc", n_jobs=1,
                                 cv=StratifiedKFold(inner, shuffle=True, random_state=seed)).fit(X[tr], y[tr])
            else:
                m = clone(est).fit(X[tr], y[tr])
            s = scores(m, X[te])
            res[name]["aucs"].append(roc_auc_score(y[te], s))
            res[name]["aps"].append(average_precision_score(y[te], s))
            res[name]["oof"][te] = s
    out = {}
    for name, r in res.items():
        out[name] = {"auc": float(np.mean(r["aucs"])), "auc_sd": float(np.std(r["aucs"])),
                     "ap": float(np.mean(r["aps"]))}
        if keep_oof:
            out[name]["oof"] = r["oof"]
    return out
