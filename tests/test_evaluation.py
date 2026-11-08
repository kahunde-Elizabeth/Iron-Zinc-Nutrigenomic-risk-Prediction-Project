import numpy as np

from izn.evaluation.cv import cv_metrics, feature_sets, lopo, permutation_auc
from izn.models.zoo import make_models


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(n, 4)).astype(float)
    y = (X[:, 0] + rng.normal(0, 0.7, n) > 1.4).astype(int)
    pops = np.array(["A", "B", "C"] * (n // 3))
    return X, y, pops


def test_signal_is_found_and_permutation_is_near_chance():
    X, y, _ = _data()
    m = make_models(1)["logreg"]
    real = cv_metrics(m, X, y, reps=1)["auc_mean"]
    perm = permutation_auc(m, X, y, n_perm=5)
    assert real > 0.75 and 0.4 < perm["perm_auc_mean"] < 0.6


def test_lopo_runs_each_population_and_handles_missing_class():
    X, y, pops = _data()
    res = lopo(make_models(1)["logreg"], X, y, pops)
    assert set(res) == {"A", "B", "C"} and all(v > 0.7 for v in res.values())
    y2 = y.copy()
    y2[pops == "A"] = 0
    assert np.isnan(lopo(make_models(1)["logreg"], X, y2, pops)["A"])


def test_feature_sets_shapes():
    X, _, pops = _data()
    fs = feature_sets(["a", "b", "c", "d"], ["a", "c"], pops, X)
    assert fs["all"].shape[1] == 4 and fs["rule_only"].shape[1] == 2
    assert fs["no_rule"].shape[1] == 2 and fs["pop_only"].shape[1] == 3
