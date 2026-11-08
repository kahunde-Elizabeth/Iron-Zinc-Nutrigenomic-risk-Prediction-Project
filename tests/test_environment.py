import math

import pytest
import yaml

from izn.labels.environment import COVARIATES, env_score, generate_covariates, interaction, liability
from izn.labels.simulate import auc, draws, threshold_labels, zscore

CFG = yaml.safe_load(open("configs/simulation_env.yaml", encoding="utf-8"))


def test_covariates_reproducible_and_consistent():
    a = generate_covariates(400, CFG, 1)
    assert a == generate_covariates(400, CFG, 1) and set(a) == set(COVARIATES)
    for i in range(400):
        assert not (a["child"][i] and a["pregnant"][i])            # pregnancy only in women
        assert a["blood_loss"][i] >= max(a["heavy_menses"][i], a["hookworm"][i])
        assert (a["age_months"][i] > 0) == bool(a["child"][i])
    assert 0.35 < sum(a["child"]) / 400 < 0.65


def test_signal_ordering_and_share_check():
    n = 600
    cov = generate_covariates(n, CFG, 3)
    zg = zscore([float(i % 5) for i in range(n)])
    ze = zscore(env_score(cov, CFG["env_coefficients"]["iron"]))
    zge = interaction(zg, cov["inflammation"])
    eps, fu = draws(1, "iron", n)
    comps = [(0.2, zg), (0.1, zge), (0.3, ze)]
    y = threshold_labels(liability(comps, eps), fu, 0.2, 0.0)
    assert sum(y) == 120
    signal = liability(comps, [0.0] * n)
    assert auc(signal, y) > auc(zg, y) > 0.5                      # genes + environment beats genes alone
    with pytest.raises(ValueError):
        liability([(0.7, zg), (0.5, ze)], eps)


def test_interaction_is_not_the_main_effect():
    n = 800
    cov = generate_covariates(n, CFG, 5)
    zg = zscore([float(i % 4) for i in range(n)])
    zge = interaction(zg, cov["inflammation"])
    r = sum(a * b for a, b in zip(zg, zge)) / n
    assert abs(r) < 0.15 and math.isclose(sum(zge) / n, 0, abs_tol=1e-9)
