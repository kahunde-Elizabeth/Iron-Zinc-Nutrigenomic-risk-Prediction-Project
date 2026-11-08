import pytest

from izn.labels.simulate import auc, draws, make_labels, risk_dosage, score, zscore


def test_risk_dosage_flips_when_risk_is_ref():
    assert risk_dosage(2, "A", "G", "G") == 2
    assert risk_dosage(2, "A", "G", "A") == 0
    assert risk_dosage(1, "A", "G", "A") == 1
    with pytest.raises(ValueError):
        risk_dosage(-1, "A", "G", "G")
    with pytest.raises(ValueError):
        risk_dosage(1, "A", "G", "T")


def test_score_and_zscore():
    feats = {"r1": [0, 1, 2], "r2": [2, 2, 0]}
    meta = {"r1": {"ref": "A", "alt": "G"}, "r2": {"ref": "C", "alt": "T"}}
    assert score(feats, meta, {"r1": "G", "r2": "C"}) == [0.0, 1.0, 4.0]   # r2 risk=ref so 2-dose
    z = zscore([1.0, 2.0, 3.0])
    assert abs(sum(z)) < 1e-9
    with pytest.raises(ValueError):
        zscore([1.0, 1.0])


def test_prevalence_exact_and_h2_controls_signal():
    n = 400
    raw = [float(i % 5) for i in range(n)]
    z = zscore(raw)
    eps, fu = draws(1, "iron", n)
    assert eps == draws(1, "iron", n)[0]                      # reproducible
    lo = make_labels(z, eps, fu, 0.05, 0.2, 0.0)
    hi = make_labels(z, eps, fu, 0.6, 0.2, 0.0)
    assert sum(lo) == sum(hi) == 80
    assert auc(z, hi) > auc(z, lo) > 0.45
    noisy = make_labels(z, eps, fu, 0.6, 0.2, 0.10)
    assert sum(a != b for a, b in zip(hi, noisy)) == 40


def test_auc_basic_and_ties():
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
