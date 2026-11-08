import json

import numpy as np
import pandas as pd
import pytest
import yaml

from izn.evaluation import blind as B
from izn.labels.teacher_rules import (any_signal, draw_teacher, linear_r2, make_teacher_labels, teacher_signal)

BT = yaml.safe_load(open("configs/blind_teacher.yaml", encoding="utf-8"))
RR = yaml.safe_load(open("configs/randomized_study.yaml", encoding="utf-8"))
CFG = {**RR, **BT}
TINY = {"logreg": {"C": [1.0]}, "random_forest": {"min_samples_leaf": [3]}, "gradient_boosting": {"max_depth": [2]}, "svm_rbf": {"C": [1.0]}}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 240
    feats = [f"s{i}" for i in range(8)]
    geno = {f: rng.binomial(2, p, n).astype(float) for f, p in zip(feats, [0.4, 0.3, 0.2, 0.5, 0.15, 0.35, 0.45, 0.25])}
    cov = {"inflammation": rng.binomial(1, 0.3, n).astype(float), "malaria": rng.binomial(1, 0.2, n).astype(float),
           "diet_iron": rng.standard_normal(n), "blood_loss": rng.binomial(1, 0.25, n).astype(float),
           "age_years": rng.uniform(15, 49, n), "diet_zinc": rng.standard_normal(n)}
    return {"sample": [f"id{i}" for i in range(n)], "pop": np.array(["YRI"] * n), "features": feats, "geno": geno, "cov": cov,
            "cov_cols": list(cov)}


def _teacher(data, seed, family=None):
    return draw_teacher(np.random.default_rng(seed), data["features"], data["cov_cols"], BT["teacher_rules"], data["geno"], data["cov"], "t", family)


def test_teachers_reproducible_and_valid(data):
    a, b, c = _teacher(data, 1, "mlp"), _teacher(data, 1, "mlp"), _teacher(data, 2, "mlp")
    assert a == b and a != c
    for fam in ("mlp", "trees"):
        for seed in range(8):
            r = _teacher(data, seed, fam)
            sig = teacher_signal(r, data["geno"], data["cov"])
            assert np.isfinite(sig).all() and sig.std() > 0
            assert abs(sig.std() - np.sqrt(r["h_total"])) < 1e-6          # signal variance is the chosen signal share
            assert 0.0 <= linear_r2(r, data["geno"], data["cov"]) <= 1.0
            json.dumps(r)                                                  # sealable: plain JSON types only


def test_labels_and_nonadditivity(data):
    nf = []
    for seed in range(10):
        r = _teacher(data, seed)
        y, info = make_teacher_labels(r, data["geno"], data["cov"], np.random.default_rng(seed))
        assert abs(y.mean() - r["prevalence"]) < 0.05 and info["oracle_auc_full"] > 0.55
        assert abs(info["nonadditive_fraction"] + info["linear_r2"] - 1) < 1e-3
        nf.append(info["nonadditive_fraction"])
    assert max(nf) > 0.15                                                  # teachers are not all straight lines


def test_signal_survives_json_roundtrip(data):
    r = _teacher(data, 3, "trees")
    r2 = json.loads(json.dumps(r))
    assert np.allclose(any_signal(r, data["geno"], data["cov"]), any_signal(r2, data["geno"], data["cov"]))


def test_blind_roundtrip_and_tamper_detection(data, tmp_path):
    cfg = {**CFG, "cv": {"outer_splits": 3, "inner_splits": 2}}
    d1 = B.seal(data, cfg, 3, tmp_path, entropy=12345)
    lab = pd.read_csv(tmp_path / "labels_blind.tsv", sep="\t")
    assert list(lab.columns) == ["sample", "blind_00", "blind_01", "blind_02"]
    assert "rule" not in (tmp_path / "labels_blind.tsv").read_text()        # the labels file leaks nothing about the rules
    with pytest.raises(SystemExit):                                         # cannot reveal before the choice is recorded
        B.reveal(data, tmp_path, 0.3, 42)
    res = B.analyze(data, cfg, TINY, tmp_path, 0.3, 42, jobs=1)
    assert len(res) == 3 and res.chosen.isin(B.CONTENDERS).all()
    assert (tmp_path / "preregistered_choice.sha256").exists()
    tab, s = B.reveal(data, tmp_path, 0.3, 42)
    assert len(tab) == 3 and (tab.oracle_auc_test > 0.5).all() and s["n"] == 3
    assert (tab.best_auc >= tab.chosen_auc - 1e-9).all()                    # best in hindsight is never worse than the choice
    sp = tmp_path / B.SEALED_NAME
    sp.write_text(sp.read_text().replace("blind_00", "blind_0X", 1))        # edit the rules after sealing
    with pytest.raises(SystemExit):
        B.reveal(data, tmp_path, 0.3, 42)


def test_same_entropy_same_labels(data, tmp_path):
    cfg = {**CFG, "cv": {"outer_splits": 3, "inner_splits": 2}}
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    B.seal(data, cfg, 2, tmp_path / "a", entropy=7)
    B.seal(data, cfg, 2, tmp_path / "b", entropy=7)
    assert (tmp_path / "a/labels_blind.tsv").read_text() == (tmp_path / "b/labels_blind.tsv").read_text()
    assert (tmp_path / "a" / B.SEALED_NAME).read_bytes() == (tmp_path / "b" / B.SEALED_NAME).read_bytes()


def test_teacher_rule_row_shape(data):
    cfg = {**CFG, "cv": {"outer_splits": 3, "inner_splits": 2}}
    row, rows = B.run_teacher_rule(0, data, cfg, TINY, 42)
    assert {"rule_id", "nonadditive_fraction", "oracle_auc_full", "family"} <= set(row)
    assert {"snps_plus_env", "env_only", "snps_only"} == {r["feature_set"] for r in rows}
