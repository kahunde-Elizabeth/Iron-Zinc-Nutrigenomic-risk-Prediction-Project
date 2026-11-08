import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from izn.evaluation.nested import PARAM_PREFIX, make_specs, nested_evaluate
from izn.labels.random_rules import (coded_dose, draw_rule, labels_from_signal, make_labels, named_rule,
                                     nonadditive_fraction, signal, zs)

CFG = {"signal_share": [0.35, 0.7], "nonadditive_fraction": [0.0, 0.8], "genetic_fraction_of_additive": [0.1, 0.5],
       "n_snps": [3, 5], "snp_coding_probs": {"additive": 0.6, "dominant": 0.2, "recessive": 0.2}, "n_env_terms": [3, 5],
       "env_coef_abs": [0.3, 1.0], "n_nonadd_terms": [1, 3], "nonadd_type_probs": {"gxe": 0.35, "gxg": 0.15, "exe_and": 0.3, "gthresh": 0.2},
       "prevalence": [0.15, 0.35], "label_noise": [0.0, 0.05]}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 300
    geno = {f"s{i}": rng.binomial(2, p, n).astype(float) for i, p in enumerate([0.4, 0.3, 0.2, 0.5, 0.15, 0.35])}
    cov = {"inflammation": rng.binomial(1, 0.3, n).astype(float), "malaria": rng.binomial(1, 0.2, n).astype(float),
           "diet_iron": rng.standard_normal(n), "blood_loss": rng.binomial(1, 0.25, n).astype(float),
           "age_years": rng.uniform(15, 49, n)}
    return geno, cov


def test_zs_and_degenerate():
    assert abs(zs([1, 2, 3, 4]).mean()) < 1e-12 and abs(zs([1, 2, 3, 4]).std() - 1) < 1e-12
    assert zs([5, 5, 5]).tolist() == [0, 0, 0]


def test_coded_dose():
    g = np.array([0, 1, 2.0])
    assert coded_dose(g, False, "additive").tolist() == [0, 1, 2]
    assert coded_dose(g, True, "additive").tolist() == [2, 1, 0]
    assert coded_dose(g, False, "dominant").tolist() == [0, 1, 1]
    assert coded_dose(g, False, "recessive").tolist() == [0, 0, 1]


def test_rules_reproducible_and_valid(data):
    geno, cov = data
    a = draw_rule(np.random.default_rng(1), list(geno), list(cov), CFG, geno, cov, "r")
    b = draw_rule(np.random.default_rng(1), list(geno), list(cov), CFG, geno, cov, "r")
    c = draw_rule(np.random.default_rng(2), list(geno), list(cov), CFG, geno, cov, "r")
    assert a == b and a != c
    for seed in range(15):
        r = draw_rule(np.random.default_rng(seed), list(geno), list(cov), CFG, geno, cov, "r")
        assert 0.35 - 1e-9 <= r["h_g"] + r["h_env"] + r["h_nl"] <= 0.7 + 1e-9
        assert 3 <= len(r["snps"]) <= 5 and 0.15 <= r["prevalence"] <= 0.35
        assert (r["h_nl"] > 1e-3) == bool(r["terms"])
        assert np.isfinite(signal(r, geno, cov)).all()


def test_labels_prevalence_and_oracle(data):
    geno, cov = data
    r = draw_rule(np.random.default_rng(3), list(geno), list(cov), CFG, geno, cov, "r")
    y, info = make_labels(r, geno, cov, np.random.default_rng(4))
    assert abs(y.mean() - r["prevalence"]) < 0.04
    assert info["oracle_auc_full"] > 0.6                      # the true signal predicts its own labels


def test_labels_from_signal_threshold_and_noise():
    sig = np.linspace(-2, 2, 100)
    y = labels_from_signal(sig, 1.0, np.zeros(100), np.random.default_rng(0).random(100), 0.2, 0.0)
    assert y.sum() == 20 and y[-20:].all()                    # top 20% of liability
    y2 = labels_from_signal(sig, 1.0, np.zeros(100), np.random.default_rng(0).random(100), 0.2, 0.1)
    assert (y != y2).sum() == 10                              # exactly 10% flipped


def test_named_rule_flip_and_errors(data):
    geno, cov = data
    meta = {"s0": {"ref": "C", "alt": "T"}, "s1": {"ref": "G", "alt": "A"}}
    st = {"h_g": 0.2, "h_env": 0.3, "h_nl": 0.1, "terms": [{"type": "gscore_x", "cov": "inflammation"}]}
    r = named_rule("n", "iron", {"s0": "T", "s1": "G"}, None, meta, {"diet_iron": -0.6, "malaria": 0.5}, st, 0.2, 0.0)
    assert [s["flip"] for s in r["snps"]] == [False, True]
    assert abs(nonadditive_fraction(r) - 0.1 / 0.6) < 1e-9
    with pytest.raises(ValueError):
        named_rule("n", "iron", {"s0": "A"}, None, meta, {"diet_iron": -0.6}, st, 0.2, 0.0)
    y, info = make_labels(r, geno, cov, np.random.default_rng(0))
    assert info["oracle_auc_full"] > 0.6


def test_structure_terms_resolve_by_trait():
    meta = {"s0": {"ref": "C", "alt": "T"}}
    st = {"h_g": 0.1, "h_env": 0.1, "h_nl": 0.1, "terms": [{"type": "exe_and", "a": {"iron": "diet_iron", "zinc": "diet_zinc"},
                                                              "dir_a": -1, "b": "malaria", "dir_b": 1}]}
    assert named_rule("n", "zinc", {"s0": "T"}, None, meta, {"malaria": 1.0}, st, 0.2, 0)["terms"][0]["a"] == "diet_zinc"
    assert named_rule("n", "iron", {"s0": "T"}, None, meta, {"malaria": 1.0}, st, 0.2, 0)["terms"][0]["a"] == "diet_iron"


def test_specs_cover_every_algorithm_with_a_grid():
    grids = {"logreg": {"C": [1]}, "random_forest": {"min_samples_leaf": [1]}, "gradient_boosting": {"max_depth": [2]},
             "svm_rbf": {"C": [1]}}
    specs = make_specs(grids)
    assert {"logreg@tuned", "random_forest@tuned", "gradient_boosting@tuned", "svm_rbf@tuned", "stacking@default",
            "logreg@default"} <= set(specs)
    assert all(specs[k][1] for k in specs if k.endswith("@tuned"))
    assert set(PARAM_PREFIX) == set(grids)


def test_nested_cv_signal_vs_random_labels(data):
    """Real signal is recovered; completely random labels give AUC near 0.5 (the null control)."""
    geno, cov = data
    X = np.column_stack(list(geno.values()) + list(cov.values()))
    rule = draw_rule(np.random.default_rng(5), list(geno), list(cov), CFG, geno, cov, "r")
    y, _ = make_labels(rule, geno, cov, np.random.default_rng(6))
    spec = {"logreg@tuned": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                             {PARAM_PREFIX["logreg"] + "C": [0.1, 1.0]})}
    real = nested_evaluate(X, y, spec, outer=4, inner=3, keep_oof=True)["logreg@tuned"]
    assert real["auc"] > 0.6 and np.isfinite(real["oof"]).all()
    rand_y = np.random.default_rng(7).integers(0, 2, len(y))
    null = nested_evaluate(X, rand_y, spec, outer=4, inner=3)["logreg@tuned"]
    assert 0.4 < null["auc"] < 0.6


# ------------------------------------------------------------------ stress test: pure interactions
STRESS = {**CFG, "orthogonalise": True, "nonadditive_fraction": [0.8, 1.0], "n_nonadd_terms": [2, 4],
          "nonadd_type_probs": {"gxe": 0.5, "gxg_prod": 0.2, "exe_prod": 0.3}}


@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(11)
    n = 1500
    geno = {f"s{i}": rng.binomial(2, p, n).astype(float) for i, p in enumerate([0.4, 0.3, 0.2, 0.5, 0.15, 0.35])}
    cov = {"inflammation": rng.binomial(1, 0.3, n).astype(float), "malaria": rng.binomial(1, 0.2, n).astype(float),
           "diet_iron": rng.standard_normal(n), "blood_loss": rng.binomial(1, 0.25, n).astype(float),
           "age_years": rng.uniform(15, 49, n)}
    return geno, cov


def test_stress_nonadditive_part_has_no_linear_signal(data):
    from izn.labels.random_rules import components, residualise
    geno, cov = data
    for seed in range(6):
        r = draw_rule(np.random.default_rng(seed), list(geno), list(cov), STRESS, geno, cov, "s")
        assert r["orthogonalise"] and r["terms"] and nonadditive_fraction(r) >= 0.8 - 1e-9
        assert {t["type"] for t in r["terms"]} <= {"gxe", "gxg_prod", "exe_prod"}
        nl = components(r, geno, cov)["nonadd"]
        assert nl.std() > 0.99                                    # standardised, not collapsed
        assert np.abs(residualise(nl, geno, cov) - nl).max() < 1e-6   # nothing left for a linear model to remove


def test_linear_model_fails_but_flexible_model_wins_on_pure_interactions(big):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import roc_auc_score
    geno, cov = big
    rule = draw_rule(np.random.default_rng(1), list(geno), list(cov), {**STRESS, "nonadditive_fraction": [0.95, 1.0],
                                                                       "signal_share": [0.65, 0.7]}, geno, cov, "s")
    y, _ = make_labels(rule, geno, cov, np.random.default_rng(2))
    X = np.column_stack(list(geno.values()) + list(cov.values()))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    lr = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), X, y, cv=cv, method="decision_function")
    rf = cross_val_predict(RandomForestClassifier(300, min_samples_leaf=5, random_state=0, n_jobs=1), X, y, cv=cv, method="predict_proba")[:, 1]
    assert roc_auc_score(y, lr) < 0.6                              # no linear signal to find
    assert roc_auc_score(y, rf) > roc_auc_score(y, lr) + 0.05      # interactions are learnable with enough data
