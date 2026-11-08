import numpy as np
import pytest
import yaml

from izn.demo.bundle import TIER_NAMES, build_bundle, load_bundle, save_bundle, score_person, tier_of

RR = yaml.safe_load(open("configs/randomized_study.yaml", encoding="utf-8"))


@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(0)
    n = 300
    feats = [f"s{i}" for i in range(8)]
    geno = {f: rng.binomial(2, p, n).astype(float) for f, p in zip(feats, [0.4, 0.3, 0.2, 0.5, 0.15, 0.35, 0.45, 0.25])}
    cov = {"inflammation": rng.binomial(1, 0.3, n).astype(float), "malaria": rng.binomial(1, 0.2, n).astype(float),
           "diet_iron": rng.standard_normal(n), "blood_loss": rng.binomial(1, 0.25, n).astype(float), "age_years": rng.uniform(15, 49, n)}
    data = {"sample": [f"id{i}" for i in range(n)], "pop": np.array(["YRI"] * n), "features": feats, "geno": geno, "cov": cov, "cov_cols": list(cov)}
    return data, *build_bundle(data, RR, 42, None, n_candidates=6)


def test_scoring_matches_sklearn_probabilities(built):
    data, bundle, people = built
    m = bundle["model"]
    row = people.iloc[0]
    res = score_person(bundle, {c: row[c] for c in m["features"]})
    z = (np.array([row[c] for c in m["features"]]) - np.array(m["mean"])) / np.array(m["scale"])
    assert abs(res["prob"] - 1 / (1 + np.exp(-(m["intercept"] + z @ np.array(m["coef"]))))) < 1e-12
    assert abs(sum(v for _, v in res["contributions"]) + m["intercept"] - np.log(res["prob"] / (1 - res["prob"]))) < 1e-9


def test_tiers_rise_and_cutpoints_are_prespecified(built):
    _, bundle, people = built
    t = bundle["tiers"]
    assert t["cut_low"] == pytest.approx(bundle["metrics"]["prevalence"]) and t["cut_high"] == pytest.approx(2 * t["cut_low"])
    assert [tier_of(p, t["cut_low"], t["cut_high"]) for p in (0.0, t["cut_low"] + 1e-6, 0.99)] == TIER_NAMES
    rates = [r["observed_rate"] for r in t["table"] if r["observed_rate"] is not None]
    assert rates == sorted(rates)                                           # observed rate rises from low to high
    assert bundle["metrics"]["auc_heldout"] > 0.6 and sum(r["n"] for r in t["table"]) == len(people)


def test_roundtrip(built, tmp_path):
    _, bundle, people = built
    save_bundle(bundle, people, tmp_path)
    b2, p2 = load_bundle(tmp_path)
    assert b2["model"]["coef"] == bundle["model"]["coef"] and len(p2) == len(people)


def test_demo_rule_directions_are_biologically_plausible(built):
    _, bundle, _ = built
    from izn.demo.bundle import PLAUSIBLE_DIRECTION
    sign = {c: np.sign(w) for c, w in zip(bundle["model"]["features"], bundle["model"]["coef"])}
    # the fitted model should agree with the rule's direction wherever the rule uses that covariate and the effect is not tiny
    checked = [c for c in PLAUSIBLE_DIRECTION if c in sign]
    agree = sum(sign[c] == PLAUSIBLE_DIRECTION[c] for c in checked)
    assert agree >= 0.7 * len(checked)


def test_demo_diet_sliders_lower_risk(built):
    _, bundle, _ = built
    coef = dict(zip(bundle["model"]["features"], bundle["model"]["coef"]))
    for c in ("diet_iron",):
        if c in coef:
            assert coef[c] < 0                                              # more dietary iron -> lower estimated risk
