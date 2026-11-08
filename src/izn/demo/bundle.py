"""Demo bundle: one transparent model (logistic regression), its held-out scores, risk tiers and calibration.

Used by the demo web app (app/demo_app.py). Labels come from a SIMULATED rule, so every number is a statement about the
pipeline, not about real iron or zinc deficiency. The model is logistic regression on purpose: its score can be
explained feature by feature, which a stacked ensemble cannot.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from izn.evaluation.study_data import design, design_columns
from izn.labels.random_rules import describe, draw_rule, make_labels

COV_LABEL = {"child": "Child under five", "age_months": "Age (months)", "age_years": "Age (years)", "young_child": "Child under two",
             "pregnant": "Pregnant", "heavy_menses": "Heavy periods", "hookworm": "Hookworm infection", "blood_loss": "Ongoing blood loss",
             "malaria": "Recent malaria", "inflammation": "Current inflammation or infection", "diarrhoea": "Recent diarrhoea",
             "diet_iron": "Dietary iron intake (z-score)", "diet_zinc": "Dietary zinc intake (z-score)"}
TIER_NAMES = ["Low", "Medium", "High"]
# Biologically plausible direction of each covariate's effect on deficiency risk (+1 raises risk, -1 lowers it). The random-rule
# study draws these signs at random on purpose; a DEMO that a person will read must not tell them that eating more iron raises risk.
PLAUSIBLE_DIRECTION = {"diet_iron": -1, "diet_zinc": -1, "pregnant": 1, "heavy_menses": 1, "hookworm": 1, "blood_loss": 1, "malaria": 1,
                       "inflammation": 1, "diarrhoea": 1, "child": 1, "young_child": 1, "age_months": -1}


def tier_of(p: float, cut_low: float, cut_high: float) -> str:
    return TIER_NAMES[0] if p < cut_low else TIER_NAMES[1] if p < cut_high else TIER_NAMES[2]


def _panel_info(panel_path: Path | None) -> dict:
    if panel_path is None or not Path(panel_path).exists():
        return {}
    return {r["rsid"]: r for r in yaml.safe_load(Path(panel_path).read_text(encoding="utf-8"))["panel"]}


def build_bundle(data: dict, cfg: dict, seed: int, panel_path: Path | None = None, n_candidates: int = 30) -> tuple[dict, pd.DataFrame]:
    geno, cov = data["geno"], data["cov"]
    cands = []
    for i in range(n_candidates):                       # the demo rule is the one with the MEDIAN oracle AUC of the first 30 random rules
        rng = np.random.default_rng([seed, i])
        rule = draw_rule(rng, data["features"], data["cov_cols"], cfg["random_rules"], geno, cov, f"demo_{i:03d}")
        rule["env"] = {c: round(abs(v) * PLAUSIBLE_DIRECTION.get(c, 1 if v >= 0 else -1), 3) for c, v in rule["env"].items()}
        for c in ("diet_iron", "diet_zinc"):              # both diet inputs are on the demo's sliders, so the demo rule must use them
            if c in cov and c not in rule["env"]:
                rule["env"][c] = -0.5
        y, info = make_labels(rule, geno, cov, rng)
        cands.append((info["oracle_auc_full"], rule, y, info))
    med = float(np.median([c[0] for c in cands]))
    oracle, rule, y, info = min(cands, key=lambda c: abs(c[0] - med))

    X, cols = design(data, "snps_plus_env"), design_columns(data)
    est = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                       {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc",
                       cv=StratifiedKFold(3, shuffle=True, random_state=seed))
    oof = cross_val_predict(est, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    est.fit(X, y)
    sc, lr = est.best_estimator_.named_steps["standardscaler"], est.best_estimator_.named_steps["logisticregression"]

    prev = float(y.mean())
    cut_low, cut_high = prev, 2 * prev                  # fixed BEFORE looking at results: below, up to double, over double the cohort rate
    tiers = [tier_of(p, cut_low, cut_high) for p in oof]
    people = pd.DataFrame({"sample": data["sample"], "pop": data["pop"], "label": y, "heldout_prob": np.round(oof, 4), "tier": tiers})
    for j, c in enumerate(cols):
        people[c] = X[:, j]
    table = [{"tier": t, "n": int((people.tier == t).sum()), "share": float((people.tier == t).mean()),
              "observed_rate": float(people.loc[people.tier == t, "label"].mean()) if (people.tier == t).any() else None} for t in TIER_NAMES]
    q = pd.qcut(oof, 5, duplicates="drop")
    calib = [{"predicted": float(oof[q == b].mean()), "observed": float(y[q == b].mean()), "n": int((q == b).sum())} for b in q.categories]

    panel = _panel_info(panel_path)
    finfo = {}
    for c in cols:
        if c in geno:
            p = panel.get(c, {})
            finfo[c] = {"kind": "snp", "label": f"{c} ({p.get('gene', 'gene n/a')})", "gene": p.get("gene"), "trait": p.get("trait")}
        else:
            v = np.asarray(cov[c], float)
            finfo[c] = {"kind": "binary" if set(np.unique(v)) <= {0.0, 1.0} else "continuous", "label": COV_LABEL.get(c, c),
                        "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
    bundle = {"generated": str(date.today()), "rule_name": rule["name"], "rule_description": describe(rule), "rule_oracle_auc": oracle,
              "model": {"features": cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist(),
                        "intercept": float(lr.intercept_[0]), "C": float(est.best_params_["logisticregression__C"])},
              "metrics": {"n": int(len(y)), "prevalence": prev, "auc_heldout": float(roc_auc_score(y, oof)), "ap_heldout": float(average_precision_score(y, oof))},
              "tiers": {"cut_low": cut_low, "cut_high": cut_high, "table": table}, "calibration": calib, "features": finfo}
    return bundle, people


def score_person(bundle: dict, values: dict) -> dict:
    """Probability, tier and per-feature contributions (log-odds relative to the average person) for one person."""
    m = bundle["model"]
    x = np.array([float(values[c]) for c in m["features"]])
    z = (x - np.array(m["mean"])) / np.array(m["scale"])
    contrib = z * np.array(m["coef"])
    logit = m["intercept"] + float(contrib.sum())
    p = float(1.0 / (1.0 + np.exp(-logit)))
    t = bundle["tiers"]
    order = np.argsort(-np.abs(contrib))
    return {"prob": p, "tier": tier_of(p, t["cut_low"], t["cut_high"]),
            "contributions": [(m["features"][i], float(contrib[i])) for i in order]}


def save_bundle(bundle: dict, people: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=1), encoding="utf-8")
    people.to_csv(out_dir / "people.csv", index=False)


def load_bundle(out_dir: Path) -> tuple[dict, pd.DataFrame]:
    return json.loads((Path(out_dir) / "bundle.json").read_text(encoding="utf-8")), pd.read_csv(Path(out_dir) / "people.csv")
