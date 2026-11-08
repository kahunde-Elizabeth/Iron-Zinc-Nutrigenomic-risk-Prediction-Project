"""Shared pieces of the randomized-rule study and the stress test: one rule -> nested-CV comparison, and the summary tables."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from izn.evaluation.nested import make_specs, nested_evaluate
from izn.evaluation.study_data import design
from izn.labels.random_rules import describe, draw_rule, make_labels, nonadditive_fraction

CONTENDERS = ["logreg@tuned", "random_forest@tuned", "gradient_boosting@tuned", "svm_rbf@tuned", "stacking@default"]


def run_rule(i: int, data: dict, cfg: dict, grids: dict, seed: int, section: str = "random_rules",
             prefix: str = "rule") -> tuple[dict, list[dict]]:
    t0 = time.time()
    rng = np.random.default_rng([seed, i])
    rule = draw_rule(rng, data["features"], data["cov_cols"], cfg[section], data["geno"], data["cov"], f"{prefix}_{i:03d}")
    y, info = make_labels(rule, data["geno"], data["cov"], rng)
    cv = cfg["cv"]
    rows = []

    def add(fs, res):
        for spec, r in res.items():
            algo, tuning = spec.split("@")
            rows.append({"rule_id": rule["name"], "feature_set": fs, "algorithm": algo, "tuning": tuning,
                         "auc": r["auc"], "auc_sd": r["auc_sd"], "ap": r["ap"]})

    specs = make_specs(grids, seed)
    add("snps_plus_env", nested_evaluate(design(data, "snps_plus_env"), y, specs, cv["outer_splits"], cv["inner_splits"], seed))
    lr = {"logreg@tuned": specs["logreg@tuned"]}          # cheap reference runs for the genetic gain
    for fs in ("env_only", "snps_only"):
        add(fs, nested_evaluate(design(data, fs), y, lr, cv["outer_splits"], cv["inner_splits"], seed))
    row = {"rule_id": rule["name"], "h_g": round(rule["h_g"], 3), "h_env": round(rule["h_env"], 3),
           "h_nl": round(rule["h_nl"], 3), "signal_share": round(rule["h_g"] + rule["h_env"] + rule["h_nl"], 3),
           "nonadditive_fraction": round(nonadditive_fraction(rule), 3), "n_snps": len(rule["snps"]),
           "n_env_terms": len(rule["env"]), "n_nonadd_terms": len(rule["terms"]), "prevalence": rule["prevalence"],
           "label_noise": rule["noise"], **info, "description": describe(rule), "seconds": round(time.time() - t0, 1)}
    return row, rows


def summarise(rules: pd.DataFrame, res: pd.DataFrame) -> dict[str, pd.DataFrame]:
    main = res[res.feature_set == "snps_plus_env"].copy()
    main["model"] = main.algorithm + "@" + main.tuning
    cont = main[main.model.isin(CONTENDERS)].pivot(index="rule_id", columns="model", values="auc")[CONTENDERS]
    cont.columns = [c.split("@")[0] for c in cont.columns]
    oracle = rules.set_index("rule_id").oracle_auc_full.reindex(cont.index)
    best = cont.max(axis=1)
    summary = pd.DataFrame({
        "mean_auc": cont.mean(), "sd_across_rules": cont.std(),
        "mean_gap_to_oracle": (-(cont.sub(oracle, axis=0))).mean(),
        "win_rate": (cont.eq(best, axis=0)).mean(),
        "within_0.01_of_best": (cont.sub(best, axis=0) >= -0.01).mean(),
        "mean_rank": cont.rank(axis=1, ascending=False).mean()}).round(4)
    wide = main[main.algorithm.isin(["logreg", "random_forest", "gradient_boosting", "svm_rbf"])].pivot_table(
        index=["rule_id", "algorithm"], columns="tuning", values="auc").reset_index()
    tuning = (wide.assign(gain=wide.tuned - wide.default).groupby("algorithm").gain.agg(["mean", "std"]).round(4)
              .rename(columns={"mean": "tuned_minus_default", "std": "sd"}))
    nf = rules.set_index("rule_id").nonadditive_fraction.reindex(cont.index)
    bins = pd.cut(nf, [-0.001, 0.2, 0.4, 0.6, 1.0], labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8+"])
    by_bin = cont.groupby(bins, observed=True).mean().round(3)
    by_bin.insert(0, "n_rules", bins.value_counts().reindex(by_bin.index))
    nonlin_best = cont[["random_forest", "gradient_boosting", "svm_rbf", "stacking"]].max(axis=1)
    cross = pd.DataFrame({"nonadd_bin": bins, "best_nonlinear_minus_logreg": (nonlin_best - cont.logreg)}) \
        .groupby("nonadd_bin", observed=True).best_nonlinear_minus_logreg.agg(["mean", "std"]).round(4)
    fs = res[(res.algorithm == "logreg") & (res.tuning == "tuned")].pivot(index="rule_id", columns="feature_set", values="auc")
    gain = pd.DataFrame({"snps_plus_env": fs.snps_plus_env, "env_only": fs.env_only, "snps_only": fs.snps_only})
    gain["genetic_gain"] = gain.snps_plus_env - gain.env_only
    return {"model_summary": summary, "tuning_gain": tuning, "auc_by_nonadditive_share": by_bin,
            "nonlinear_advantage_by_share": cross, "genetic_gain_logreg": gain.describe().round(4)}
