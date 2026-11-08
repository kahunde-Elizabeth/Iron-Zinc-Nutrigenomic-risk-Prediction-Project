"""Phase 6b: models on genes + synthetic environment labels.

Feature sets: snps_only, env_only, snps_plus_env. 'genetic gain' = AUC(snps_plus_env) - AUC(env_only):
how much the real genotypes add once the synthetic environment is known. Labels and covariates are SIMULATED.
Outputs results/model_results_env.tsv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from izn.evaluation.cv import cv_metrics, lopo, permutation_auc
from izn.labels.environment import COVARIATES
from izn.models.zoo import make_models


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/p1_labels_env.parquet"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/label_manifest_env.tsv"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--out", type=Path, default=Path("results/model_results_env.tsv"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep-hg", action="store_true", help="also run every genetic share (snps_plus_env only)")
    args = ap.parse_args()

    feats = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))["features"]
    d = yaml.safe_load(args.env.read_text(encoding="utf-8"))["reporting_default"]
    ft, ev, lb = (pq.read_table(p).to_pydict() for p in (args.features, args.env_data, args.labels))
    assert ft["sample"] == ev["sample"] == lb["sample"], "rows are not in the same order"
    G = np.array([ft[f] for f in feats], dtype=float).T
    cov_cols = [c for c in COVARIATES if len(set(ev[c])) > 1]
    E = np.array([ev[c] for c in cov_cols], dtype=float).T
    sets = {"snps_only": G, "env_only": E, "snps_plus_env": np.hstack([G, E])}
    pops = np.array(ft["pop"])
    with args.manifest.open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))
    models = make_models(args.seed)

    def is_default(m):
        return (float(m["prevalence_target"]), float(m["h_g"]), float(m["h_ge"]), float(m["label_noise"])) == \
               (d["prevalence"], d["h_g"], d["h_ge"], d["label_noise"])

    def selected(m):
        if is_default(m):
            return True
        return args.sweep_hg and (float(m["prevalence_target"]), float(m["h_ge"]), float(m["label_noise"])) == \
            (d["prevalence"], d["h_ge"], d["label_noise"])

    rows = []
    for m in [m for m in manifest if selected(m)]:
        y = np.array(lb[m["column"]])
        full = is_default(m)
        for fs, X in sets.items():
            if not full and fs != "snps_plus_env":
                continue
            for name, model in models.items():
                t0 = time.time()
                r = {"scenario": m["scenario"], "trait": m["trait"], "h_g": m["h_g"], "h_ge": m["h_ge"],
                     "feature_set": fs, "algorithm": name, "oracle_auc_genetic": m["oracle_auc_genetic"],
                     "oracle_auc_full": m["oracle_auc_full"], **cv_metrics(model, X, y, args.reps, args.seed)}
                if full:
                    r["lopo_auc_mean"] = float(np.nanmean(list(lopo(model, X, y, pops).values())))
                    if fs == "snps_plus_env" and args.perms:
                        r.update(permutation_auc(model, X, y, args.perms, args.seed))
                r["seconds"] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"{m['scenario'][:12]:12s} {m['trait']:5s} hg={m['h_g']:<4} {fs:14s} {name:18s} "
                      f"AUC={r['auc_mean']:.3f} (ceiling full {m['oracle_auc_full']})", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, delimiter="\t", restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"done: {len(rows)} rows -> {args.out}")
    print("\ngenetic gain at the reporting default (snps_plus_env minus env_only):")
    for sc_, tr in sorted({(r["scenario"], r["trait"]) for r in rows}):
        for a in models:
            g = {r["feature_set"]: r["auc_mean"] for r in rows
                 if (r["scenario"], r["trait"], r["algorithm"]) == (sc_, tr, a) and r["lopo_auc_mean" if "lopo_auc_mean" in r else "algorithm"] != ""
                 and float(r["h_g"]) == d["h_g"] and float(r["h_ge"]) == d["h_ge"]}
            if "snps_plus_env" in g and "env_only" in g:
                print(f"  {sc_[:12]:12s} {tr:5s} {a:18s} {g['snps_plus_env'] - g['env_only']:+.3f}")


if __name__ == "__main__":
    main()
