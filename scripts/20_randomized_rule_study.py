"""Randomized-rule study: many random label RULES instead of one hand-written rule.

For each random rule: simulate labels from real genotypes + synthetic covariates, then compare logistic
regression, random forest, gradient boosting, RBF-SVM (each default AND tuned) and a stacking ensemble with
nested cross-validation. Reports mean AUC, win rate, gap to the oracle ceiling and how results change with the
non-additive share of the rule. EVERYTHING about the labels is simulated; the rule distribution (configs/
randomized_study.yaml) is an assumption.

Outputs (results/): rr_random_rules.tsv, rr_random_results.tsv, rr_random_summary.tsv
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from izn.evaluation.rr_study import CONTENDERS, run_rule, summarise
from izn.evaluation.study_data import load_study_data

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-rules", type=int, default=None, help="override the number of random rules")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (rules run in parallel); -1 = all cores")
    ap.add_argument("--quick", action="store_true", help="small grids and 12 rules: a fast check, not the final run")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg["seed"]
    grids = cfg["quick_grids"] if args.quick else cfg["grids"]
    n_rules = args.n_rules or (12 if args.quick else cfg["random_rules"]["n_rules"])
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    print(f"{len(data['sample'])} samples, {len(data['features'])} SNPs + {len(data['cov_cols'])} synthetic covariates; "
          f"{n_rules} random rules, {'quick' if args.quick else 'full'} grids, seed {seed}", flush=True)

    t0 = time.time()
    rule_rows, res_rows = [], []
    tasks = (delayed(run_rule)(i, data, cfg, grids, seed) for i in range(n_rules))
    try:
        gen = Parallel(n_jobs=args.jobs, return_as="generator")(tasks)
    except TypeError:                                     # older joblib: results arrive all at the end
        gen = Parallel(n_jobs=args.jobs)(tasks)
    for k, (row, rows) in enumerate(gen, 1):
        rule_rows.append(row)
        res_rows += rows
        best = max((r for r in rows if r["feature_set"] == "snps_plus_env" and f"{r['algorithm']}@{r['tuning']}" in CONTENDERS),
                   key=lambda r: r["auc"])
        print(f"[{k:3d}/{n_rules}] {row['rule_id']} nonadd={row['nonadditive_fraction']:.2f} oracle={row['oracle_auc_full']:.3f} "
              f"best={best['algorithm']}@{best['tuning']} AUC={best['auc']:.3f} ({row['seconds']}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = pd.DataFrame(rule_rows).sort_values("rule_id")
    res = pd.DataFrame(res_rows).sort_values(["rule_id", "feature_set", "algorithm", "tuning"])
    rules.to_csv(args.out_dir / "rr_random_rules.tsv", sep="\t", index=False)
    res.to_csv(args.out_dir / "rr_random_results.tsv", sep="\t", index=False)
    summ = summarise(rules, res)
    pd.concat({k: v.reset_index() for k, v in summ.items()}, names=["table"]).reset_index(level=0) \
        .to_csv(args.out_dir / "rr_random_summary.tsv", sep="\t", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}/rr_random_*.tsv\n")
    for k, v in summ.items():
        print(f"== {k}\n{v.to_string()}\n")
    print("Reminder: labels are simulated. These numbers describe how algorithms behave across assumed rules, "
          "not how well they would predict real iron or zinc deficiency.")


if __name__ == "__main__":
    main()
