"""Stress test: rules that are 80-100% non-additive, built only from pure interactions, with the non-additive part
orthogonalised so a linear model gets NO signal from it (configs/randomized_study.yaml -> stress_test).

If logistic regression still ties the flexible models here, additiveness is not what drives the main result. If a
flexible model clearly wins, that is the boundary to report. Labels and rules are SIMULATED.

Outputs (results/): rr_stress_rules.tsv, rr_stress_results.tsv, rr_stress_summary.tsv, figures/fig17_stress_test.png
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
from izn.reporting import figures as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-rules", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="small grids and 8 rules: a fast check")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = cfg["seed"] + 1000                      # different random rules from the main study
    grids = cfg["quick_grids"] if args.quick else cfg["grids"]
    n_rules = args.n_rules or (8 if args.quick else cfg["stress_test"]["n_rules"])
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    print(f"stress test: {n_rules} rules, 80-100% non-additive pure interactions, {'quick' if args.quick else 'full'} grids", flush=True)

    t0 = time.time()
    tasks = (delayed(run_rule)(i, data, cfg, grids, seed, "stress_test", "stress") for i in range(n_rules))
    try:
        gen = Parallel(n_jobs=args.jobs, return_as="generator")(tasks)
    except TypeError:
        gen = Parallel(n_jobs=args.jobs)(tasks)
    rule_rows, res_rows = [], []
    for k, (row, rows) in enumerate(gen, 1):
        rule_rows.append(row)
        res_rows += rows
        best = max((r for r in rows if r["feature_set"] == "snps_plus_env" and f"{r['algorithm']}@{r['tuning']}" in CONTENDERS), key=lambda r: r["auc"])
        print(f"[{k:3d}/{n_rules}] {row['rule_id']} nonadd={row['nonadditive_fraction']:.2f} oracle={row['oracle_auc_full']:.3f} "
              f"best={best['algorithm']}@{best['tuning']} AUC={best['auc']:.3f} ({row['seconds']}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = pd.DataFrame(rule_rows).sort_values("rule_id")
    res = pd.DataFrame(res_rows).sort_values(["rule_id", "feature_set", "algorithm", "tuning"])
    rules.to_csv(args.out_dir / "rr_stress_rules.tsv", sep="\t", index=False)
    res.to_csv(args.out_dir / "rr_stress_results.tsv", sep="\t", index=False)
    summ = summarise(rules, res)
    pd.concat({k: v.reset_index() for k, v in summ.items()}, names=["table"]).reset_index(level=0) \
        .to_csv(args.out_dir / "rr_stress_summary.tsv", sep="\t", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min\n")
    for k in ("model_summary", "tuning_gain", "genetic_gain_logreg"):
        print(f"== {k} (STRESS TEST)\n{summ[k].to_string()}\n")

    mr, mres = args.out_dir / "rr_random_rules.tsv", args.out_dir / "rr_random_results.tsv"
    main_rules = pd.read_csv(mr, sep="\t") if mr.exists() and mres.exists() else None
    main_res = pd.read_csv(mres, sep="\t") if main_rules is not None else None
    if main_res is not None:
        a, b = F._contenders(main_res), F._contenders(res)
        nl = ["random_forest", "gradient_boosting", "svm_rbf", "stacking"]
        cmp = pd.DataFrame({"main_mean_auc": a.mean(), "stress_mean_auc": b.mean()}).round(4)
        print("== main random rules vs stress test\n" + cmp.to_string())
        for name, w in (("main", a), ("stress", b)):
            adv = w[nl].max(axis=1) - w["logreg"]
            print(f"best flexible model minus logistic regression, {name}: mean {adv.mean():+.4f} "
                  f"(95% CI {adv.mean() - 1.96 * adv.std() / np.sqrt(len(adv)):+.4f} to {adv.mean() + 1.96 * adv.std() / np.sqrt(len(adv)):+.4f}); "
                  f"flexible model better in {(adv > 0).mean():.0%} of rules")
    else:
        print("main study results not found; the comparison figure will show the stress test only")
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("figure:", F.fig_stress_comparison(main_rules, main_res, rules, res, fig_dir))
    print("\nReminder: labels are simulated; this is a bias check on the simulation, not a result about real iron or zinc deficiency.")


if __name__ == "__main__":
    main()
