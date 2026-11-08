"""Random-teacher study: many rules whose shape nobody chose (random neural networks and random decision trees).

Same nested-CV comparison as the main study (every algorithm tuned fairly), then compared with the main random rules and
the stress test if their result files exist. Labels are SIMULATED.

Outputs (results/): rr_teacher_rules.tsv, rr_teacher_results.tsv, rr_teacher_summary.tsv, figures/fig18_rule_sources.png,
figures/fig20_teacher_nonadditivity.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from izn.evaluation.blind import CONTENDERS, run_teacher_rule
from izn.evaluation.rr_study import summarise
from izn.evaluation.study_data import load_study_data
from izn.reporting import figures as F
from izn.reporting import figures_blind as FB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--blind-config", type=Path, default=Path("configs/blind_teacher.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--n-rules", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="small grids and 8 rules: a fast check")
    args = ap.parse_args()

    cfg = {**yaml.safe_load(args.config.read_text(encoding="utf-8")), **yaml.safe_load(args.blind_config.read_text(encoding="utf-8"))}
    seed = cfg["seed"] + 2000                           # different random rules from the main study and the stress test
    grids = cfg["quick_grids"] if args.quick else cfg["grids"]
    n_rules = args.n_rules or (8 if args.quick else cfg["teacher_rules"]["n_rules"])
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    print(f"random-teacher study: {n_rules} teachers (random networks / random trees), {'quick' if args.quick else 'full'} grids", flush=True)

    t0 = time.time()
    tasks = (delayed(run_teacher_rule)(i, data, cfg, grids, seed) for i in range(n_rules))
    try:
        gen = Parallel(n_jobs=args.jobs, return_as="generator")(tasks)
    except TypeError:
        gen = Parallel(n_jobs=args.jobs)(tasks)
    rule_rows, res_rows = [], []
    for k, (row, rows) in enumerate(gen, 1):
        rule_rows.append(row)
        res_rows += rows
        best = max((r for r in rows if r["feature_set"] == "snps_plus_env" and f"{r['algorithm']}@{r['tuning']}" in CONTENDERS), key=lambda r: r["auc"])
        print(f"[{k:3d}/{n_rules}] {row['rule_id']} {row['family']:5s} nonadd={row['nonadditive_fraction']:.2f} oracle={row['oracle_auc_full']:.3f} "
              f"best={best['algorithm']}@{best['tuning']} AUC={best['auc']:.3f} ({row['seconds']}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rules = pd.DataFrame(rule_rows).sort_values("rule_id")
    res = pd.DataFrame(res_rows).sort_values(["rule_id", "feature_set", "algorithm", "tuning"])
    rules.to_csv(args.out_dir / "rr_teacher_rules.tsv", sep="\t", index=False)
    res.to_csv(args.out_dir / "rr_teacher_results.tsv", sep="\t", index=False)
    summ = summarise(rules, res)
    pd.concat({k: v.reset_index() for k, v in summ.items()}, names=["table"]).reset_index(level=0) \
        .to_csv(args.out_dir / "rr_teacher_summary.tsv", sep="\t", index=False)
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min\n")
    print(f"teacher non-additive share (measured): mean {rules.nonadditive_fraction.mean():.2f}, "
          f"range {rules.nonadditive_fraction.min():.2f} to {rules.nonadditive_fraction.max():.2f}\n")
    for k in ("model_summary", "tuning_gain", "auc_by_nonadditive_share", "genetic_gain_logreg"):
        print(f"== {k} (RANDOM TEACHERS)\n{summ[k].to_string()}\n")

    groups, nl = [], ["random_forest", "gradient_boosting", "svm_rbf", "stacking"]
    for label, rp in (("random rules", "rr_random_results.tsv"), ("stress test", "rr_stress_results.tsv")):
        if (args.out_dir / rp).exists():
            groups.append((label, F._contenders(pd.read_csv(args.out_dir / rp, sep="\t"))))
    groups.append(("random teachers", F._contenders(res)))
    cmp = pd.DataFrame({lab: w.mean() for lab, w in groups}).round(4)
    print("== mean AUC by rule source\n" + cmp.to_string())
    for lab, w in groups:
        adv = w[nl].max(axis=1) - w["logreg"]
        half = 1.96 * adv.std() / np.sqrt(len(adv))
        print(f"best flexible model minus logistic regression, {lab}: mean {adv.mean():+.4f} (95% CI {adv.mean() - half:+.4f} to {adv.mean() + half:+.4f}); "
              f"flexible model better in {(adv > 0).mean():.0%} of rules")
    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    print("figures:", FB.fig_regimes(groups, fig_dir), FB.fig_teacher_nonadditivity(rules, fig_dir))
    print("\nReminder: labels are simulated; this checks whether the conclusion depends on rules I wrote, not how well real deficiency can be predicted.")


if __name__ == "__main__":
    main()
