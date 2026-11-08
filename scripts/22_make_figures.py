"""Make every report figure from the outputs of scripts 20 and 21 and the real genotype table.

Writes PNGs to results/figures/. Figures whose inputs are missing are skipped with a message.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from izn.reporting import figures as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    feats = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))["features"]
    ft = pq.read_table(args.features).to_pandas()
    geno, pops = ft[feats], ft["pop"].to_numpy()
    made = [F.fig_snp_ld(geno, args.out_dir), F.fig_allele_freq(geno, pops, args.out_dir), F.fig_pca(geno, pops, args.out_dir)]

    def read(name):
        p = args.results / name
        return pd.read_csv(p, sep="\t") if p.exists() else None

    ref_p = args.results / "rr_reference.parquet"
    if ref_p.exists():
        ref = pq.read_table(ref_p).to_pandas()
        name_p = args.results / "rr_reference_name.txt"
        ref_name = name_p.read_text().strip() if name_p.exists() else "reference run"
        made += [F.fig_correlation(ref, feats, args.out_dir, ref_name), F.fig_label_correlation(ref, feats, args.out_dir, ref_name),
                 F.fig_confusion(ref, feats, args.out_dir, ref_name), F.fig_roc_pr(ref, feats, args.out_dir, ref_name),
                 F.fig_importance(ref, feats, args.out_dir, ref_name)]
    else:
        print("skipped: correlation, confusion matrix, ROC, importance (run scripts/21_scenario_grid.py first)")

    rules, res, grid = read("rr_random_rules.tsv"), read("rr_random_results.tsv"), read("rr_scenario_grid.tsv")
    if rules is not None and res is not None:
        made += [F.fig_random_summary(rules, res, args.out_dir), F.fig_vs_nonadditive(rules, res, args.out_dir),
                 F.fig_tuning_gain(res, args.out_dir), F.fig_rule_distribution(rules, args.out_dir)]
    else:
        print("skipped: random-study figures (run scripts/20_randomized_rule_study.py first)")
    if grid is not None:
        made += [F.fig_scenario_grid(grid, args.out_dir), F.fig_genetic_gain(grid, rules, res, args.out_dir)]
    else:
        print("skipped: scenario-grid figures (run scripts/21_scenario_grid.py first)")
    print(f"{len(made)} figures written to {args.out_dir}:")
    for p in made:
        print("  ", p.name)


if __name__ == "__main__":
    main()
