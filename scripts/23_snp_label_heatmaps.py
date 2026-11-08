"""Heat maps relating the SNPs to the synthetic covariates and to the simulated labels.

fig15: SNP x (covariates + label) for the reference run (needs results/rr_reference.parquet from script 21).
fig16: SNP x (scenario | trait | structure): correlation of each SNP's dosage with the label of that rule
       (same seeds as script 21, replicate 0). Boxed cells are the SNPs the rule actually uses.
Also writes results/snp_label_correlation.tsv. Labels and covariates are SIMULATED.
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from izn.evaluation.study_data import load_study_data, scenario_cases
from izn.labels.random_rules import make_labels, named_rule
from izn.reporting import figures as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    feats = data["features"]
    made = []

    ref_p = args.results / "rr_reference.parquet"
    if ref_p.exists():
        ref = pq.read_table(ref_p).to_pandas()
        name_p = args.results / "rr_reference_name.txt"
        made.append(F.fig_snp_vs_variables(ref, feats, args.out_dir, name_p.read_text().strip() if name_p.exists() else "reference run"))
    else:
        print("skipped fig15: run scripts/21_scenario_grid.py first (it writes rr_reference.parquet)")

    scn = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    sg, seed = cfg["scenario_grid"], cfg["seed"]
    cols, in_rule = {}, {}
    for case in scenario_cases(scn):
        for st_name, st in sg["structures"].items():
            rule = named_rule(f"{case['scenario']}__{case['trait']}__{st_name}", case["trait"], case["snps"], case["weights"],
                              data["meta"], data["env_cfg"]["env_coefficients"][case["trait"]], st, sg["prevalence"], sg["label_noise"])
            rng = np.random.default_rng([seed, 0, zlib.crc32(rule["name"].encode())])      # same as script 21, replicate 0
            y, _ = make_labels(rule, data["geno"], data["cov"], rng)
            key = f"{case['scenario']} | {case['trait']} | {st_name}"
            cols[key] = [np.corrcoef(data["geno"][f], y)[0, 1] if data["geno"][f].std() > 0 else np.nan for f in feats]
            in_rule[key] = [f in case["snps"] for f in feats]
    corr = pd.DataFrame(cols, index=feats)
    corr.round(4).to_csv(args.results / "snp_label_correlation.tsv", sep="\t")
    made.append(F.fig_snp_label_by_rule(corr, pd.DataFrame(in_rule, index=feats), args.out_dir, len(data["sample"])))
    print(f"{len(made)} figures written to {args.out_dir}: " + ", ".join(p.name for p in made))
    inr = pd.DataFrame(in_rule, index=feats)
    print(f"\nmean |r| with the label, SNPs IN the rule: {np.nanmean(np.abs(corr.values[inr.values])):.3f}; "
          f"SNPs NOT in the rule: {np.nanmean(np.abs(corr.values[~inr.values])):.3f} (chance level ~ {0.8 / np.sqrt(len(data['sample'])):.3f})")


if __name__ == "__main__":
    main()
