"""Scenario grid: the named SNP scenarios (configs/scenarios.yaml) x four label structures, side by side.

Structures (configs/randomized_study.yaml): additive, gxe, and_threshold, strong_nonadd. Each uses the same total
signal share, so the only thing that changes is how non-additive the rule is. Every algorithm is tuned with nested
CV; noise replicates give a mean and spread. Labels and covariates are SIMULATED.

Outputs (results/): rr_scenario_grid.tsv, rr_scenario_summary.tsv, rr_reference.parquet (features, labels and
out-of-fold scores of the reference run, used by the figures: correlation, confusion matrix, ROC).
"""
from __future__ import annotations

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from joblib import Parallel, delayed

from izn.evaluation.nested import make_specs, nested_evaluate
from izn.evaluation.study_data import design, design_columns, load_study_data
from izn.labels.random_rules import make_labels, named_rule, nonadditive_fraction

CONTENDERS = ["logreg@tuned", "random_forest@tuned", "gradient_boosting@tuned", "svm_rbf@tuned", "stacking@default"]


def cases(scn_cfg: dict, structures: dict) -> list[dict]:
    """(scenario, trait) pairs with a rule; scenarios that repeat another's SNPs and weights are skipped and noted."""
    seen, out = {}, []
    for scen, body in scn_cfg["scenarios"].items():
        for trait in ("iron", "zinc"):
            snps = body.get(trait) or {}
            if not snps:
                continue
            weights = (body.get("weights") or {}).get(trait)
            key = (trait, tuple(sorted(snps.items())), tuple(sorted((weights or {}).items())))
            if key in seen:
                print(f"note: {scen}/{trait} has the same SNPs and weights as {seen[key]}; not repeated")
                continue
            seen[key] = scen
            out.append({"scenario": scen, "trait": trait, "snps": snps, "weights": weights})
    return out


def run_case(case: dict, structure_name: str, rep: int, data: dict, cfg: dict, grids: dict, seed: int,
             keep_oof: bool) -> tuple[list[dict], dict | None]:
    sg, cv = cfg["scenario_grid"], cfg["cv"]
    env_coef = data["env_cfg"]["env_coefficients"][case["trait"]]
    rule = named_rule(f"{case['scenario']}__{case['trait']}__{structure_name}", case["trait"], case["snps"],
                      case["weights"], data["meta"], env_coef, sg["structures"][structure_name],
                      sg["prevalence"], sg["label_noise"])
    rng = np.random.default_rng([seed, rep, zlib.crc32(rule["name"].encode())])
    y, info = make_labels(rule, data["geno"], data["cov"], rng)
    fseed = seed + rep
    specs = make_specs(grids, fseed)
    rows = []

    def add(fs, res):
        for spec, r in res.items():
            algo, tuning = spec.split("@")
            rows.append({"scenario": case["scenario"], "trait": case["trait"], "structure": structure_name,
                         "nonadditive_fraction": round(nonadditive_fraction(rule), 3), "replicate": rep,
                         "feature_set": fs, "algorithm": algo, "tuning": tuning, "auc": r["auc"], "auc_sd": r["auc_sd"],
                         "ap": r["ap"], **info})

    main = nested_evaluate(design(data, "snps_plus_env"), y, specs, cv["outer_splits"], cv["inner_splits"], fseed, keep_oof)
    add("snps_plus_env", {k: {kk: vv for kk, vv in v.items() if kk != "oof"} for k, v in main.items()})
    lr = {"logreg@tuned": specs["logreg@tuned"]}
    for fs in ("env_only", "snps_only"):
        add(fs, nested_evaluate(design(data, fs), y, lr, cv["outer_splits"], cv["inner_splits"], fseed))
    ref = None
    if keep_oof:
        ref = {"y": y, **{f"oof__{k}": v["oof"] for k, v in main.items() if k in CONTENDERS}}
    return rows, ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--reference", default="evidence_only:iron:gxe",
                    help="scenario:trait:structure whose first replicate is saved for the confusion matrix / ROC figures")
    ap.add_argument("--replicates", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="small grids, 1 replicate: a fast check")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else cfg["seed"]
    grids = cfg["quick_grids"] if args.quick else cfg["grids"]
    reps = args.replicates or (1 if args.quick else cfg["scenario_grid"]["replicates"])
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    scn = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    if args.panel.exists():                          # same consistency check as 05_simulate_labels.py
        panel = {p["rsid"]: p for p in yaml.safe_load(args.panel.read_text(encoding="utf-8"))["panel"]}
        for scen, body in scn["scenarios"].items():
            for trait in ("iron", "zinc"):
                for rsid, allele in (body.get(trait) or {}).items():
                    if panel[rsid]["risk_allele"] != allele:
                        raise SystemExit(f"{rsid}: scenario says {allele}, panel says {panel[rsid]['risk_allele']}")
    structures = cfg["scenario_grid"]["structures"]
    todo = cases(scn, structures)
    ref_s, ref_t, ref_st = args.reference.split(":")
    if (ref_s, ref_t) not in {(c["scenario"], c["trait"]) for c in todo} or ref_st not in structures:
        ref_s, ref_t, ref_st = todo[0]["scenario"], todo[0]["trait"], next(iter(structures))
        print(f"note: reference not found; using {ref_s}:{ref_t}:{ref_st}")
    jobs = [(c, st, r) for c in todo for st in structures for r in range(reps)]
    print(f"{len(todo)} scenario/trait cases x {len(structures)} structures x {reps} replicates = {len(jobs)} runs "
          f"({'quick' if args.quick else 'full'} grids)", flush=True)

    def is_ref(c, st, r):
        return (c["scenario"], c["trait"], st, r) == (ref_s, ref_t, ref_st, 0)

    t0 = time.time()
    tasks = (delayed(run_case)(c, st, r, data, cfg, grids, seed, is_ref(c, st, r)) for c, st, r in jobs)
    try:
        gen = Parallel(n_jobs=args.jobs, return_as="generator")(tasks)
    except TypeError:                                     # older joblib: results arrive all at the end
        gen = Parallel(n_jobs=args.jobs)(tasks)
    rows, ref = [], None
    for k, (rr, rf) in enumerate(gen, 1):
        rows += rr
        if rf is not None:
            ref = rf
        a = next(x for x in rr if x["feature_set"] == "snps_plus_env" and x["algorithm"] == "logreg" and x["tuning"] == "tuned")
        print(f"[{k:3d}/{len(jobs)}] {a['scenario']:26s} {a['trait']:5s} {a['structure']:14s} rep{a['replicate']} "
              f"logreg(tuned)={a['auc']:.3f} oracle={a['oracle_auc_full']:.3f}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_dir / "rr_scenario_grid.tsv", sep="\t", index=False)
    main_df = df[df.feature_set == "snps_plus_env"].copy()
    main_df["model"] = main_df.algorithm + "@" + main_df.tuning
    cont = main_df[main_df.model.isin(CONTENDERS)]
    summ = (cont.groupby(["scenario", "trait", "structure", "model"]).agg(auc=("auc", "mean"), sd=("auc", "std"),
            oracle=("oracle_auc_full", "mean")).round(3).reset_index())
    summ.to_csv(args.out_dir / "rr_scenario_summary.tsv", sep="\t", index=False)
    if ref is not None:
        cols = design_columns(data)
        X = design(data, "snps_plus_env")
        tbl = {"sample": data["sample"], "pop": list(data["pop"]), **{c: X[:, i] for i, c in enumerate(cols)}, **ref}
        pq.write_table(pa.table(tbl), args.out_dir / "rr_reference.parquet")
        pd.Series({"reference": f"{ref_s}:{ref_t}:{ref_st}"}).to_csv(args.out_dir / "rr_reference_name.txt", index=False, header=False)
    else:
        print("warning: reference run missing; confusion-matrix figures will be skipped")

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min\n")
    wide = summ.pivot_table(index=["scenario", "trait", "model"], columns="structure", values="auc")
    wide = wide[[s for s in structures if s in wide.columns]]
    print("tuned AUC (snps_plus_env), mean over replicates:")
    print(wide.round(3).to_string())
    print("\nReminder: labels are simulated; these AUCs describe recovery of assumed rules, not real deficiency prediction.")


if __name__ == "__main__":
    main()
