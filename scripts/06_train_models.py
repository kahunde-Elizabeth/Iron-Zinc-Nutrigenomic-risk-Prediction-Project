"""Phase 4: compare algorithms on the simulated labels (P1 only).

For every scenario x trait at the reporting-default setting it runs, per algorithm and feature set:
repeated 5-fold CV (AUC, average precision), leave-one-population-out AUC, and (feature set 'all')
a permutation-label control. --sweep also runs the full h2 grid (CV only, feature set 'all').
Outputs: results/model_results.tsv, results/lopo_by_population.tsv, results/model_sweep.tsv
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from izn.evaluation.cv import cv_metrics, feature_sets, lopo, permutation_auc
from izn.models.zoo import make_models


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/p1_labels.parquet"))
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/label_manifest.tsv"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--perms", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    feats = cfg["features"]
    d = cfg["simulation"]["reporting_default"]
    ft = pq.read_table(args.features).to_pydict()
    lb = pq.read_table(args.labels).to_pydict()
    assert ft["sample"] == lb["sample"], "feature and label rows are not in the same order"
    geno = np.array([ft[f] for f in feats], dtype=float).T
    pops = np.array(ft["pop"])
    with args.manifest.open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh, delimiter="\t"))

    def pick(m):
        return (float(m["prevalence_target"]), float(m["h2"]), float(m["label_noise"])) == \
               (d["prevalence"], d["h2"], d["label_noise"])

    models = make_models(args.seed)
    rows, lopo_rows = [], []
    for m in [m for m in manifest if pick(m)]:
        y = np.array(lb[m["column"]])
        rule = [s.split(":")[0] for s in m["snps"].split(",")]
        for fs_name, X in feature_sets(feats, rule, pops, geno).items():
            for name, model in models.items():
                t0 = time.time()
                r = {"scenario": m["scenario"], "trait": m["trait"], "feature_set": fs_name, "algorithm": name,
                     "oracle_auc": m["oracle_auc"], **cv_metrics(model, X, y, args.reps, args.seed)}
                lp = lopo(model, X, y, pops)
                r["lopo_auc_mean"] = float(np.nanmean(list(lp.values())))
                for p, v in lp.items():
                    lopo_rows.append({"scenario": m["scenario"], "trait": m["trait"], "feature_set": fs_name,
                                      "algorithm": name, "held_out_pop": p, "auc": v})
                if fs_name == "all" and args.perms:
                    r.update(permutation_auc(model, X, y, args.perms, args.seed))
                r["seconds"] = round(time.time() - t0, 1)
                rows.append(r)
                print(f"{m['scenario'][:12]:12s} {m['trait']:5s} {fs_name:9s} {name:18s} "
                      f"AUC={r['auc_mean']:.3f} (oracle {m['oracle_auc']}) LOPO={r['lopo_auc_mean']:.3f}", flush=True)
    write(args.out_dir / "model_results.tsv", rows)
    write(args.out_dir / "lopo_by_population.tsv", lopo_rows)

    if args.sweep:
        sweep = []
        for m in manifest:
            if float(m["prevalence_target"]) != d["prevalence"] or float(m["label_noise"]) != d["label_noise"]:
                continue
            y = np.array(lb[m["column"]])
            for name, model in models.items():
                sweep.append({"scenario": m["scenario"], "trait": m["trait"], "h2": m["h2"], "algorithm": name,
                              "oracle_auc": m["oracle_auc"], **cv_metrics(model, geno, y, args.reps, args.seed)})
        write(args.out_dir / "model_sweep.tsv", sweep)
        print(f"sweep: {len(sweep)} rows -> {args.out_dir / 'model_sweep.tsv'}")
    print(f"done: {len(rows)} rows -> {args.out_dir / 'model_results.tsv'}")


if __name__ == "__main__":
    main()
