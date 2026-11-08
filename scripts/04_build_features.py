"""Phase 3a: build the P1 feature table (Parquet) from the extracted panel genotypes.

Features are raw alt-allele counts (0/1/2, -1 = missing). Risk-allele alignment happens later,
only inside the label rule, so models never depend on an assumed direction.
Output: data/processed/p1_features.parquet (+ p1_feature_meta.tsv with ref/alt per SNP).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from izn.features.panel_features import build_table, read_pop_map
from izn.io.panel_extract import load_panel, read_sample_list


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotypes", type=Path, default=Path("data/interim/panel_genotypes.tsv"))
    ap.add_argument("--samples", type=Path, required=True, help="P1 sample list")
    ap.add_argument("--panel-file", type=Path,
                    default=Path("data/external/integrated_call_samples_v3.20130502.ALL.panel"))
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/p1_features.parquet"))
    args = ap.parse_args()

    features = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))["features"]
    panel = load_panel(args.panel)
    samples = read_sample_list(args.samples)
    cols, meta = build_table(args.genotypes, samples, panel, features, read_pop_map(args.panel_file))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), args.out)
    meta_path = args.out.with_name("p1_feature_meta.tsv")
    with meta_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rsid", "ref", "alt"])
        for f in features:
            w.writerow([f, meta[f]["ref"], meta[f]["alt"]])

    t = pq.read_table(args.out)
    print(f"{t.num_rows} samples x {len(features)} SNPs -> {args.out}")
    pops = {}
    for p in t.column("pop").to_pylist():
        pops[p] = pops.get(p, 0) + 1
    print("populations:", pops)
    miss = sum(1 for f in features for v in t.column(f).to_pylist() if v < 0)
    print("missing genotype calls:", miss)


if __name__ == "__main__":
    main()
