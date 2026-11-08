"""Check candidate SNPs against Ensembl GRCh37 (needs internet). Output is a CANDIDATE table:
you must still confirm risk alleles and effects in the source papers before using it."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from izn.io.ensembl import lookup_many

DEFAULT = ["rs855791", "rs4820268", "rs3811647", "rs1799852", "rs1800629", "rs1800562",
           "rs1799945", "rs13419724", "rs13266634","rs114992984", "rs76182486", "rs11558471", "rs3008217", "rs13107325"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rsids", nargs="+", default=DEFAULT)
    ap.add_argument("--out", type=Path, default=Path("data/interim/panel_candidates_grch37.tsv"))
    args = ap.parse_args()
    rows = lookup_many(args.rsids)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rsid_requested", "rsid_returned", "chrom", "pos_grch37", "allele_string", "biallelic_snv",
            "minor_allele", "maf_1000g_global", "genes", "error"]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r.get("rsid_requested"), r.get("chrom"), r.get("pos_grch37"), r.get("allele_string"),
              r.get("genes"), r.get("error", ""))
    print(f"\nwritten: {args.out}  (check gene names against what you expect before using)")


if __name__ == "__main__":
    main()
