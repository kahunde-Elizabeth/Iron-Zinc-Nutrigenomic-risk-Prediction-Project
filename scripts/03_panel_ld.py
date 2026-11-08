"""Correlation (r^2) between panel SNPs inside one sample group, to spot near-duplicate SNPs.

Reads data/interim/panel_genotypes.tsv (0/1/2 alt-allele doses, -1 = missing).
Uses genotype correlation, which approximates LD without needing phased haplotypes.
"""
from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path


def pearson_r2(x: list[int], y: list[int]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if a >= 0 and b >= 0]
    n = len(pairs)
    if n < 3:
        return float("nan")
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    sxx = sum((a - mx) ** 2 for a, _ in pairs)
    syy = sum((b - my) ** 2 for _, b in pairs)
    if sxx == 0 or syy == 0:
        return float("nan")          # monomorphic in this group
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    return (sxy / math.sqrt(sxx * syy)) ** 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genotypes", type=Path, default=Path("data/interim/panel_genotypes.tsv"))
    ap.add_argument("--samples", type=Path, required=True, help="sample list for the group (e.g. P1)")
    ap.add_argument("--min-r2", type=float, default=0.2, help="only print pairs at or above this r^2")
    args = ap.parse_args()

    keep = {l.split()[0] for l in args.samples.read_text(encoding="utf-8").splitlines() if l.strip()}
    with args.genotypes.open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        rows = [r for r in rd if r[0] in keep]
    cols = {name: [int(r[i + 1]) for r in rows] for i, name in enumerate(header[1:])}
    print(f"{len(rows)} samples in group; {len(cols)} SNP columns\n")
    found = 0
    for a, b in combinations(cols, 2):
        r2 = pearson_r2(cols[a], cols[b])
        if r2 == r2 and r2 >= args.min_r2:
            found += 1
            print(f"r2={r2:.2f}  {a}  <->  {b}")
    if not found:
        print(f"no pair with r2 >= {args.min_r2}")
    mono = [n for n, v in cols.items() if len(set(x for x in v if x >= 0)) < 2]
    if mono:
        print("\nmonomorphic in this group (cannot be correlated):", ", ".join(mono))


if __name__ == "__main__":
    main()
