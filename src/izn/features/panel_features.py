"""Turn the extracted panel genotypes into a small model-ready table (one row per sample)."""
from __future__ import annotations

import csv
from pathlib import Path


def read_pop_map(panel_file: Path) -> dict[str, tuple[str, str]]:
    """1000 Genomes .panel file: sample, pop, super_pop, gender (tab separated, header first)."""
    out: dict[str, tuple[str, str]] = {}
    with Path(panel_file).open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        next(rd)
        for r in rd:
            if len(r) >= 3 and r[0]:
                out[r[0]] = (r[1], r[2])
    return out


def column_to_rsid(columns: list[str], panel: list[dict]) -> dict[str, str]:
    """Genotype columns are named chrom:pos:ref>alt; map them to rsIDs through the panel config."""
    by_pos = {(str(p["chrom"]), int(p["pos_grch37"])): p["rsid"] for p in panel}
    out = {}
    for c in columns:
        chrom, pos, _ = c.split(":", 2)
        out[c] = by_pos[(chrom, int(pos))]
    return out


def build_table(genotypes: Path, sample_ids: list[str], panel: list[dict], features: list[str],
                pop_map: dict[str, tuple[str, str]]) -> tuple[dict[str, list], dict[str, dict]]:
    """Return (columns dict ready for a Parquet table, per-SNP meta with ref/alt)."""
    with Path(genotypes).open(encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        rows = {r[0]: r[1:] for r in rd}
    missing = [s for s in sample_ids if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} samples not in genotype file, e.g. {missing[:3]}")
    names = column_to_rsid(header[1:], panel)
    idx = {names[c]: i for i, c in enumerate(header[1:])}
    absent = [f for f in features if f not in idx]
    if absent:
        raise ValueError(f"features not in genotype file: {absent}")
    meta = {}
    for c in header[1:]:
        _, _, alleles = c.split(":", 2)
        ref, alt = alleles.split(">")
        meta[names[c]] = {"ref": ref, "alt": alt}
    cols: dict[str, list] = {"sample": list(sample_ids),
                             "pop": [pop_map.get(s, ("NA", "NA"))[0] for s in sample_ids],
                             "super_pop": [pop_map.get(s, ("NA", "NA"))[1] for s in sample_ids]}
    for f in features:
        cols[f] = [int(rows[s][idx[f]]) for s in sample_ids]
    return cols, meta
