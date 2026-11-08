"""Pull the SNP panel out of the big 1000 Genomes VCFs (bcftools in Docker) and QC the result.

Only the logic that does not need Docker lives here, so it can be unit-tested.
"""
from __future__ import annotations

import gzip
from collections import defaultdict
from pathlib import Path

import yaml

VCF_TEMPLATE = "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_{ver}.20130502.genotypes.vcf.gz"
DEFAULT_IMAGE = "staphb/bcftools:latest"


def load_panel(path: Path, only_extract: bool = True) -> list[dict]:
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["panel"]
    return [r for r in rows if r.get("extract", True)] if only_extract else rows


def group_by_chrom(panel: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for r in panel:
        out[str(r["chrom"])].append(r)
    return dict(out)


def regions_arg(snps: list[dict]) -> str:
    """bcftools -r value: one single-base region per SNP."""
    return ",".join(f"{r['chrom']}:{r['pos_grch37']}-{r['pos_grch37']}" for r in snps)


def docker_cmd(workdir: Path, image: str, vcf_name: str, regions: str, out_name: str) -> list[str]:
    """Run bcftools in a Linux container: region query -> split multiallelics -> SNVs only."""
    script = (f"bcftools view -r {regions} /data/{vcf_name} -Ou"
              f" | bcftools norm -m-any -Ou"
              f" | bcftools view -v snps -Oz -o /data/{out_name}")
    return ["docker", "run", "--rm", "-v", f"{workdir}:/data", "--entrypoint", "bash", image, "-c", script]


def read_sample_list(path: Path) -> list[str]:
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.append(line.split()[0])
    return ids


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def _dose(gt: str) -> int:
    """Alt-allele count from a GT string such as 0|1; -1 if any allele is missing."""
    alleles = gt.replace("|", "/").split("/")
    if any(a in (".", "") for a in alleles):
        return -1
    return sum(1 for a in alleles if a != "0")


def read_panel_vcf(path: Path) -> tuple[list[str], list[dict]]:
    samples: list[str] = []
    records: list[dict] = []
    with _open(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                samples = f[9:]
                continue
            fmt = f[8].split(":")
            gi = fmt.index("GT")
            records.append({"chrom": f[0], "pos": int(f[1]), "id": f[2], "ref": f[3], "alt": f[4],
                            "dose": [_dose(x.split(":")[gi]) for x in f[9:]]})
    return samples, records


def match_panel(records: list[dict], panel: list[dict]) -> list[dict]:
    """One row per panel SNP: how many VCF records were found at its position."""
    by_pos: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in records:
        by_pos[(r["chrom"], r["pos"])].append(r)
    out = []
    for p in panel:
        hits = by_pos.get((str(p["chrom"]), int(p["pos_grch37"])), [])
        status = "missing" if not hits else ("ok" if len(hits) == 1 else "multi_alt_records")
        out.append({"rsid": p["rsid"], "gene": p["gene"], "chrom": p["chrom"], "pos": p["pos_grch37"],
                    "vcf_id": ";".join(h["id"] for h in hits), "n_records": len(hits), "status": status})
    return out


def qc_rows(records: list[dict], samples: list[str], groups: dict[str, list[str]],
            min_mac: int = 10) -> list[dict]:
    """Allele frequency per record per sample group. keep_afr is a FLAG, nothing is dropped here."""
    idx = {s: i for i, s in enumerate(samples)}
    rows = []
    for r in records:
        row = {"chrom": r["chrom"], "pos": r["pos"], "vcf_id": r["id"], "ref": r["ref"], "alt": r["alt"]}
        for g, ids in groups.items():
            d = [r["dose"][idx[s]] for s in ids if s in idx]
            called = [x for x in d if x >= 0]
            alt = sum(called)
            n_alleles = 2 * len(called)
            af = alt / n_alleles if n_alleles else float("nan")
            mac = min(alt, n_alleles - alt) if n_alleles else 0
            row.update({f"{g}_n": len(d), f"{g}_missing": len(d) - len(called), f"{g}_alt_af": round(af, 5),
                        f"{g}_mac": mac})
        first = next(iter(groups))
        row["keep_flag"] = row[f"{first}_mac"] >= min_mac
        rows.append(row)
    return rows
