"""Look up GRCh37 coordinates for candidate SNPs on the Ensembl GRCh37 REST server."""
from __future__ import annotations

import time

import requests

ENSEMBL_GRCH37 = "https://grch37.rest.ensembl.org"
CANONICAL = {str(i) for i in range(1, 23)} | {"X", "Y"}


def fetch_json(path: str, session: requests.Session | None = None, timeout: int = 30) -> object:
    s = session or requests.Session()
    r = s.get(f"{ENSEMBL_GRCH37}{path}", headers={"Content-Type": "application/json"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def grch37_mapping(record: dict) -> dict | None:
    """Pick the mapping on a standard chromosome (ignores haplotype/patch contigs)."""
    for m in record.get("mappings", []):
        if m.get("assembly_name") == "GRCh37" and str(m.get("seq_region_name")) in CANONICAL:
            return {"chrom": str(m["seq_region_name"]), "pos": int(m["start"]), "end": int(m["end"]),
                    "allele_string": m.get("allele_string", "")}
    return None


def is_biallelic_snv(allele_string: str) -> bool:
    alleles = allele_string.split("/")
    return len(alleles) == 2 and all(len(a) == 1 and a in "ACGT" for a in alleles)


def genes_at(chrom: str, pos: int, session: requests.Session | None = None) -> list[str]:
    data = fetch_json(f"/overlap/region/human/{chrom}:{pos}-{pos}?feature=gene", session)
    return sorted({g.get("external_name", g.get("id", "?")) for g in data})  # type: ignore[union-attr]


def lookup(rsid: str, session: requests.Session | None = None) -> dict:
    rec = fetch_json(f"/variation/human/{rsid}", session)
    m = grch37_mapping(rec)  # type: ignore[arg-type]
    out = {"rsid_requested": rsid, "rsid_returned": rec.get("name"),  # type: ignore[union-attr]
           "minor_allele": rec.get("minor_allele"), "maf_1000g_global": rec.get("MAF"),  # type: ignore[union-attr]
           "chrom": None, "pos_grch37": None, "allele_string": None, "biallelic_snv": False, "genes": ""}
    if m:
        out.update(chrom=m["chrom"], pos_grch37=m["pos"], allele_string=m["allele_string"],
                   biallelic_snv=is_biallelic_snv(m["allele_string"]))
        out["genes"] = ";".join(genes_at(m["chrom"], m["pos"], session))
    return out


def lookup_many(rsids: list[str], pause: float = 0.4) -> list[dict]:
    session = requests.Session()
    rows = []
    for r in rsids:
        try:
            rows.append(lookup(r, session))
        except requests.RequestException as exc:
            rows.append({"rsid_requested": r, "error": str(exc)})
        time.sleep(pause)
    return rows
