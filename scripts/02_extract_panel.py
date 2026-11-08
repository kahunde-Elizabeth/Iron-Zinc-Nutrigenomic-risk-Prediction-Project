"""Phase 2: extract the SNP panel from the VCFs in the object store.

For each chromosome: download that VCF + index from the store to a staging folder (outside the repo),
run bcftools in Docker to cut out only the panel positions, keep the small result, delete the big file.
Writes: data/interim/panel_genotypes.tsv (samples x SNPs, 0/1/2, -1 = missing) and panel_qc.tsv.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
from pathlib import Path

from izn.io.ingest import StoreConfig, ensure_bucket, make_client
from izn.io.panel_extract import (DEFAULT_IMAGE, VCF_TEMPLATE, docker_cmd, group_by_chrom, load_panel,
                                  match_panel, qc_rows, read_panel_vcf, read_sample_list, regions_arg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--bucket", default="vcf-raw")
    ap.add_argument("--out-bucket", default="panel")
    ap.add_argument("--ver", default="v5a", choices=["v5a", "v5b"])
    ap.add_argument("--staging", type=Path, required=True, help="temporary folder OUTSIDE the repo")
    ap.add_argument("--out-dir", type=Path, default=Path("data/interim"))
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--sample-list", nargs="+", required=True, metavar="NAME=PATH",
                    help="e.g. P1=samples_P1.txt LWK=samples_LWK.txt; the FIRST one is used for keep_flag")
    ap.add_argument("--min-mac", type=int, default=10)
    args = ap.parse_args()

    groups = {}
    for item in args.sample_list:
        name, path = item.split("=", 1)
        groups[name] = read_sample_list(Path(path))

    panel = load_panel(args.panel)
    client = make_client(StoreConfig(os.environ["S3_ENDPOINT"], os.environ["S3_ACCESS_KEY"],
                                     os.environ["S3_SECRET_KEY"]))
    ensure_bucket(client, args.out_bucket)
    args.staging.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_records, samples = [], None
    for chrom, snps in group_by_chrom(panel).items():
        name = VCF_TEMPLATE.format(chrom=chrom, ver=args.ver)
        out_name = f"panel_chr{chrom}.vcf.gz"
        print(f"chr{chrom}: {len(snps)} SNPs; fetching {name} from the store ...", flush=True)
        for n in (name, name + ".tbi"):
            if not (args.staging / n).exists():
                client.download_file(args.bucket, n, str(args.staging / n))
        cmd = docker_cmd(args.staging, args.image, name, regions_arg(snps), out_name)
        subprocess.run(cmd, check=True)
        client.upload_file(str(args.staging / out_name), args.out_bucket, out_name)
        smp, recs = read_panel_vcf(args.staging / out_name)
        samples = samples or smp
        assert smp == samples, "sample order differs between chromosomes"
        all_records += recs
        for n in (name, name + ".tbi"):
            (args.staging / n).unlink(missing_ok=True)   # big files are removed, the store keeps them

    # genotype matrix
    cols = [f"{r['chrom']}:{r['pos']}:{r['ref']}>{r['alt']}" for r in all_records]
    with (args.out_dir / "panel_genotypes.tsv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["sample"] + cols)
        for i, s in enumerate(samples):
            w.writerow([s] + [r["dose"][i] for r in all_records])

    # match + QC
    match = match_panel(all_records, panel)
    qc = qc_rows(all_records, samples, groups, args.min_mac)
    for path, rows in (("panel_match.tsv", match), ("panel_qc.tsv", qc)):
        with (args.out_dir / path).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
    print(f"{len(samples)} samples, {len(all_records)} records")
    for m in match:
        print(f"  {m['rsid']:12s} {m['gene']:9s} {m['status']:18s} vcf_id={m['vcf_id']}")
    print(f"written to {args.out_dir}: panel_genotypes.tsv, panel_match.tsv, panel_qc.tsv")


if __name__ == "__main__":
    main()
