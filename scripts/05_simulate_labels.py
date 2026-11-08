"""Phase 3b: simulate iron and zinc risk labels for every scenario / prevalence / h2 / noise setting.

Reads data/processed/p1_features.parquet. Writes data/processed/p1_labels.parquet (one 0/1 column per
setting, named scenario__trait__pXX__hXX__nXX) and label_manifest.tsv describing each column.
`oracle_auc` is the AUC of the true SNP score against the label: the ceiling no model should beat.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from izn.labels.simulate import auc, draws, make_labels, score, zscore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/p1_labels.parquet"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    panel = {p["rsid"]: p for p in yaml.safe_load(args.panel.read_text(encoding="utf-8"))["panel"]}
    sim = cfg["simulation"]
    table = pq.read_table(args.features).to_pydict()
    n = len(table["sample"])
    with args.meta.open(encoding="utf-8") as fh:
        meta = {r["rsid"]: r for r in csv.DictReader(fh, delimiter="\t")}

    out = {"sample": table["sample"], "pop": table["pop"]}
    manifest = []
    for scen, traits in cfg["scenarios"].items():
        for trait in ("iron", "zinc"):
            snps = traits.get(trait) or {}
            if not snps:
                continue
            weights = (traits.get("weights") or {}).get(trait)
            if weights and set(weights) != set(snps):
                raise SystemExit(f"{scen}/{trait}: weights must cover exactly the label SNPs")
            for rsid, allele in snps.items():           # the config must agree with the verified panel
                if panel[rsid]["risk_allele"] != allele:
                    raise SystemExit(f"{rsid}: scenario says {allele}, panel says {panel[rsid]['risk_allele']}")
            raw = score(table, meta, snps, weights)
            z = zscore(raw)
            out[f"{scen}__{trait}__score"] = raw
            eps, flip_u = draws(sim["seed"], trait, n)
            for prev in sim["prevalence"]:
                for h2 in sim["h2"]:
                    for noise in sim["label_noise"]:
                        name = f"{scen}__{trait}__p{int(prev*100)}__h{int(h2*100)}__n{int(noise*100)}"
                        y = make_labels(z, eps, flip_u, h2, prev, noise)
                        out[name] = y
                        manifest.append({"column": name, "scenario": scen, "trait": trait,
                                         "snps": ",".join(f"{k}:{v}" for k, v in snps.items()),
                                         "weighting": "beta" if weights else "equal",
                                         "prevalence_target": prev, "h2": h2, "label_noise": noise,
                                         "positives": sum(y), "prevalence_achieved": round(sum(y) / n, 4),
                                         "oracle_auc": round(auc(z, y), 4)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(out), args.out)
    mpath = args.out.with_name("label_manifest.tsv")
    with mpath.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]), delimiter="\t")
        w.writeheader()
        w.writerows(manifest)
    print(f"{n} samples, {len(manifest)} label columns -> {args.out}")
    d = cfg["simulation"]["reporting_default"]
    print("reporting default settings:")
    for m in manifest:
        if (m["prevalence_target"], m["h2"], m["label_noise"]) == (d["prevalence"], d["h2"], d["label_noise"]):
            print(f"  {m['scenario']:26s} {m['trait']:5s} positives={m['positives']}  oracle_auc={m['oracle_auc']}")


if __name__ == "__main__":
    main()
