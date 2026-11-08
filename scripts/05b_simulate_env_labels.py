"""Phase 4b: labels from genes + synthetic environment (diet, malaria, inflammation, blood loss, age, pregnancy).

Reads the real P1 genotypes, invents the covariates, and writes:
  data/processed/p1_env.parquet            synthetic covariates (one row per sample)
  data/processed/p1_labels_env.parquet     0/1 labels per setting
  data/processed/label_manifest_env.tsv    settings + two ceilings: oracle_auc_genetic, oracle_auc_full
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from izn.labels.environment import COVARIATES, env_score, generate_covariates, interaction, liability
from izn.labels.simulate import auc, draws, score, threshold_labels, zscore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = ap.parse_args()

    sc = yaml.safe_load(args.scenarios.read_text(encoding="utf-8"))
    env = yaml.safe_load(args.env.read_text(encoding="utf-8"))
    panel = {p["rsid"]: p for p in yaml.safe_load(args.panel.read_text(encoding="utf-8"))["panel"]}
    table = pq.read_table(args.features).to_pydict()
    n = len(table["sample"])
    with args.meta.open(encoding="utf-8") as fh:
        meta = {r["rsid"]: r for r in csv.DictReader(fh, delimiter="\t")}

    cov = generate_covariates(n, env, env["seed"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"sample": table["sample"], "pop": table["pop"], **cov}), args.out_dir / "p1_env.parquet")

    vs = env["variance_shares"]
    out = {"sample": table["sample"], "pop": table["pop"]}
    manifest = []
    for scen, traits in sc["scenarios"].items():
        for trait in ("iron", "zinc"):
            snps = traits.get(trait) or {}
            if not snps:
                continue
            for rsid, allele in snps.items():
                if panel[rsid]["risk_allele"] != allele:
                    raise SystemExit(f"{rsid}: scenario says {allele}, panel says {panel[rsid]['risk_allele']}")
            weights = (traits.get("weights") or {}).get(trait)
            z_g = zscore(score(table, meta, snps, weights))
            z_e = zscore(env_score(cov, env["env_coefficients"][trait]))
            z_ge = interaction(z_g, cov["inflammation"])
            eps, flip_u = draws(env["seed"], trait, n)
            for prev in env["prevalence"]:
                for hg in vs["h_g"]:
                    for hge in vs["h_ge"]:
                        for noise in env["label_noise"]:
                            comps = [(hg, z_g), (hge, z_ge), (vs["h_env"], z_e)]
                            y = threshold_labels(liability(comps, eps), flip_u, prev, noise)
                            signal = liability(comps, [0.0] * n)
                            name = (f"{scen}__{trait}__p{round(prev*100)}__g{round(hg*100)}"
                                    f"__x{round(hge*100)}__n{round(noise*100)}")
                            out[name] = y
                            manifest.append({"column": name, "scenario": scen, "trait": trait,
                                             "snps": ",".join(f"{k}:{v}" for k, v in snps.items()),
                                             "prevalence_target": prev, "h_g": hg, "h_ge": hge,
                                             "h_env": vs["h_env"], "label_noise": noise,
                                             "prevalence_achieved": round(sum(y) / n, 4),
                                             "oracle_auc_genetic": round(auc(z_g, y), 4),
                                             "oracle_auc_full": round(auc(signal, y), 4)})
    pq.write_table(pa.table(out), args.out_dir / "p1_labels_env.parquet")
    mp = args.out_dir / "label_manifest_env.tsv"
    with mp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0]), delimiter="\t")
        w.writeheader()
        w.writerows(manifest)
    d = env["reporting_default"]
    print(f"{n} samples, {len(manifest)} label columns, {len(COVARIATES)} synthetic covariates")
    print("reporting default (ceilings: genetic part alone / genes + environment):")
    for m in manifest:
        if (m["prevalence_target"], m["h_g"], m["h_ge"], m["label_noise"]) == (d["prevalence"], d["h_g"], d["h_ge"], d["label_noise"]):
            print(f"  {m['scenario']:26s} {m['trait']:5s} genetic={m['oracle_auc_genetic']}  full={m['oracle_auc_full']}")


if __name__ == "__main__":
    main()
