"""Load the real genotype table and the synthetic covariates in the shape the randomized study needs."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from izn.labels.environment import COVARIATES, generate_covariates


def load_study_data(features: Path, scenarios: Path, env_cfg_path: Path, env_data: Path | None = None,
                    meta_path: Path | None = None) -> dict:
    feats = yaml.safe_load(Path(scenarios).read_text(encoding="utf-8"))["features"]
    ft = pq.read_table(features).to_pydict()
    missing = [f for f in feats if f not in ft]
    if missing:
        raise SystemExit(f"features table lacks {missing}: rerun scripts/04_build_features.py first")
    env_cfg = yaml.safe_load(Path(env_cfg_path).read_text(encoding="utf-8"))
    n = len(ft["sample"])
    if env_data is not None and Path(env_data).exists():
        ev = pq.read_table(env_data).to_pydict()
        if ev["sample"] != ft["sample"]:
            raise SystemExit("p1_env.parquet rows are not in the same order as the features table")
        cov = {c: np.array(ev[c], dtype=float) for c in COVARIATES}
    else:                                             # same seeded generator as scripts/05b_simulate_env_labels.py
        cov = {c: np.array(v, dtype=float) for c, v in generate_covariates(n, env_cfg, env_cfg["seed"]).items()}
    cov_cols = [c for c in COVARIATES if len(set(cov[c])) > 1]
    meta = {}
    if meta_path is not None and Path(meta_path).exists():
        with Path(meta_path).open(encoding="utf-8") as fh:
            meta = {r["rsid"]: r for r in csv.DictReader(fh, delimiter="\t")}
    return {"sample": ft["sample"], "pop": np.array(ft["pop"]), "features": feats,
            "geno": {f: np.array(ft[f], dtype=float) for f in feats}, "cov": cov, "cov_cols": cov_cols,
            "env_cfg": env_cfg, "meta": meta}


def design(data: dict, which: str) -> np.ndarray:
    G = np.column_stack([data["geno"][f] for f in data["features"]])
    E = np.column_stack([data["cov"][c] for c in data["cov_cols"]])
    return {"snps_only": G, "env_only": E, "snps_plus_env": np.hstack([G, E])}[which]


def design_columns(data: dict) -> list[str]:
    return list(data["features"]) + list(data["cov_cols"])


def scenario_cases(scn_cfg: dict) -> list[dict]:
    """(scenario, trait) pairs that have a rule; a scenario repeating another's SNPs and weights is skipped."""
    seen, out = {}, []
    for scen, body in scn_cfg["scenarios"].items():
        for trait in ("iron", "zinc"):
            snps = body.get(trait) or {}
            if not snps:
                continue
            weights = (body.get("weights") or {}).get(trait)
            key = (trait, tuple(sorted(snps.items())), tuple(sorted((weights or {}).items())))
            if key in seen:
                continue
            seen[key] = scen
            out.append({"scenario": scen, "trait": trait, "snps": snps, "weights": weights})
    return out
