"""Random and named label RULES for the randomized-rule study (all SIMULATED; genotypes are real).

A rule turns real genotypes + synthetic covariates into a liability, then into 0/1 labels:

    signal = sqrt(h_g) z(genetic) + sqrt(h_env) z(environment) + sqrt(h_nl) z(non-additive part)
    signal is rescaled to variance h_total; liability = signal + sqrt(1 - h_total) * noise
    label = top `prevalence` fraction of liability, then a `noise` fraction of labels flipped.

The non-additive part can hold: SNP x covariate products (gxe, gscore_x), SNP x SNP epistasis (gxg),
two covariates that must both be in their risk direction (exe_and) and a genetic-score threshold (gthresh).
Rules are plain dicts so they can be written to a table and re-created.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def zs(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return np.zeros_like(x) if sd < 1e-12 else (x - x.mean()) / sd


def coded_dose(g: np.ndarray, flip: bool, coding: str) -> np.ndarray:
    d = 2.0 - g if flip else np.asarray(g, dtype=float)
    if coding == "additive":
        return d
    if coding == "dominant":
        return (d >= 1).astype(float)
    if coding == "recessive":
        return (d >= 2).astype(float)
    raise ValueError(f"unknown coding {coding}")


def high(x: np.ndarray, direction: int) -> np.ndarray:
    """True where a covariate sits on its risk side (binary: the 1 or the 0; continuous: top/bottom 40%)."""
    x = np.asarray(x, dtype=float)
    if set(np.unique(x)) <= {0.0, 1.0}:
        return (x == 1.0) if direction > 0 else (x == 0.0)
    cut = np.quantile(x, 0.6 if direction > 0 else 0.4)
    return x >= cut if direction > 0 else x <= cut


def genetic_z(snps: list[dict], geno: dict[str, np.ndarray]) -> np.ndarray:
    tot = 0.0
    for s in snps:
        tot = tot + s["weight"] * coded_dose(geno[s["col"]], s["flip"], s["coding"])
    return zs(tot)


def env_z(env: dict[str, float], cov: dict[str, np.ndarray]) -> np.ndarray:
    tot = 0.0
    for k, c in env.items():
        tot = tot + c * zs(cov[k])
    return zs(tot)


def term_raw(t: dict, geno, cov, gz, snps) -> np.ndarray:
    kind = t["type"]
    if kind == "gxe":
        return zs(geno[t["snp"]]) * zs(cov[t["cov"]])
    if kind == "gscore_x":
        return gz * zs(cov[t["cov"]])
    if kind == "gxg":
        a, b = t.get("a"), t.get("b")
        if a is None:
            if len(snps) < 2:
                return np.zeros_like(gz)
            a, b = snps[0]["col"], snps[1]["col"]
        return ((geno[a] > 0) & (geno[b] > 0)).astype(float)
    if kind == "gxg_prod":                       # product of two standardised dosages: no main effect when SNPs are independent
        return zs(geno[t["a"]]) * zs(geno[t["b"]])
    if kind == "exe_prod":                       # product of two standardised covariates
        return zs(cov[t["a"]]) * zs(cov[t["b"]])
    if kind == "exe_and":
        return (high(cov[t["a"]], t["dir_a"]) & high(cov[t["b"]], t["dir_b"])).astype(float)
    if kind == "gthresh":
        return (gz > np.quantile(gz, t["q"])).astype(float)
    raise ValueError(f"unknown term type {kind}")


def nonadditive_z(terms: list[dict], geno, cov, gz, snps) -> np.ndarray:
    tot = np.zeros_like(gz)
    for t in terms:
        tot = tot + zs(term_raw(t, geno, cov, gz, snps))
    return zs(tot)


def residualise(x: np.ndarray, geno: dict, cov: dict) -> np.ndarray:
    """Remove everything a linear model on all SNPs and covariates could use (least-squares residual)."""
    B = np.column_stack([np.ones(len(x))] + [np.asarray(v, float) for v in geno.values()] + [np.asarray(v, float) for v in cov.values()])
    coef, *_ = np.linalg.lstsq(B, x, rcond=None)
    return x - B @ coef


def components(rule: dict, geno, cov) -> dict[str, np.ndarray]:
    gz = genetic_z(rule["snps"], geno)
    nonadd = nonadditive_z(rule["terms"], geno, cov, gz, rule["snps"]) if rule["terms"] else np.zeros_like(gz)
    if rule.get("orthogonalise") and rule["terms"]:      # stress test: the non-additive part carries NO linear signal at all
        nonadd = zs(residualise(nonadd, geno, cov))
    return {"genetic": gz, "env": env_z(rule["env"], cov), "nonadd": nonadd}


def signal(rule: dict, geno, cov) -> np.ndarray:
    c = components(rule, geno, cov)
    h_total = rule["h_g"] + rule["h_env"] + rule["h_nl"]
    raw = np.sqrt(rule["h_g"]) * c["genetic"] + np.sqrt(rule["h_env"]) * c["env"] + np.sqrt(rule["h_nl"]) * c["nonadd"]
    return np.sqrt(h_total) * zs(raw)


def labels_from_signal(sig: np.ndarray, h_total: float, eps: np.ndarray, flip_u: np.ndarray,
                       prevalence: float, noise: float) -> np.ndarray:
    liab = sig + np.sqrt(max(0.0, 1.0 - h_total)) * eps
    n = len(liab)
    k = int(round(prevalence * n))
    thr = np.sort(liab)[n - k]
    y = (liab >= thr).astype(int)
    m = int(round(noise * n))
    if m:
        y[np.argsort(flip_u)[:m]] ^= 1
    return y


def make_labels(rule: dict, geno, cov, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Labels for one rule; also returns the oracle ceilings (AUC of the true signal / of its additive part)."""
    n = len(next(iter(cov.values())))
    sig = signal(rule, geno, cov)
    eps, flip_u = rng.standard_normal(n), rng.random(n)
    h_total = rule["h_g"] + rule["h_env"] + rule["h_nl"]
    y = labels_from_signal(sig, h_total, eps, flip_u, rule["prevalence"], rule["noise"])
    c = components(rule, geno, cov)
    additive = np.sqrt(rule["h_g"]) * c["genetic"] + np.sqrt(rule["h_env"]) * c["env"]
    ok = 0 < y.sum() < n
    info = {"prevalence_achieved": round(float(y.mean()), 4),
            "oracle_auc_full": round(float(roc_auc_score(y, sig)), 4) if ok else float("nan"),
            "oracle_auc_additive_part": round(float(roc_auc_score(y, additive)), 4) if ok else float("nan")}
    return y, info


# ------------------------------------------------------------------ drawing random rules
def _uniform(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


def _choice(rng, probs: dict):
    keys = list(probs)
    p = np.array([probs[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def draw_rule(rng: np.random.Generator, snp_cols: list[str], cov_cols: list[str], cfg: dict,
              geno: dict, cov: dict, name: str) -> dict:
    """One random rule. Redraws until every component that carries variance is non-degenerate."""
    usable = [c for c in snp_cols if np.asarray(geno[c]).std() > 1e-9 and (np.asarray(geno[c]) > 0).sum() >= 10]
    for _ in range(200):
        k = int(rng.integers(cfg["n_snps"][0], min(cfg["n_snps"][1], len(usable)) + 1))
        chosen = list(rng.choice(usable, size=k, replace=False))
        snps = [{"col": c, "flip": bool(rng.random() < 0.5), "weight": round(_uniform(rng, [0.5, 1.5]), 3),
                 "coding": _choice(rng, cfg["snp_coding_probs"])} for c in chosen]
        n_env = int(rng.integers(cfg["n_env_terms"][0], min(cfg["n_env_terms"][1], len(cov_cols)) + 1))
        env_cols = list(rng.choice(cov_cols, size=n_env, replace=False))
        env = {c: round(float(rng.choice([-1, 1]) * _uniform(rng, cfg["env_coef_abs"])), 3) for c in env_cols}

        S = _uniform(rng, cfg["signal_share"])
        f = _uniform(rng, cfg["nonadditive_fraction"])
        r = _uniform(rng, cfg["genetic_fraction_of_additive"])
        h_nl, h_g, h_env = f * S, (1 - f) * S * r, (1 - f) * S * (1 - r)

        terms = []
        if h_nl > 1e-3:
            for _t in range(int(rng.integers(cfg["n_nonadd_terms"][0], cfg["n_nonadd_terms"][1] + 1))):
                kind = _choice(rng, cfg["nonadd_type_probs"])
                if kind == "gxe":
                    terms.append({"type": "gxe", "snp": str(rng.choice(chosen)), "cov": str(rng.choice(cov_cols))})
                elif kind == "gxg" and len(chosen) >= 2:
                    a, b = rng.choice(chosen, size=2, replace=False)
                    terms.append({"type": "gxg", "a": str(a), "b": str(b)})
                elif kind == "gxg_prod" and len(chosen) >= 2:
                    a, b = rng.choice(chosen, size=2, replace=False)
                    terms.append({"type": "gxg_prod", "a": str(a), "b": str(b)})
                elif kind == "exe_prod" and len(cov_cols) >= 2:
                    a, b = rng.choice(cov_cols, size=2, replace=False)
                    terms.append({"type": "exe_prod", "a": str(a), "b": str(b)})
                elif kind == "exe_and" and len(cov_cols) >= 2:
                    a, b = rng.choice(cov_cols, size=2, replace=False)
                    terms.append({"type": "exe_and", "a": str(a), "dir_a": int(rng.choice([-1, 1])),
                                  "b": str(b), "dir_b": int(rng.choice([-1, 1]))})
                elif kind == "gthresh":
                    terms.append({"type": "gthresh", "q": round(float(rng.uniform(0.6, 0.85)), 2)})
            if not terms:
                terms = [{"type": "gxe", "snp": str(rng.choice(chosen)), "cov": str(rng.choice(cov_cols))}] if cfg.get("orthogonalise") \
                    else [{"type": "gthresh", "q": 0.7}]
        rule = {"name": name, "snps": snps, "env": env, "terms": terms, "h_g": h_g, "h_env": h_env, "h_nl": h_nl,
                "prevalence": round(_uniform(rng, cfg["prevalence"]), 3), "noise": round(_uniform(rng, cfg["label_noise"]), 3)}
        if cfg.get("orthogonalise"):
            rule["orthogonalise"] = True
        if _non_degenerate(rule, geno, cov):
            return rule
    raise RuntimeError("could not draw a non-degenerate rule; check the genotype table")


def _non_degenerate(rule: dict, geno, cov) -> bool:
    c = components(rule, geno, cov)
    if c["genetic"].std() < 1e-9 or c["env"].std() < 1e-9:
        return False
    return not (rule["h_nl"] > 1e-3 and c["nonadd"].std() < 1e-9)


def describe(rule: dict) -> str:
    snp = ",".join(f"{s['col']}{'-' if s['flip'] else '+'}{s['coding'][:3]}" for s in rule["snps"])
    nl = ";".join(t["type"] for t in rule["terms"]) or "none"
    return f"snps[{snp}] env[{','.join(rule['env'])}] nonadd[{nl}]"


# ------------------------------------------------------------------ named rules (scenarios.yaml SNPs)
def _resolve(v, trait):
    return v[trait] if isinstance(v, dict) else v


def named_rule(name: str, trait: str, snp_alleles: dict[str, str], weights: dict[str, float] | None,
               meta: dict[str, dict], env_coefficients: dict[str, float], structure: dict,
               prevalence: float, noise: float) -> dict:
    """Rule on the SNPs and risk alleles of a scenario in scenarios.yaml, with a fixed non-additive structure."""
    snps = []
    for rsid, risk in snp_alleles.items():
        m = meta[rsid]
        if risk == m["alt"]:
            flip = False
        elif risk == m["ref"]:
            flip = True
        else:
            raise ValueError(f"{rsid}: risk allele {risk} is neither ref {m['ref']} nor alt {m['alt']}")
        snps.append({"col": rsid, "flip": flip, "weight": float(weights[rsid]) if weights else 1.0, "coding": "additive"})
    terms = []
    for t in structure.get("terms", []):
        t = {k: _resolve(v, trait) for k, v in t.items()}
        terms.append(t)
    return {"name": name, "snps": snps, "env": dict(env_coefficients), "terms": terms, "h_g": structure["h_g"],
            "h_env": structure["h_env"], "h_nl": structure["h_nl"], "prevalence": prevalence, "noise": noise}


def nonadditive_fraction(rule: dict) -> float:
    tot = rule["h_g"] + rule["h_env"] + rule["h_nl"]
    return rule["h_nl"] / tot if tot else 0.0
