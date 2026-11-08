"""Synthetic non-genetic covariates and the genes + environment liability. Everything here is simulated."""
from __future__ import annotations

import math
import random

from izn.labels.simulate import zscore

COVARIATES = ["child", "age_months", "age_years", "young_child", "pregnant", "heavy_menses", "hookworm",
              "blood_loss", "malaria", "inflammation", "diarrhoea", "diet_iron", "diet_zinc"]


def generate_covariates(n: int, cfg: dict, seed: int) -> dict[str, list[float]]:
    rng = random.Random(f"cov-{seed}")
    c = cfg["cohort"]
    out: dict[str, list[float]] = {k: [] for k in COVARIATES}
    rho = c["diet_zinc_corr_with_iron"]
    for _ in range(n):
        child = rng.random() < c["p_child"]
        age_m = rng.uniform(6, 59) if child else 0.0
        age_y = 0.0 if child else rng.uniform(15, 49)
        preg = (not child) and rng.random() < c["p_pregnant_woman"]
        menses = (not child) and rng.random() < c["p_heavy_menses"]
        hook = rng.random() < c["p_hookworm"]
        role = "child" if child else "woman"
        mal = rng.random() < c["p_malaria"][role]
        infl = rng.random() < min(1.0, c["p_inflammation_base"] + (0.35 if mal else 0.0))
        diar = rng.random() < c["p_diarrhoea"][role]
        di = rng.gauss(0, 1)
        dz = rho * di + math.sqrt(1 - rho ** 2) * rng.gauss(0, 1)
        row = {"child": child, "age_months": age_m, "age_years": age_y, "young_child": child and age_m < 24,
               "pregnant": preg, "heavy_menses": menses, "hookworm": hook, "blood_loss": menses or hook,
               "malaria": mal, "inflammation": infl, "diarrhoea": diar, "diet_iron": di, "diet_zinc": dz}
        for k in COVARIATES:
            out[k].append(float(row[k]))
    return out


def env_score(cov: dict[str, list[float]], coefficients: dict[str, float]) -> list[float]:
    n = len(cov["child"])
    z = {k: zscore(cov[k]) for k in coefficients}      # standardise so coefficients are comparable
    return [sum(coefficients[k] * z[k][i] for k in coefficients) for i in range(n)]


def interaction(z_gen: list[float], inflammation: list[float]) -> list[float]:
    """Genetic effect that only acts (or acts more) when inflammation is present: zG x (infl - mean)."""
    m = sum(inflammation) / len(inflammation)
    return zscore([g * (i - m) for g, i in zip(z_gen, inflammation)])


def liability(components: list[tuple[float, list[float]]], eps: list[float]) -> list[float]:
    """components = [(variance share, standardised signal), ...]; the rest of the variance is noise."""
    used = sum(s for s, _ in components)
    if used > 1.0 + 1e-9:
        raise ValueError(f"variance shares add to {used:.2f} (> 1)")
    res = math.sqrt(max(0.0, 1.0 - used))
    return [sum(math.sqrt(s) * z[i] for s, z in components) + res * eps[i] for i in range(len(eps))]
