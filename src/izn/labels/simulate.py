"""Simulated iron / zinc risk labels from real genotypes (liability-threshold model).

NOT measured phenotypes. The SNP score only decides how much of the simulated liability is genetic;
results therefore describe how well algorithms recover an assumed rule, not real deficiency.
Pure standard library on purpose (no numpy needed).
"""
from __future__ import annotations

import math
import random


def risk_dosage(dose: int, ref: str, alt: str, risk: str) -> int:
    """Copies of the RISK allele from an alt-allele count (0/1/2)."""
    if dose < 0:
        raise ValueError("missing genotype in a label SNP")
    if risk == alt:
        return dose
    if risk == ref:
        return 2 - dose
    raise ValueError(f"risk allele {risk} is neither ref {ref} nor alt {alt}")


def score(features: dict[str, list[int]], meta: dict[str, dict], risk_alleles: dict[str, str],
          weights: dict[str, float] | None = None) -> list[float]:
    """Count of risk alleles per sample, equal weights unless `weights` (rsID -> weight) is given."""
    n = len(next(iter(features.values())))
    out = [0.0] * n
    for rsid, risk in risk_alleles.items():
        m = meta[rsid]
        for i, d in enumerate(features[rsid]):
            out[i] += (weights[rsid] if weights else 1.0) * risk_dosage(d, m["ref"], m["alt"], risk)
    return out


def zscore(x: list[float]) -> list[float]:
    n = len(x)
    mu = sum(x) / n
    sd = math.sqrt(sum((v - mu) ** 2 for v in x) / n)
    if sd == 0:
        raise ValueError("score has no variation in this sample")
    return [(v - mu) / sd for v in x]


def draws(seed: int, trait: str, n: int) -> tuple[list[float], list[float]]:
    """Fixed random numbers per trait, so every h2/prevalence setting reuses the same noise."""
    rng = random.Random(f"{seed}-{trait}")
    return [rng.gauss(0, 1) for _ in range(n)], [rng.random() for _ in range(n)]


def threshold_labels(liab: list[float], flip_u: list[float], prevalence: float, noise: float) -> list[int]:
    """Top `prevalence` fraction of liability -> 1, then flip the `noise` fraction at random."""
    k = round(prevalence * len(liab))
    thr = sorted(liab)[len(liab) - k]
    y = [1 if v >= thr else 0 for v in liab]
    if noise > 0:
        m = round(noise * len(y))
        for i in sorted(range(len(y)), key=lambda j: flip_u[j])[:m]:
            y[i] = 1 - y[i]
    return y


def make_labels(z: list[float], eps: list[float], flip_u: list[float], h2: float,
                prevalence: float, noise: float) -> list[int]:
    liab = [math.sqrt(h2) * a + math.sqrt(1 - h2) * e for a, e in zip(z, eps)]
    return threshold_labels(liab, flip_u, prevalence, noise)


def auc(s: list[float], y: list[int]) -> float:
    """Mann-Whitney AUC with average ranks for ties."""
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (sum(r for r, yy in zip(ranks, y) if yy) - n1 * (n1 + 1) / 2) / (n1 * n0)
