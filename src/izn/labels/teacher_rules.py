"""Random-teacher rules: the labelling rule is a randomly initialised model, not a hand-written formula.

A teacher is either a small random neural network (family "mlp") or an ensemble of random decision trees
(family "trees"). It reads a random subset of SNP dosages and covariates; nobody chooses its shape, so it does not
favour any learning algorithm by construction. Its output becomes the SIGNAL of a liability-threshold model
(same labelling step as the other rule families: signal + noise -> top `prevalence` fraction -> a `noise` fraction
flipped). Everything is SIMULATED; the teacher says nothing about real iron or zinc biology.

How additive a teacher turns out to be is measured afterwards (`linear_r2`: share of its output a straight-line
model on the same inputs can explain), not chosen.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from izn.labels import random_rules as RR
from izn.labels.random_rules import labels_from_signal, zs

ACT = {"tanh": np.tanh, "relu": lambda x: np.maximum(x, 0.0)}


def _col(name: str, geno: dict, cov: dict) -> np.ndarray:
    return np.asarray(geno[name] if name in geno else cov[name], dtype=float)


def inputs_matrix(rule: dict, geno: dict, cov: dict) -> np.ndarray:
    return np.column_stack([zs(_col(n, geno, cov)) for n in rule["inputs"]])


def _mlp_out(p: dict, X: np.ndarray) -> np.ndarray:
    h = X
    for W, b in zip(p["W"], p["b"]):
        h = ACT[p["act"]](h @ np.asarray(W) + np.asarray(b))
    return h @ np.asarray(p["w_out"])


def _tree_out(p: dict, X: np.ndarray) -> np.ndarray:
    n, d = X.shape[0], p["depth"]
    feat, thr, leaf = np.asarray(p["feat"]), np.asarray(p["thr"]), np.asarray(p["leaf"])
    idx = np.zeros(n, dtype=int)
    for _ in range(d):
        go_right = X[np.arange(n), feat[idx]] > thr[idx]
        idx = 2 * idx + 1 + go_right.astype(int)
    return leaf[idx - (2 ** d - 1)]


def teacher_raw(rule: dict, geno: dict, cov: dict) -> np.ndarray:
    X = inputs_matrix(rule, geno, cov)
    if rule["family"] == "mlp":
        return _mlp_out(rule["mlp"], X)
    return sum(_tree_out(t, X) for t in rule["trees"])


def teacher_signal(rule: dict, geno: dict, cov: dict) -> np.ndarray:
    return np.sqrt(rule["h_total"]) * zs(teacher_raw(rule, geno, cov))


def any_signal(rule: dict, geno: dict, cov: dict) -> np.ndarray:
    """Signal of either rule kind (teacher, or a hand-built generator rule from random_rules)."""
    return teacher_signal(rule, geno, cov) if rule.get("kind") == "teacher" else RR.signal(rule, geno, cov)


def linear_r2(rule: dict, geno: dict, cov: dict) -> float:
    """Adjusted R^2 of a straight-line fit of the teacher output on its own inputs (1 = fully additive)."""
    X = inputs_matrix(rule, geno, cov)
    y = zs(teacher_raw(rule, geno, cov))
    n, p = X.shape
    A = np.column_stack([np.ones(n), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = 1.0 - float(((y - A @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())
    return float(np.clip(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1), 0.0, 1.0))


# ------------------------------------------------------------------ drawing random teachers
def _draw_mlp(rng: np.random.Generator, n_in: int, cfg: dict) -> dict:
    n_layers = int(rng.integers(cfg["hidden_layers"][0], cfg["hidden_layers"][1] + 1))
    gain = RR._uniform(rng, cfg["gain"])
    act = str(rng.choice(cfg["activations"]))
    Ws, bs, fan = [], [], n_in
    for _ in range(n_layers):
        width = int(rng.integers(cfg["width"][0], cfg["width"][1] + 1))
        Ws.append((rng.standard_normal((fan, width)) * gain / np.sqrt(fan)).round(4).tolist())
        bs.append((rng.standard_normal(width) * 0.5).round(4).tolist())
        fan = width
    return {"act": act, "W": Ws, "b": bs, "w_out": (rng.standard_normal(fan) / np.sqrt(fan)).round(4).tolist()}


def _draw_trees(rng: np.random.Generator, X: np.ndarray, cfg: dict) -> list[dict]:
    trees = []
    for _ in range(int(rng.integers(cfg["n_trees"][0], cfg["n_trees"][1] + 1))):
        d = int(rng.integers(cfg["depth"][0], cfg["depth"][1] + 1))
        feat = rng.integers(0, X.shape[1], 2 ** d - 1)
        q = rng.uniform(0.25, 0.75, 2 ** d - 1)
        thr = [round(float(np.quantile(X[:, f], qq)), 4) for f, qq in zip(feat, q)]
        trees.append({"depth": d, "feat": feat.tolist(), "thr": thr, "leaf": rng.standard_normal(2 ** d).round(4).tolist()})
    return trees


def draw_teacher(rng: np.random.Generator, snp_cols: list[str], cov_cols: list[str], cfg: dict, geno: dict, cov: dict,
                 name: str, family: str | None = None) -> dict:
    """One random teacher. Redraws until its output is non-constant."""
    usable = [c for c in snp_cols if np.asarray(geno[c]).std() > 1e-9 and (np.asarray(geno[c]) > 0).sum() >= 10]
    covs = [c for c in cov_cols if np.asarray(cov[c]).std() > 1e-9]
    for _ in range(200):
        fam = family or RR._choice(rng, cfg["family_probs"])
        k = int(rng.integers(cfg["n_snps"][0], min(cfg["n_snps"][1], len(usable)) + 1))
        m = int(rng.integers(cfg["n_env_terms"][0], min(cfg["n_env_terms"][1], len(covs)) + 1))
        inputs = [str(c) for c in rng.choice(usable, size=k, replace=False)] + [str(c) for c in rng.choice(covs, size=m, replace=False)]
        rule = {"name": name, "kind": "teacher", "family": fam, "inputs": inputs,
                "h_total": round(RR._uniform(rng, cfg["signal_share"]), 3),
                "prevalence": round(RR._uniform(rng, cfg["prevalence"]), 3),
                "noise": round(RR._uniform(rng, cfg["label_noise"]), 3)}
        X = inputs_matrix(rule, geno, cov)
        if fam == "mlp":
            rule["mlp"] = _draw_mlp(rng, X.shape[1], cfg["mlp"])
        else:
            rule["trees"] = _draw_trees(rng, X, cfg["trees"])
        raw = teacher_raw(rule, geno, cov)
        if np.isfinite(raw).all() and raw.std() > 1e-6 and len(np.unique(np.round(raw, 6))) > 10:
            return rule
    raise RuntimeError("could not draw a non-degenerate teacher; check the genotype table")


def make_teacher_labels(rule: dict, geno: dict, cov: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    n = len(next(iter(cov.values())))
    sig = teacher_signal(rule, geno, cov)
    eps, flip_u = rng.standard_normal(n), rng.random(n)
    y = labels_from_signal(sig, rule["h_total"], eps, flip_u, rule["prevalence"], rule["noise"])
    ok = 0 < y.sum() < n
    r2 = linear_r2(rule, geno, cov)
    return y, {"prevalence_achieved": round(float(y.mean()), 4),
               "oracle_auc_full": round(float(roc_auc_score(y, sig)), 4) if ok else float("nan"),
               "linear_r2": round(r2, 4), "nonadditive_fraction": round(1.0 - r2, 4)}


def describe_teacher(rule: dict) -> str:
    return f"{rule['family']} teacher on {len(rule['inputs'])} inputs [{','.join(rule['inputs'])}]"
