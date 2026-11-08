"""Random-teacher study and the sealed blind challenge.

Teacher study: many random-teacher rules (random neural networks / random trees) pushed through the same nested-CV
comparison as the main study.

Blind challenge (a commit-reveal protocol):
  seal     draws K hidden rules from a pool that mixes random teachers and the hand-built rule generator, using
           a seed from the operating system (nobody picks or sees it). Only the LABELS are written out. The rules go
           into a sealed file whose SHA-256 is saved as a commitment.
  analyze  sees labels, SNPs and covariates only. Chooses a model per challenge by nested CV on a training split,
           writes that choice (with its hash) BEFORE touching the locked test split, then scores every model on it.
  reveal   refuses to run until the choice file exists, checks the sealed file against the commitment, then compares
           the pipeline's choice with the hidden rule: oracle AUC, regret versus the best model in hindsight, and versus
           the fixed strategies "always logistic regression" / "always stacking".
This is procedural blinding: it shows the pipeline can pick a good model without knowing the rule, which is the
situation real data would put it in. Labels are SIMULATED; nothing here measures real iron or zinc deficiency.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from izn.evaluation.cv import scores
from izn.evaluation.nested import make_specs, nested_evaluate
from izn.evaluation.study_data import design
from izn.labels import random_rules as RR
from izn.labels.teacher_rules import (any_signal, describe_teacher, draw_teacher, make_teacher_labels)

CONTENDERS = ["logreg@tuned", "random_forest@tuned", "gradient_boosting@tuned", "svm_rbf@tuned", "stacking@default"]
SEALED_NAME = "SEALED_rules_do_not_open.json"


# ------------------------------------------------------------------ teacher study (same output shape as rr_study.run_rule)
def run_teacher_rule(i: int, data: dict, cfg: dict, grids: dict, seed: int, prefix: str = "teacher") -> tuple[dict, list[dict]]:
    t0 = time.time()
    rng = np.random.default_rng([seed, i])
    rule = draw_teacher(rng, data["features"], data["cov_cols"], cfg["teacher_rules"], data["geno"], data["cov"], f"{prefix}_{i:03d}")
    y, info = make_teacher_labels(rule, data["geno"], data["cov"], rng)
    cv = cfg["cv"]
    rows: list[dict] = []

    def add(fs, res):
        for spec, r in res.items():
            algo, tuning = spec.split("@")
            rows.append({"rule_id": rule["name"], "feature_set": fs, "algorithm": algo, "tuning": tuning,
                         "auc": r["auc"], "auc_sd": r["auc_sd"], "ap": r["ap"]})

    specs = make_specs(grids, seed)
    add("snps_plus_env", nested_evaluate(design(data, "snps_plus_env"), y, specs, cv["outer_splits"], cv["inner_splits"], seed))
    lr = {"logreg@tuned": specs["logreg@tuned"]}
    for fs in ("env_only", "snps_only"):
        add(fs, nested_evaluate(design(data, fs), y, lr, cv["outer_splits"], cv["inner_splits"], seed))
    n_snps = sum(c in data["geno"] for c in rule["inputs"])
    row = {"rule_id": rule["name"], "family": rule["family"], "signal_share": rule["h_total"],
           "nonadditive_fraction": info["nonadditive_fraction"], "n_snps": n_snps, "n_env_terms": len(rule["inputs"]) - n_snps,
           "prevalence": rule["prevalence"], "label_noise": rule["noise"], **info,
           "description": describe_teacher(rule), "seconds": round(time.time() - t0, 1)}
    return row, rows


# ------------------------------------------------------------------ blind challenge
def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serialisable: {type(o)}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _split(y: np.ndarray, test_frac: float, seed: int, j: int) -> tuple[np.ndarray, np.ndarray]:
    return train_test_split(np.arange(len(y)), test_size=test_frac, stratify=y, random_state=seed + 17 * j)


def seal(data: dict, cfg: dict, k: int, out_dir: Path, entropy: int | None = None) -> str:
    """Draw k hidden rules, write the labels, seal the rules and return the commitment hash."""
    out_dir.mkdir(parents=True, exist_ok=True)
    entropy = entropy if entropy is not None else int.from_bytes(os.urandom(16), "big")
    geno, cov = data["geno"], data["cov"]
    labels: dict = {"sample": data["sample"]}
    sealed = {"entropy": entropy, "challenges": []}
    for j, child in enumerate(np.random.SeedSequence(entropy).spawn(k)):
        rng = np.random.default_rng(child)
        fam = RR._choice(rng, cfg["blind"]["family_probs"])
        name = f"blind_{j:02d}"
        if fam == "generator":
            rule = RR.draw_rule(rng, data["features"], data["cov_cols"], cfg["random_rules"], geno, cov, name)
            y, info = RR.make_labels(rule, geno, cov, rng)
            nf = RR.nonadditive_fraction(rule)
        else:
            rule = draw_teacher(rng, data["features"], data["cov_cols"], cfg["teacher_rules"], geno, cov, name, family=fam)
            y, info = make_teacher_labels(rule, geno, cov, rng)
            nf = info["nonadditive_fraction"]
        labels[name] = y
        sealed["challenges"].append({"name": name, "family": fam, "nonadditive_fraction": float(nf), "info": info, "rule": rule})
    pd.DataFrame(labels).to_csv(out_dir / "labels_blind.tsv", sep="\t", index=False)
    sp = out_dir / SEALED_NAME
    sp.write_text(json.dumps(sealed, sort_keys=True, default=_jsonable), encoding="utf-8")
    digest = _sha256(sp)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    (out_dir / "commitment.txt").write_text(f"sha256 {digest}\nsealed {stamp}\nchallenges {k}\n", encoding="utf-8")
    return digest


def _contender_specs(grids: dict, seed: int) -> dict:
    return {k: v for k, v in make_specs(grids, seed).items() if k in CONTENDERS}


def _cv_phase(j, X, y, grids, cv, test_frac, seed):
    tr, _ = _split(y, test_frac, seed, j)
    res = nested_evaluate(X[tr], y[tr], _contender_specs(grids, seed), cv["outer_splits"], cv["inner_splits"], seed)
    return j, {m: r["auc"] for m, r in res.items()}


def _test_phase(j, X, y, grids, cv, test_frac, seed):
    tr, te = _split(y, test_frac, seed, j)
    out = {}
    for name, (est, grid) in _contender_specs(grids, seed).items():
        if grid:
            m = GridSearchCV(clone(est), grid, scoring="roc_auc", n_jobs=1,
                             cv=StratifiedKFold(cv["inner_splits"], shuffle=True, random_state=seed)).fit(X[tr], y[tr])
        else:
            m = clone(est).fit(X[tr], y[tr])
        out[name] = float(roc_auc_score(y[te], scores(m, X[te])))
    return j, out


def analyze(data: dict, cfg: dict, grids: dict, out_dir: Path, test_frac: float, seed: int, jobs: int = -1) -> pd.DataFrame:
    """Uses labels, SNPs and covariates only. Never reads the sealed file."""
    lab = pd.read_csv(out_dir / "labels_blind.tsv", sep="\t")
    if lab["sample"].tolist() != list(data["sample"]):
        raise SystemExit("labels_blind.tsv rows do not match the features table; was it sealed on a different table?")
    names = [c for c in lab.columns if c != "sample"]
    X, cv = design(data, "snps_plus_env"), cfg["cv"]
    ys = [lab[c].to_numpy(int) for c in names]
    cvres = dict(Parallel(n_jobs=jobs)(delayed(_cv_phase)(j, X, ys[j], grids, cv, test_frac, seed) for j in range(len(names))))
    choice = pd.DataFrame([{"challenge": names[j], "chosen": max(CONTENDERS, key=lambda m: cvres[j][m]),
                            **{f"cv_{m.split('@')[0]}": round(cvres[j][m], 4) for m in CONTENDERS}} for j in range(len(names))])
    cp = out_dir / "preregistered_choice.tsv"
    choice.to_csv(cp, sep="\t", index=False)                              # written BEFORE any test-split score exists
    digest = _sha256(cp)
    (out_dir / "preregistered_choice.sha256").write_text(f"sha256 {digest}\nwritten {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S UTC}\n")
    testres = dict(Parallel(n_jobs=jobs)(delayed(_test_phase)(j, X, ys[j], grids, cv, test_frac, seed) for j in range(len(names))))
    test = pd.DataFrame([{"challenge": names[j], **{f"test_{m.split('@')[0]}": round(testres[j][m], 4) for m in CONTENDERS}}
                         for j in range(len(names))])
    test.to_csv(out_dir / "blind_test_scores.tsv", sep="\t", index=False)
    return choice.merge(test, on="challenge")


def reveal(data: dict, out_dir: Path, test_frac: float, seed: int, force: bool = False) -> tuple[pd.DataFrame, dict]:
    cp, sp, cm = out_dir / "preregistered_choice.tsv", out_dir / SEALED_NAME, out_dir / "commitment.txt"
    if not (cp.exists() and (out_dir / "blind_test_scores.tsv").exists()) and not force:
        raise SystemExit("run `analyze` first: the choice must be recorded before the rules are revealed")
    committed = cm.read_text(encoding="utf-8").split()[1]
    if _sha256(sp) != committed:
        raise SystemExit("the sealed file does not match the commitment: it was changed after sealing")
    sealed = json.loads(sp.read_text(encoding="utf-8"))
    choice = pd.read_csv(cp, sep="\t")
    test = pd.read_csv(out_dir / "blind_test_scores.tsv", sep="\t")
    lab = pd.read_csv(out_dir / "labels_blind.tsv", sep="\t")
    models = [m.split("@")[0] for m in CONTENDERS]
    rows = []
    for j, ch in enumerate(sealed["challenges"]):
        y = lab[ch["name"]].to_numpy(int)
        _, te = _split(y, test_frac, seed, j)
        oracle = float(roc_auc_score(y[te], any_signal(ch["rule"], data["geno"], data["cov"])[te]))
        t = test[test.challenge == ch["name"]].iloc[0]
        c = choice[choice.challenge == ch["name"]].iloc[0]
        tauc = {m: float(t[f"test_{m}"]) for m in models}
        chosen = c["chosen"].split("@")[0]
        best = max(tauc, key=tauc.get)
        rows.append({"challenge": ch["name"], "family": ch["family"], "nonadditive_fraction": round(ch["nonadditive_fraction"], 3),
                     "oracle_auc_test": round(oracle, 4), "chosen": chosen, "chosen_auc": round(tauc[chosen], 4),
                     "best_in_hindsight": best, "best_auc": round(tauc[best], 4), "logreg_auc": round(tauc["logreg"], 4),
                     "stacking_auc": round(tauc["stacking"], 4), **{f"test_{m}": round(v, 4) for m, v in tauc.items()}})
    tab = pd.DataFrame(rows)
    reg = {"chosen": tab.best_auc - tab.chosen_auc, "always_logreg": tab.best_auc - tab.logreg_auc,
           "always_stacking": tab.best_auc - tab.stacking_auc}
    summ = {"n": len(tab), "mean_oracle": float(tab.oracle_auc_test.mean()), "mean_gap_chosen_to_oracle": float((tab.oracle_auc_test - tab.chosen_auc).mean()),
            "regret": {k: (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")) for k, v in reg.items()},
            "chosen_best_or_within_0.01": float(((tab.best_auc - tab.chosen_auc) <= 0.01).mean()),
            "chosen_counts": tab.chosen.value_counts().to_dict(), "families": tab.family.value_counts().to_dict()}
    tab.to_csv(out_dir / "reveal.tsv", sep="\t", index=False)
    return tab, summ
