"""Figures for the random-teacher study (fig18) and the sealed blind challenge (fig19). Uses the style of figures.py."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from izn.reporting.figures import (BLUE, INK, INK2, MODEL_COLOR, MODEL_LABEL, MODEL_MARKER, MODEL_ORDER, ORANGE, AQUA,
                                   SIM_NOTE, _contenders, _save, _style)

GROUP_COLOR = [BLUE, ORANGE, AQUA]
FLEX = ["random_forest", "gradient_boosting", "svm_rbf", "stacking"]


def fig_regimes(groups: list[tuple[str, pd.DataFrame]], out: Path) -> Path:
    """groups = [(label, wide-table of AUC per rule x model)]. Left: mean AUC per model. Right: best flexible minus logistic regression."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.3, 1]})
    g = len(groups)
    offs = np.linspace(-0.2, 0.2, g) if g > 1 else [0.0]
    for gi, (lab, w) in enumerate(groups):
        mu, se = w.mean()[MODEL_ORDER], 1.96 * w.std()[MODEL_ORDER] / np.sqrt(len(w))
        for i, m in enumerate(MODEL_ORDER):
            a.errorbar(i + offs[gi], mu[m], yerr=se[m], fmt="none", ecolor=INK2, capsize=2.5, lw=1)
            a.scatter(i + offs[gi], mu[m], s=62, marker=MODEL_MARKER[m], zorder=3, color=GROUP_COLOR[gi], edgecolor="white", linewidth=0.8)
        a.scatter([], [], s=62, marker="o", color=GROUP_COLOR[gi], label=f"{lab} (n={len(w)})")
    a.set_xticks(range(len(MODEL_ORDER)), [MODEL_LABEL[m].replace(" ", "\n", 1) for m in MODEL_ORDER], fontsize=8)
    a.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    a.set_ylabel("mean AUC (nested CV, tuned); bars = 95% CI")
    a.set_title("Mean AUC by model and rule source")
    a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=min(g, 3))
    a.grid(axis="x", visible=False)
    rng = np.random.default_rng(0)
    for k, (lab, w) in enumerate(groups):
        adv = (w[FLEX].max(axis=1) - w["logreg"]).to_numpy()
        b.scatter(k + rng.uniform(-0.12, 0.12, len(adv)), adv, s=24, color=GROUP_COLOR[k], alpha=0.7, edgecolor="white", linewidth=0.4)
        b.hlines(adv.mean(), k - 0.28, k + 0.28, color=INK, lw=2.5)
        b.text(k + 0.3, adv.mean(), f"{adv.mean():+.3f}", va="center", fontsize=8)
    b.axhline(0, color=INK2, lw=1, ls="--")
    b.set_xticks(range(g), [lab.replace(" ", "\n", 1) for lab, _ in groups], fontsize=8)
    b.set_xlim(-0.6, g - 0.1 + 0.4)
    b.set_ylabel("best flexible model minus logistic regression (AUC)")
    b.set_title("Above zero: a flexible model beat logistic regression")
    b.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig18_rule_sources.png", "Teacher rules are random networks / random trees (nobody chose their shape). " + SIM_NOTE)


def fig_teacher_nonadditivity(rules: pd.DataFrame, out: Path) -> Path:
    """How additive were the random teachers? (measured, not chosen)"""
    _style()
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for fam, col in (("mlp", BLUE), ("trees", ORANGE)):
        s = rules[rules.family == fam].nonadditive_fraction
        ax.hist(s, bins=np.linspace(0, 1, 11), alpha=0.75, color=col, edgecolor="white", label=f"{fam} teachers (n={len(s)})")
    ax.set_xlabel("non-additive share of the teacher output (1 - linear R$^2$)")
    ax.set_ylabel("number of teachers")
    ax.set_title("How additive were the random teachers?")
    ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig20_teacher_nonadditivity.png", SIM_NOTE)


def fig_blind_reveal(tab: pd.DataFrame, out: Path) -> Path:
    """Left: test AUC of each model per hidden rule, oracle as a dash, pipeline's pre-registered choice ringed. Right: regret per strategy."""
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.8, 1]})
    t = tab.sort_values("oracle_auc_test").reset_index(drop=True)
    for i, r in t.iterrows():
        for m in MODEL_ORDER:
            a.scatter(i, r[f"test_{m}"], s=44, marker=MODEL_MARKER[m], color=MODEL_COLOR[m], zorder=3, alpha=0.9,
                      edgecolor="white", linewidth=0.5)
        a.hlines(r.oracle_auc_test, i - 0.32, i + 0.32, color=INK, lw=2)
        a.scatter(i, r[f"test_{r.chosen}"], s=170, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=4)
    for m in MODEL_ORDER:
        a.scatter([], [], s=44, marker=MODEL_MARKER[m], color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    a.plot([], [], color=INK, lw=2, label="oracle (true rule)")
    a.scatter([], [], s=120, facecolor="none", edgecolor=INK, linewidth=1.6, label="pipeline's pre-registered choice")
    a.set_xticks(range(len(t)), [f"{c[-2:]}\n{ {'generator': 'hand-built'}.get(f, f)}" for c, f in zip(t.challenge, t.family)], fontsize=7.5)
    a.set_xlabel("hidden rule (sorted by oracle AUC; label = challenge number, rule family)")
    a.set_ylabel("AUC on the locked test split")
    a.set_title("Each hidden rule: every model, the ceiling, and the choice made blind")
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    a.grid(axis="x", visible=False)
    strat = {"chosen blind": t.best_auc - t.chosen_auc, "always\nlogreg": t.best_auc - t.logreg_auc, "always\nstacking": t.best_auc - t.stacking_auc}
    rng = np.random.default_rng(1)
    for k, (lab, v) in enumerate(strat.items()):
        b.scatter(k + rng.uniform(-0.1, 0.1, len(v)), v, s=26, color=[BLUE, ORANGE, AQUA][k], alpha=0.75, edgecolor="white", linewidth=0.4)
        b.hlines(v.mean(), k - 0.27, k + 0.27, color=INK, lw=2.5)
        b.text(k + 0.3, v.mean(), f"{v.mean():.3f}", va="center", fontsize=8)
    b.set_xticks(range(3), list(strat), fontsize=8)
    b.set_xlim(-0.6, 3.0)
    b.set_ylim(bottom=-0.005)
    b.set_ylabel("regret = best model's AUC minus this strategy's (lower is better)")
    b.set_title("Cost of each strategy")
    b.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig19_blind_challenge.png",
                 "Hidden rules drawn with an operating-system seed, sealed before analysis; test split locked until the choice was written. " + SIM_NOTE)
