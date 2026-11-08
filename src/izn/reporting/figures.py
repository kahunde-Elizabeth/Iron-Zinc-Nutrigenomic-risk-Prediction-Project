"""Figures for the prototype report. Palette: validated categorical order (blue, orange, aqua, yellow, magenta);
identity is never colour-alone (markers, labels, value annotations). Sequential = one hue; correlation = diverging
orange-grey-blue. Every figure that shows simulated labels says so in a footnote."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
MODEL_ORDER = ["logreg", "random_forest", "gradient_boosting", "svm_rbf", "stacking"]
MODEL_COLOR = dict(zip(MODEL_ORDER, [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]))
MODEL_MARKER = dict(zip(MODEL_ORDER, ["o", "s", "^", "D", "v"]))
MODEL_LABEL = {"logreg": "Logistic regression", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
               "svm_rbf": "SVM (RBF)", "stacking": "Stacking ensemble"}
POP_COLOR = {"YRI": BLUE, "LWK": ORANGE, "GWD": AQUA, "MSL": VIOLET, "ESN": MAGENTA}
POP_MARKER = {"YRI": "o", "LWK": "s", "GWD": "^", "MSL": "D", "ESN": "v"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#a9c8f0", "#2a78d6", "#12386b"])
DIV = LinearSegmentedColormap.from_list("div", [ORANGE, "#f3f2ef", BLUE])
SIM_NOTE = "Genotypes: 1000 Genomes P1 (real). Labels, covariates and rules: SIMULATED; results describe assumed rules, not real deficiency."


def _style() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                         "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def _save(fig, out: Path, name: str, note: str | None = SIM_NOTE) -> Path:
    if note:                                    # place the footnote just below whatever is lowest (tick labels, axis labels)
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        fig.text(0.01, bb.y0 / fig.get_figheight() - 0.012, note, fontsize=7, color=INK2, ha="left", va="top")
    fig.savefig(out / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out / name


def _heat(ax, M, xl, yl, cmap, vmin, vmax, annotate=True, fmt="{:.2f}", fs=7, mask_upper=False):
    M = np.array(M, dtype=float)
    if mask_upper:
        M = np.where(np.triu(np.ones_like(M, dtype=bool), 1), np.nan, M)
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xl)), xl, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(yl)), yl, fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        mid = (vmin + vmax) / 2
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    dark = (M[i, j] - mid) / (vmax - vmin) > 0.28 if cmap is SEQ else abs(M[i, j]) > 0.55
                    ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=fs, color="white" if dark else INK)
    return im


# ------------------------------------------------------------------ descriptive figures
def fig_snp_ld(geno: pd.DataFrame, out: Path) -> Path:
    _style()
    r2 = np.corrcoef(geno.to_numpy(float).T) ** 2
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = _heat(ax, r2, geno.columns, geno.columns, SEQ, 0, 1, fs=6.5)
    fig.colorbar(im, ax=ax, label="r² (genotype correlation squared)", shrink=0.8)
    ax.set_title("Linkage disequilibrium between panel SNPs (P1, 504 people)")
    return _save(fig, out, "fig03_snp_ld.png", "Real genotypes. r² near 1 = the two SNPs carry the same signal.")


def fig_allele_freq(geno: pd.DataFrame, pops: np.ndarray, out: Path) -> Path:
    _style()
    order = [p for p in ["YRI", "ESN", "MSL", "GWD", "LWK"] if p in set(pops)] + sorted(set(pops) - set("YRI ESN MSL GWD LWK".split()))
    F = np.array([geno[pops == p].mean(axis=0) / 2 for p in order])
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    im = _heat(ax, F, geno.columns, [f"{p} (n={int((pops == p).sum())})" for p in order], SEQ, 0, max(0.5, F.max()), fs=7)
    fig.colorbar(im, ax=ax, label="alt-allele frequency", shrink=0.9)
    ax.set_title("Alt-allele frequency of each panel SNP, by population")
    return _save(fig, out, "fig04_allele_freq_by_pop.png", "Real genotypes. Frequency of the VCF alt allele (not necessarily the risk allele).")


def fig_pca(geno: pd.DataFrame, pops: np.ndarray, out: Path) -> Path:
    _style()
    Z = StandardScaler().fit_transform(geno.to_numpy(float))
    pca = PCA(n_components=2, random_state=0).fit(Z)
    P = pca.transform(Z)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for p in POP_COLOR:
        m = pops == p
        if m.any():
            ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.75, color=POP_COLOR[p], marker=POP_MARKER[p],
                       edgecolor="white", linewidth=0.4, label=f"{p} (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%} of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%} of variance)")
    ax.set_title("Unsupervised check: structure of the 13-SNP genotypes")
    ax.legend(title="Population", fontsize=8, title_fontsize=8, loc="best")
    return _save(fig, out, "fig05_pca_population.png",
                 "Real genotypes. Only 13 SNPs, so ancestry is only weakly resolved; this is a structure check, not a risk model.")


# ------------------------------------------------------------------ reference run (correlation, confusion, ROC)
def _ref_parts(ref: pd.DataFrame, features: list[str]):
    oof = {c.replace("oof__", "").split("@")[0]: ref[c].to_numpy() for c in ref.columns if c.startswith("oof__")}
    cols = [c for c in ref.columns if c not in ("sample", "pop", "y") and not c.startswith("oof__")]
    snps = [c for c in cols if c in features]
    return oof, snps, [c for c in cols if c not in snps]


def fig_correlation(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str) -> Path:
    _style()
    oof, snps, covs = _ref_parts(ref, features)
    cols = snps + covs
    M = pd.concat([ref[cols], ref["y"].rename("LABEL (deficient)")], axis=1)
    C = M.corr().to_numpy()
    names = list(M.columns)
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    im = _heat(ax, C, names, names, DIV, -1, 1, fmt="{:.1f}", fs=5.5, mask_upper=True)
    ax.axhline(len(snps) - 0.5, color=INK2, lw=0.8)
    ax.axvline(len(snps) - 0.5, color=INK2, lw=0.8)
    ax.axhline(len(cols) - 0.5, color=INK2, lw=0.8)
    ax.axvline(len(cols) - 0.5, color=INK2, lw=0.8)
    fig.colorbar(im, ax=ax, label="Pearson correlation", shrink=0.7)
    ax.set_title(f"Correlation between genotypes, synthetic covariates and the simulated label ({ref_name})")
    ax.text(len(snps) / 2 - 0.5, -1.9, "SNPs (real)", ha="center", fontsize=8, color=INK2)
    ax.text(len(snps) + len(covs) / 2 - 0.5, -1.9, "covariates (simulated)", ha="center", fontsize=8, color=INK2)
    return _save(fig, out, "fig01_correlation_matrix.png")


def fig_label_correlation(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str) -> Path:
    _style()
    _, snps, covs = _ref_parts(ref, features)
    r = pd.Series({c: np.corrcoef(ref[c], ref["y"])[0, 1] for c in snps + covs}).dropna()
    r = r.reindex(r.abs().sort_values().index)
    fig, ax = plt.subplots(figsize=(6.6, 0.28 * len(r) + 1.4))
    colors = [BLUE if c in snps else ORANGE for c in r.index]
    ax.barh(r.index, r.values, color=colors, height=0.66)
    for y, v in enumerate(r.values):
        ax.text(v + (0.008 if v >= 0 else -0.008), y, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Correlation with the simulated label (1 = deficient)")
    ax.set_title(f"Which variables move with the label? ({ref_name})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(r.min() - 0.09, r.max() + 0.09)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="SNP genotype (real)"), Patch(color=ORANGE, label="covariate (simulated)")],
              loc="lower right", fontsize=8)
    return _save(fig, out, "fig02_label_correlations.png")


def fig_confusion(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str) -> Path:
    _style()
    oof, _, _ = _ref_parts(ref, features)
    y = ref["y"].to_numpy().astype(int)
    k = int(y.sum())
    models = [m for m in MODEL_ORDER if m in oof]
    fig, axes = plt.subplots(1, len(models), figsize=(3.0 * len(models), 3.5))
    for ax, m in zip(np.atleast_1d(axes), models):
        s = oof[m]
        thr = np.sort(s)[::-1][k - 1]
        pred = (s >= thr).astype(int)
        cm = np.array([[((y == 0) & (pred == 0)).sum(), ((y == 0) & (pred == 1)).sum()],
                       [((y == 1) & (pred == 0)).sum(), ((y == 1) & (pred == 1)).sum()]])
        rate = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rate, cmap=SEQ, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({rate[i, j]:.0%})", ha="center", va="center", fontsize=9,
                        color="white" if rate[i, j] > 0.55 else INK)
        sens, spec = rate[1, 1], rate[0, 0]
        ax.set_xticks([0, 1], ["not flagged", "flagged"])
        ax.set_yticks([0, 1], ["not deficient", "deficient"])
        ax.set_xlabel("Model flags")
        ax.set_title(f"{MODEL_LABEL[m]}\nsens {sens:.0%} · spec {spec:.0%}", fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Simulated truth")
    fig.suptitle(f"Confusion matrices, out-of-fold ({ref_name}); the top {k / len(y):.0%} of scores are flagged", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, out, "fig06_confusion_matrices.png",
                 "Threshold = flag as many people as there are simulated cases (so flagged count = case count). " + SIM_NOTE)


def fig_roc_pr(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str) -> Path:
    _style()
    oof, _, _ = _ref_parts(ref, features)
    y = ref["y"].to_numpy().astype(int)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    for m in [m for m in MODEL_ORDER if m in oof]:
        fpr, tpr, _ = roc_curve(y, oof[m])
        a.plot(fpr, tpr, color=MODEL_COLOR[m], lw=2, label=f"{MODEL_LABEL[m]} (AUC {sk_auc(fpr, tpr):.3f})")
        p, r, _ = precision_recall_curve(y, oof[m])
        b.plot(r, p, color=MODEL_COLOR[m], lw=2, label=MODEL_LABEL[m])
    a.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    b.axhline(y.mean(), color=INK2, lw=1, ls="--")
    b.text(0.99, y.mean() + 0.01, f"chance = prevalence {y.mean():.0%}", ha="right", fontsize=7, color=INK2)
    a.set_xlabel("False-positive rate")
    a.set_ylabel("True-positive rate")
    a.set_title("ROC curves (out-of-fold)")
    a.legend(fontsize=7.5, loc="lower right")
    b.set_xlabel("Recall")
    b.set_ylabel("Precision")
    b.set_title("Precision-recall curves")
    b.set_ylim(0, 1.02)
    fig.suptitle(ref_name, x=0.01, ha="left", fontsize=8, color=INK2)
    return _save(fig, out, "fig07_roc_pr.png")


def fig_importance(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str, seed: int = 42) -> Path:
    _style()
    _, snps, covs = _ref_parts(ref, features)
    cols = snps + covs
    X, y = ref[cols].to_numpy(float), ref["y"].to_numpy().astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1).fit(Xtr, ytr)
    coef = pd.Series(lr[-1].coef_[0], index=cols).sort_values()
    pi = permutation_importance(rf, Xte, yte, scoring="roc_auc", n_repeats=15, random_state=seed)
    imp = pd.Series(pi.importances_mean, index=cols).sort_values()
    h = 0.27 * len(cols) + 1.4
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, h))
    a.barh(coef.index, coef.values, color=[BLUE if c in snps else ORANGE for c in coef.index], height=0.66)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_title("Logistic regression: standardised coefficients")
    a.set_xlabel("effect on log-odds of 'deficient' per SD")
    b.barh(imp.index, imp.values, color=[BLUE if c in snps else ORANGE for c in imp.index], height=0.66)
    b.axvline(0, color=INK2, lw=0.8)
    b.set_title("Random forest: permutation importance")
    b.set_xlabel("drop in AUC when the variable is shuffled (30% held-out)")
    for ax in (a, b):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
    from matplotlib.patches import Patch
    b.legend(handles=[Patch(color=BLUE, label="SNP (real)"), Patch(color=ORANGE, label="covariate (simulated)")], loc="lower right", fontsize=8)
    fig.suptitle(ref_name, x=0.01, ha="left", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out, "fig08_feature_importance.png")


# ------------------------------------------------------------------ randomized-rule study
CONT = ["logreg@tuned", "random_forest@tuned", "gradient_boosting@tuned", "svm_rbf@tuned", "stacking@default"]


def _contenders(res: pd.DataFrame) -> pd.DataFrame:
    m = res[res.feature_set == "snps_plus_env"].copy()
    m["model"] = m.algorithm + "@" + m.tuning
    w = m[m.model.isin(CONT)].pivot(index="rule_id", columns="model", values="auc")[CONT]
    w.columns = [c.split("@")[0] for c in w.columns]
    return w


def fig_random_summary(rules: pd.DataFrame, res: pd.DataFrame, out: Path) -> Path:
    _style()
    w = _contenders(res)
    orc = rules.set_index("rule_id").oracle_auc_full.reindex(w.index)
    best = w.max(axis=1)
    win, near = w.eq(best, axis=0).mean(), (w.sub(best, axis=0) >= -0.01).mean()
    gap = -(w.sub(orc, axis=0))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    x = np.arange(len(MODEL_ORDER))
    a.bar(x - 0.2, win[MODEL_ORDER], 0.38, color=[MODEL_COLOR[m] for m in MODEL_ORDER], label="outright winner")
    a.bar(x + 0.2, near[MODEL_ORDER], 0.38, color=[MODEL_COLOR[m] for m in MODEL_ORDER], alpha=0.45, label="within 0.01 AUC of the best")
    for i, m in enumerate(MODEL_ORDER):
        a.text(i - 0.2, win[m] + 0.01, f"{win[m]:.0%}", ha="center", fontsize=7.5)
        a.text(i + 0.2, near[m] + 0.01, f"{near[m]:.0%}", ha="center", fontsize=7.5)
    a.set_xticks(x, [MODEL_LABEL[m].replace(" ", "\n", 1) for m in MODEL_ORDER], fontsize=8)
    a.set_ylim(0, min(1.2, max(near.max(), win.max()) + 0.3))
    a.set_ylabel("share of random rules")
    a.set_title(f"How often each model is best ({len(w)} random rules)")
    from matplotlib.patches import Patch
    a.legend(handles=[Patch(facecolor=INK2, label="outright winner"), Patch(facecolor=INK2, alpha=0.4, label="within 0.01 AUC of the best")],
             fontsize=7.5, loc="upper center", ncol=2)
    a.grid(axis="x", visible=False)
    mu, se = gap.mean()[MODEL_ORDER], gap.std()[MODEL_ORDER] / np.sqrt(len(w))
    b.barh(range(len(MODEL_ORDER)), mu, xerr=1.96 * se, color=[MODEL_COLOR[m] for m in MODEL_ORDER], height=0.6,
           error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    for i, m in enumerate(MODEL_ORDER):
        b.text(mu[m] + 1.96 * se[m] + 0.003, i, f"{mu[m]:.3f}", va="center", fontsize=7.5)
    b.set_yticks(range(len(MODEL_ORDER)), [MODEL_LABEL[m] for m in MODEL_ORDER], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("mean AUC gap to the oracle ceiling (lower = better; bars = 95% CI)")
    b.set_title("Distance from the best possible AUC")
    b.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig09_random_study_summary.png")


def fig_vs_nonadditive(rules: pd.DataFrame, res: pd.DataFrame, out: Path) -> Path:
    _style()
    w = _contenders(res)
    nf = rules.set_index("rule_id").nonadditive_fraction.reindex(w.index)
    edges = np.array([0, 0.2, 0.4, 0.6, 0.8001])
    mid = (edges[:-1] + edges[1:]) / 2
    bins = pd.cut(nf, edges, include_lowest=True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    for m in MODEL_ORDER:
        mean = w[m].groupby(bins, observed=False).mean()
        ok = ~np.isnan(mean.values)
        a.plot(mid[ok], mean.values[ok], color=MODEL_COLOR[m], marker=MODEL_MARKER[m], lw=2, ms=6, label=MODEL_LABEL[m])
    a.set_xlabel("non-additive share of the rule's signal")
    a.set_ylabel("mean AUC (nested CV, tuned)")
    a.set_title("Performance as the rule becomes less additive")
    a.legend(fontsize=7.5)
    adv = w[["random_forest", "gradient_boosting", "svm_rbf", "stacking"]].max(axis=1) - w["logreg"]
    b.scatter(nf, adv, s=24, color=INK2, alpha=0.6, edgecolor="white", linewidth=0.4)
    mean_adv = adv.groupby(bins, observed=False).mean()
    ok = ~np.isnan(mean_adv.values)
    b.plot(mid[ok], mean_adv.values[ok], color=ORANGE, lw=2.2, marker="o", label="mean per bin")
    b.axhline(0, color=INK, lw=1)
    b.set_xlabel("non-additive share of the rule's signal")
    b.set_ylabel("best non-linear model minus logistic regression (AUC)")
    b.set_title("Above zero: a non-linear model beat logistic regression")
    b.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig10_auc_vs_nonadditive.png")


def fig_tuning_gain(res: pd.DataFrame, out: Path) -> Path:
    _style()
    m = res[res.feature_set == "snps_plus_env"]
    algos = ["logreg", "random_forest", "gradient_boosting", "svm_rbf"]
    piv = m[m.algorithm.isin(algos)].pivot_table(index=["rule_id", "algorithm"], columns="tuning", values="auc").reset_index()
    gain = piv.assign(g=piv.tuned - piv.default)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(algos))
    d, t = piv.groupby("algorithm").default.mean()[algos], piv.groupby("algorithm").tuned.mean()[algos]
    for i, k in enumerate(algos):                      # dot plot, so the axis need not start at zero
        a.plot([i, i], [d[k], t[k]], color=INK2, lw=1.5, zorder=1)
        a.scatter(i, d[k], s=70, facecolor="white", edgecolor=MODEL_COLOR[k], linewidth=2, zorder=3)
        a.scatter(i, t[k], s=70, color=MODEL_COLOR[k], zorder=3)
        hi, lo = max(d[k], t[k]), min(d[k], t[k])           # higher value labelled above, lower below: never overlap
        a.text(i + 0.12, hi, f"{hi:.3f}", fontsize=7.5, va="bottom")
        a.text(i + 0.12, lo, f"{lo:.3f}", fontsize=7.5, va="top")
    a.scatter([], [], s=70, facecolor="white", edgecolor=INK2, linewidth=2, label="default settings")
    a.scatter([], [], s=70, color=INK2, label="tuned (nested CV)")
    a.set_xticks(x, [MODEL_LABEL[k].replace(" ", "\n", 1) for k in algos], fontsize=8)
    a.set_xlim(-0.5, len(algos) - 0.4)
    pad = 0.02
    a.set_ylim(min(d.min(), t.min()) - pad, max(d.max(), t.max()) + pad)
    a.set_ylabel("mean AUC across random rules")
    a.set_title("Default vs tuned")
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(axis="x", visible=False)
    g = gain.groupby("algorithm").g.agg(["mean", "std", "count"]).reindex(algos)
    se = g["std"] / np.sqrt(g["count"])
    b.barh(range(len(algos)), g["mean"], xerr=1.96 * se, color=[MODEL_COLOR[k] for k in algos], height=0.6,
           error_kw={"ecolor": INK2, "capsize": 3, "lw": 1})
    b.axvline(0, color=INK, lw=1)
    for i, k in enumerate(algos):
        b.text(g["mean"][k] + (1.96 * se[k]) + 0.001, i, f"{g['mean'][k]:+.3f}", va="center", fontsize=7.5)
    b.set_yticks(range(len(algos)), [MODEL_LABEL[k] for k in algos], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("AUC gain from tuning (95% CI)")
    b.set_title("What tuning bought each algorithm")
    b.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig11_tuning_gain.png")


def fig_rule_distribution(rules: pd.DataFrame, out: Path) -> Path:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, col, title in zip(axes, ["nonadditive_fraction", "signal_share", "oracle_auc_full"],
                              ["Non-additive share of signal", "Signal share of liability", "Oracle AUC (best possible)"]):
        ax.hist(rules[col], bins=12, color=BLUE, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("random rules")
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "fig14_random_rule_distribution.png", "These distributions are assumptions (configs/randomized_study.yaml), not data. " + SIM_NOTE)


# ------------------------------------------------------------------ scenario grid
def fig_scenario_grid(grid: pd.DataFrame, out: Path) -> Path:
    _style()
    m = grid[grid.feature_set == "snps_plus_env"].copy()
    m["model"] = m.algorithm + "@" + m.tuning
    m = m[m.model.isin(CONT)]
    structures = list(dict.fromkeys(grid.structure))
    cases = list(dict.fromkeys(zip(grid.scenario, grid.trait)))
    fig, axes = plt.subplots(1, len(cases), figsize=(4.4 * len(cases) + 1.6, 4.4), squeeze=False, sharey=True)
    vmin = float(m.auc.min()) - 0.02
    vmax = float(max(m.auc.max(), m.oracle_auc_full.max())) + 0.005
    for ax, (s, t) in zip(axes[0], cases):
        sub = m[(m.scenario == s) & (m.trait == t)]
        P = sub.pivot_table(index="model", columns="structure", values="auc")[structures].reindex(CONT)
        orc = sub.groupby("structure").oracle_auc_full.mean()[structures]
        M = np.vstack([P.to_numpy(), orc.to_numpy()])
        im = _heat(ax, M, [x.replace("_", "\n") for x in structures],
                   [MODEL_LABEL[c.split("@")[0]] for c in CONT] + ["Oracle (ceiling)"], SEQ, vmin, vmax, fmt="{:.3f}", fs=7.5)
        ax.axhline(len(CONT) - 0.5, color="white", lw=3)
        ax.set_title(f"{s} · {t}", fontsize=9)
        ax.set_xticks(range(len(structures)), [x.replace("_", "\n") for x in structures], rotation=0, ha="center", fontsize=8)
        if (s, t) != cases[0]:
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig.colorbar(im, ax=axes[0].tolist(), label="AUC (nested CV; mean over replicates)", shrink=0.8)
    fig.suptitle("Same SNPs, four rule structures (additive to strongly non-additive)", x=0.01, ha="left", fontsize=10, fontweight="bold")
    return _save(fig, out, "fig12_scenario_grid.png", "Oracle = AUC of the true signal. " + SIM_NOTE)


def fig_genetic_gain(grid: pd.DataFrame, rules: pd.DataFrame | None, res: pd.DataFrame | None, out: Path) -> Path:
    _style()
    lr = grid[(grid.algorithm == "logreg") & (grid.tuning == "tuned")]
    piv = lr.pivot_table(index=["scenario", "trait", "structure", "replicate"], columns="feature_set", values="auc").reset_index()
    piv["gain"] = piv.snps_plus_env - piv.env_only
    g = piv.groupby(["scenario", "trait", "structure"]).gain.agg(["mean", "std"]).reset_index()
    structures = list(dict.fromkeys(grid.structure))
    cases = list(dict.fromkeys(zip(g.scenario, g.trait)))
    n_panels = 2 if res is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 + 4.6 * (n_panels - 1), 4.2), squeeze=False)
    a = axes[0][0]
    w = 0.8 / len(cases)
    colors = [BLUE, ORANGE, AQUA, YELLOW]
    for i, (s, t) in enumerate(cases):
        sub = g[(g.scenario == s) & (g.trait == t)].set_index("structure").reindex(structures)
        xs = np.arange(len(structures)) + (i - (len(cases) - 1) / 2) * w
        a.bar(xs, sub["mean"], w * 0.92, yerr=sub["std"].fillna(0), color=colors[i % 4], label=f"{s} · {t}",
              error_kw={"ecolor": INK2, "capsize": 2, "lw": 0.8})
        for x_, v in zip(xs, sub["mean"]):
            a.text(x_, v + 0.004, f"{v:+.2f}", ha="center", fontsize=6.5)
    a.axhline(0, color=INK, lw=1)
    a.set_xticks(range(len(structures)), structures, fontsize=8)
    a.set_ylabel("AUC gain from adding SNPs to the environment (logistic regression)")
    a.set_title("Genetic gain by scenario and rule structure")
    top = float((g["mean"] + g["std"].fillna(0)).max())
    a.set_ylim(min(0.0, float(g["mean"].min()) - 0.02), top * 1.45 + 0.01)
    a.legend(fontsize=7, loc="upper left", ncol=1)
    a.grid(axis="x", visible=False)
    if res is not None:
        b = axes[0][1]
        f = res[(res.algorithm == "logreg") & (res.tuning == "tuned")].pivot(index="rule_id", columns="feature_set", values="auc")
        gain = (f.snps_plus_env - f.env_only).dropna()
        b.hist(gain, bins=14, color=BLUE, edgecolor="white")
        b.axvline(gain.mean(), color=ORANGE, lw=2)
        b.text(gain.mean(), b.get_ylim()[1] * 0.95, f" mean {gain.mean():+.3f}", color=INK, fontsize=8, va="top")
        b.set_xlabel("AUC gain from adding SNPs (random rules)")
        b.set_ylabel("random rules")
        b.set_title("Genetic gain across the random rules")
        b.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig13_genetic_gain.png")


# ------------------------------------------------------------------ SNP <-> variable / label heatmaps
def fig_snp_vs_variables(ref: pd.DataFrame, features: list[str], out: Path, ref_name: str) -> Path:
    """Rows = SNPs; columns = synthetic covariates and the simulated label. Shows what is (not) related to the SNPs."""
    _style()
    _, snps, covs = _ref_parts(ref, features)
    cols = covs + ["y"]
    M = np.array([[np.corrcoef(ref[s], ref[c])[0, 1] if ref[s].std() > 0 and ref[c].std() > 0 else np.nan for c in cols] for s in snps])
    n = len(ref)
    fig, ax = plt.subplots(figsize=(0.62 * len(cols) + 2.8, 0.42 * len(snps) + 2.4))
    im = _heat(ax, M, [c if c != "y" else "LABEL (deficient)" for c in cols], snps, DIV, -0.3, 0.3, fmt="{:+.2f}", fs=6.5)
    ax.axvline(len(covs) - 0.5, color=INK, lw=1.4)
    fig.colorbar(im, ax=ax, label="Pearson correlation (colour scale +/-0.3)", shrink=0.8)
    ax.set_title(f"SNP genotypes vs covariates and label ({ref_name})")
    return _save(fig, out, "fig15_snp_vs_variables.png",
                 f"Covariates were generated independently of genotypes (assumption), so SNP-covariate cells should be ~0. With n={n}, |r| < {2 / np.sqrt(n):.2f} is "
                 "indistinguishable from chance. " + SIM_NOTE)


def fig_snp_label_by_rule(corr: pd.DataFrame, in_rule: pd.DataFrame, out: Path, n: int) -> Path:
    """Rows = SNPs; columns = scenario / trait / structure. Boxed cells = SNP is part of that rule."""
    _style()
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(0.62 * corr.shape[1] + 3.4, 0.42 * corr.shape[0] + 3.0))
    parts = [c.split(" | ") for c in corr.columns]
    im = _heat(ax, corr.to_numpy(), [p[2] for p in parts], list(corr.index), DIV, -0.3, 0.3, fmt="{:+.2f}", fs=6.5)
    for j, c in enumerate(corr.columns):
        for i, s in enumerate(corr.index):
            if in_rule.loc[s, c]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.6))
    ax.set_xticks(range(len(parts)), [p[2].replace("_", " ") for p in parts], rotation=40, ha="right", fontsize=7)
    groups = list(dict.fromkeys((p[0], p[1]) for p in parts))             # scenario / trait group headers above the columns
    for g in groups:
        idx = [j for j, p in enumerate(parts) if (p[0], p[1]) == g]
        ax.text(np.mean(idx), -0.9, f"{g[0]}\n{g[1]}", ha="center", va="bottom", fontsize=8, fontweight="bold", color=INK)
        if idx[0] > 0:
            ax.axvline(idx[0] - 0.5, color="white", lw=4)
    fig.colorbar(im, ax=ax, label="correlation of SNP dosage with the label", shrink=0.8)
    ax.set_title("Which SNPs move with the simulated label, by scenario and rule structure", pad=38)
    return _save(fig, out, "fig16_snp_label_by_rule.png",
                 f"Boxed = SNP is in that rule (risk allele sign applied in the rule; correlation uses raw alt-allele dosage). |r| < {2 / np.sqrt(n):.2f} is chance at n={n}. " + SIM_NOTE)


# ------------------------------------------------------------------ stress test
def fig_stress_comparison(main_rules: pd.DataFrame | None, main_res: pd.DataFrame | None,
                          stress_rules: pd.DataFrame, stress_res: pd.DataFrame, out: Path) -> Path:
    """Main random rules vs 80-100% non-additive pure-interaction rules: mean AUC per model, and flexible-minus-logistic advantage."""
    _style()
    ws = _contenders(stress_res)
    wm = _contenders(main_res) if main_res is not None else None
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.8, 4.4), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(MODEL_ORDER))
    for w, filled, off, lab in [(wm, False, -0.12, "random rules (0-80% non-additive)"), (ws, True, 0.12, "stress test (80-100%, pure interactions)")]:
        if w is None:
            continue
        mu, se = w.mean()[MODEL_ORDER], 1.96 * w.std()[MODEL_ORDER] / np.sqrt(len(w))
        for i, m in enumerate(MODEL_ORDER):
            a.errorbar(i + off, mu[m], yerr=se[m], fmt="none", ecolor=INK2, capsize=3, lw=1)
            a.scatter(i + off, mu[m], s=70, marker=MODEL_MARKER[m], zorder=3, color=MODEL_COLOR[m] if filled else "white",
                      edgecolor=MODEL_COLOR[m], linewidth=2)
        a.scatter([], [], s=70, marker="o", color=INK2 if filled else "white", edgecolor=INK2, linewidth=2, label=lab)
    a.set_xticks(x, [MODEL_LABEL[m].replace(" ", "\n", 1) for m in MODEL_ORDER], fontsize=8)
    a.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    a.set_ylabel("mean AUC (nested CV, tuned); bars = 95% CI")
    a.set_title("Mean AUC by model")
    a.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    a.grid(axis="x", visible=False)
    rng = np.random.default_rng(0)
    for k, (w, lab) in enumerate([(wm, "random\nrules"), (ws, "stress\ntest")]):
        if w is None:
            continue
        adv = (w[["random_forest", "gradient_boosting", "svm_rbf", "stacking"]].max(axis=1) - w["logreg"]).to_numpy()
        b.scatter(k + rng.uniform(-0.12, 0.12, len(adv)), adv, s=26, color=BLUE if k == 0 else ORANGE, alpha=0.7, edgecolor="white", linewidth=0.4)
        b.hlines(adv.mean(), k - 0.28, k + 0.28, color=INK, lw=2.5)
        b.text(k + 0.3, adv.mean(), f"mean {adv.mean():+.3f}", va="center", fontsize=8)
    b.axhline(0, color=INK2, lw=1, ls="--")
    b.set_xticks([0, 1], ["random\nrules", "stress\ntest"])
    b.set_xlim(-0.6, 1.9)
    b.set_ylabel("best flexible model minus logistic regression (AUC)")
    b.set_title("Above zero: a flexible model beat logistic regression")
    b.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, out, "fig17_stress_test.png", "Stress rules: non-additive part is orthogonal to every SNP and covariate (no linear signal). " + SIM_NOTE)
