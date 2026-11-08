"""Iron-zinc risk pipeline: demo interface (research prototype).

Run from the project root:   streamlit run app/demo_app.py
Needs results/demo/ (python scripts/27_build_demo_bundle.py). The 'deficiency' label behind every number is SIMULATED.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from izn.demo.bundle import load_bundle, score_person

BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
TIER_COLOR = {"Low": "#1baf7a", "Medium": "#eda100", "High": "#eb6834"}
DEMO_DIR = Path(os.environ.get("IZN_DEMO_DIR", ROOT / "results" / "demo"))

st.set_page_config(page_title="Iron-zinc risk pipeline demo", page_icon="🧬", layout="wide")
st.markdown("""<style>
.badge{display:inline-block;padding:.35rem 1.1rem;border-radius:999px;color:#fff;font-weight:700;font-size:1.5rem}
.small{color:#52514e;font-size:.9rem}
.card{background:#f4f3ef;border-radius:12px;padding:1rem 1.2rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load(path: str):
    return load_bundle(Path(path))


if not (DEMO_DIR / "bundle.json").exists():
    st.error("No demo data found. Run `python scripts/27_build_demo_bundle.py` first, then restart this app.")
    st.stop()
bundle, people = load(str(DEMO_DIR))
F, M, T = bundle["features"], bundle["metrics"], bundle["tiers"]
feats = bundle["model"]["features"]
snps = [c for c in feats if F[c]["kind"] == "snp"]
tier_rate = {r["tier"]: r["observed_rate"] for r in T["table"]}


def nice(c: str) -> str:
    return F[c]["label"]


def contrib_chart(contribs, top=8):
    rows = contribs[:top][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(rows) + 0.9))
    ax.barh([nice(c) for c, _ in rows], [v for _, v in rows], color=[ORANGE if v > 0 else BLUE for _, v in rows], height=0.62)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("pushes the estimate  ←  lower  |  higher  →  (relative to an average person)", fontsize=8, color=INK2)
    ax.tick_params(labelsize=8.5, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


st.title("Iron and zinc risk: pipeline demo")
st.warning("**Research prototype.** The genotypes are real (1000 Genomes, African populations). The covariates and the 'deficiency' label are "
           "**simulated**, because real paired data is not yet available. This demo shows how the pipeline works. It is not medical advice and "
           "it does not diagnose anyone.")

tab1, tab2, tab3 = st.tabs(["Try a person", "How well does it work?", "How it works"])

# ------------------------------------------------------------------ tab 1
with tab1:
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        mode = st.radio("Whose data?", ["Someone from the cohort", "Enter a new person"], horizontal=True)
        values, label_known, held_prob = {}, None, None
        if mode == "Someone from the cohort":
            if "pick" not in st.session_state:
                st.session_state.pick = people["sample"].iloc[0]
            if st.button("Pick a random person"):
                st.session_state.pick = people["sample"].sample(1).iloc[0]
            sid = st.selectbox("Person ID", people["sample"].tolist(), index=int(people.index[people["sample"] == st.session_state.pick][0]))
            st.session_state.pick = sid
            row = people[people["sample"] == sid].iloc[0]
            values = {c: row[c] for c in feats}
            label_known, held_prob = int(row["label"]), float(row["heldout_prob"])
            kind = "child under five" if values.get("child", 0) == 1 else "woman of reproductive age"
            st.markdown(f"**{sid}**: {row['pop']} population, {kind}.")
            st.markdown("**Genetic results** (copies of the alternate allele at each variant)")
            st.dataframe(pd.DataFrame({"Variant (gene)": [nice(c) for c in snps], "Copies": [int(values[c]) for c in snps]}), hide_index=True, height=210)
            flags = [nice(c) for c in feats if F[c]["kind"] == "binary" and values[c] == 1 and c not in ("child",)]
            st.markdown("**Health and exposure factors present:** " + (", ".join(flags) if flags else "none"))
        else:
            grp = st.radio("Who is this?", ["Child under five", "Woman aged 15 to 49"], horizontal=True)
            child = grp.startswith("Child")
            if child:
                age_m = st.slider("Age (months)", 6, 59, 24)
                age_y, preg, menses = 0, False, False
            else:
                age_y = st.slider("Age (years)", 15, 49, 28)
                age_m = 0
                preg = st.checkbox("Pregnant")
                menses = st.checkbox("Heavy periods")
            c1, c2 = st.columns(2)
            hook = c1.checkbox("Hookworm infection")
            mal = c1.checkbox("Recent malaria")
            infl = c2.checkbox("Current inflammation or infection")
            diar = c2.checkbox("Recent diarrhoea")
            diet_i = st.slider("Dietary iron intake (0 = average, negative = low)", -3.0, 3.0, 0.0, 0.1)
            diet_z = st.slider("Dietary zinc intake (0 = average, negative = low)", -3.0, 3.0, 0.0, 0.1)
            with st.expander("Genetic results (copies of the alternate allele at each variant)", expanded=False):
                g = {c: st.selectbox(nice(c), [0, 1, 2], key=f"g_{c}") for c in snps}
            values = {**g, "child": float(child), "age_months": age_m, "age_years": age_y, "young_child": float(child and age_m < 24),
                      "pregnant": float(preg), "heavy_menses": float(menses), "hookworm": float(hook), "blood_loss": float(menses or hook),
                      "malaria": float(mal), "inflammation": float(infl), "diarrhoea": float(diar), "diet_iron": diet_i, "diet_zinc": diet_z}
            values = {c: values.get(c, 0.0) for c in feats}
    res = score_person(bundle, values)
    prob = held_prob if held_prob is not None else res["prob"]
    tier = res["tier"] if held_prob is None else next(t for t in ("Low", "Medium", "High") if (t == "Low" and prob < T["cut_low"]) or
                                                      (t == "Medium" and T["cut_low"] <= prob < T["cut_high"]) or (t == "High" and prob >= T["cut_high"]))
    with right:
        st.markdown("### Result")
        st.markdown(f"<span class='badge' style='background:{TIER_COLOR[tier]}'>{tier} estimated risk</span>", unsafe_allow_html=True)
        st.markdown(f"#### Estimated probability: **{prob:.0%}**")
        st.markdown(f"<div class='card'>In this simulated cohort, <b>{tier_rate[tier]:.0%}</b> of people in the <b>{tier.lower()}</b> tier were labelled "
                    f"deficient, against <b>{M['prevalence']:.0%}</b> overall.</div>", unsafe_allow_html=True)
        if held_prob is not None:
            st.caption("This score comes from a model that did not see this person while it was trained (cross-validation).")
            st.markdown(f"Simulated label for this person: **{'deficient' if label_known else 'not deficient'}**.")
        else:
            st.caption("This person is not in the cohort, so there is no label to compare with. The estimate comes from the model trained on everyone.")
        st.markdown("**What pushed the estimate up or down**")
        st.pyplot(contrib_chart(res["contributions"]), clear_figure=True)
        st.caption("Bars show each factor's contribution in the model, relative to an average person. Orange raises the estimate; blue lowers it.")

# ------------------------------------------------------------------ tab 2
with tab2:
    st.markdown("Every number here is computed on people the model had **not** seen while learning (5-fold cross-validation, with settings tuned inside each fold).")
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out AUC", f"{M['auc_heldout']:.2f}", help="0.5 = guessing, 1.0 = perfect. Measured against the simulated label.")
    c2.metric("People", f"{M['n']}")
    c3.metric("Simulated deficiency rate", f"{M['prevalence']:.0%}")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("**Do the tiers separate people?**")
        tt = pd.DataFrame(T["table"])
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.bar(tt["tier"], tt["observed_rate"] * 100, color=[TIER_COLOR[t] for t in tt["tier"]], width=0.6)
        for i, r in tt.iterrows():
            ax.text(i, r["observed_rate"] * 100 + 1.5, f"{r['observed_rate']:.0%}\n(n={r['n']})", ha="center", fontsize=8.5)
        ax.axhline(M["prevalence"] * 100, color=INK2, ls="--", lw=1)
        ax.set_ylabel("simulated deficiency rate (%)")
        ax.set_ylim(0, max(100, tt["observed_rate"].max() * 100 + 15))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(f"Tiers: low below {T['cut_low']:.0%}, medium up to {T['cut_high']:.0%}, high above. These cut-offs are the cohort rate and double it, chosen before the results were seen. Dashed line = overall rate.")
    with b:
        st.markdown("**Is a 30% estimate really about 30%?**")
        cal = pd.DataFrame(bundle["calibration"])
        fig, ax = plt.subplots(figsize=(5, 3.4))
        lim = max(cal.predicted.max(), cal.observed.max()) * 100 + 5
        ax.plot([0, lim], [0, lim], color=INK2, ls="--", lw=1)
        ax.plot(cal.predicted * 100, cal.observed * 100, "o-", color=BLUE, lw=2, ms=7)
        ax.set_xlabel("average estimate in the group (%)")
        ax.set_ylabel("simulated rate in the group (%)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Five equal groups ordered by estimate. Points near the dashed line mean the probabilities are trustworthy.")

# ------------------------------------------------------------------ tab 3
with tab3:
    st.markdown("""
**The path one person's data takes**
1. **Raw data in.** The person's genotype comes from a VCF file held in the object store (Docker, S3-compatible).
2. **Panel extraction.** Only the 13 iron and zinc variants are read out, so the pipeline never loads the whole genome.
3. **Features.** Each variant becomes 0, 1 or 2 copies of an allele. Health and diet factors are added (simulated here).
4. **Model.** A logistic regression, chosen because its score can be explained factor by factor. In the wider study four algorithms and a stacked ensemble were compared.
5. **Risk tier.** The estimated probability is placed in a low, medium or high tier, relative to the cohort rate.

**What is real, and what is not**
- Real: the genotypes of 504 people from five African populations (1000 Genomes).
- Simulated: the health and diet factors, and the deficiency label itself. Real measurements (ferritin, zinc) need approved access to paired genetic and blood data.
- So the percentages show how the pipeline behaves, not how likely any real person is to be deficient.

**Why a simulated label still helps.** It lets the whole pipeline be built and checked now: ingestion, feature building, fair model comparison, held-out evaluation and risk tiers. Swapping in real labels later changes the data, not the machinery.

**Limits to keep in mind**
- 504 people is small; single estimates are noisy and complex interactions are hard to learn at this size.
- The five 1000 Genomes populations do not include Uganda. LWK (Kenya) is the closest, and it was the hardest group to predict when held out.
- Genetic variants explain only part of iron and zinc status; diet, infection and blood loss matter too.
""")
    st.caption(f"Simulated rule used for this demo: {bundle['rule_name']}. Built {bundle['generated']}.")
