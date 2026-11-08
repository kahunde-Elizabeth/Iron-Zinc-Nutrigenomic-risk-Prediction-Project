"""Build the demo bundle for the web app: one logistic-regression model, held-out scores, risk tiers, calibration.

Output: results/demo/bundle.json and results/demo/people.csv. Then run:  streamlit run app/demo_app.py
The labels are SIMULATED (a rule from the random-rule study with the median oracle AUC); nothing here is a clinical result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from izn.demo.bundle import build_bundle, save_bundle
from izn.evaluation.study_data import load_study_data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--panel", type=Path, default=Path("configs/snp_panel.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/demo"))
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    bundle, people = build_bundle(data, cfg, cfg["seed"], args.panel)
    save_bundle(bundle, people, args.out_dir)
    m, t = bundle["metrics"], bundle["tiers"]
    print(f"demo rule {bundle['rule_name']} (oracle AUC {bundle['rule_oracle_auc']:.3f}); {m['n']} people, simulated prevalence {m['prevalence']:.2f}")
    print(f"held-out AUC {m['auc_heldout']:.3f}; tiers cut at {t['cut_low']:.2f} and {t['cut_high']:.2f}")
    for r in t["table"]:
        print(f"  {r['tier']:6s} n={r['n']:3d} ({r['share']:.0%})  simulated deficiency rate {r['observed_rate']:.0%}" if r["observed_rate"] is not None else f"  {r['tier']:6s} n=0")
    print(f"saved to {args.out_dir}. Next: streamlit run app/demo_app.py")


if __name__ == "__main__":
    main()
