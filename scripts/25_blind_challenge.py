"""Sealed blind challenge: does the pipeline pick a good model when it does NOT know the rule?

  python scripts/25_blind_challenge.py seal     # draws hidden rules, writes labels, seals the rules, prints a commitment hash
  python scripts/25_blind_challenge.py analyze  # model choice by nested CV on a training split only, then a locked test split
  python scripts/25_blind_challenge.py reveal   # opens the sealed rules, checks the commitment, compares choice with the truth

Do NOT open results/blind/SEALED_rules_do_not_open.json before `analyze` has finished. To timestamp the seal, send the
commitment hash (results/blind/commitment.txt) to your supervisor before you analyse. Labels are SIMULATED.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import yaml

from izn.evaluation import blind as B
from izn.evaluation.study_data import load_study_data
from izn.reporting import figures_blind as FB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["seal", "analyze", "reveal"])
    ap.add_argument("--features", type=Path, default=Path("data/processed/p1_features.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/processed/p1_feature_meta.tsv"))
    ap.add_argument("--env-data", type=Path, default=Path("data/processed/p1_env.parquet"))
    ap.add_argument("--scenarios", type=Path, default=Path("configs/scenarios.yaml"))
    ap.add_argument("--env", type=Path, default=Path("configs/simulation_env.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/randomized_study.yaml"))
    ap.add_argument("--blind-config", type=Path, default=Path("configs/blind_teacher.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/blind"))
    ap.add_argument("--n-challenges", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--quick", action="store_true", help="small grids and 4 challenges: a fast check, not the final run")
    ap.add_argument("--force", action="store_true", help="reveal without analyze (not recommended)")
    args = ap.parse_args()

    cfg = {**yaml.safe_load(args.config.read_text(encoding="utf-8")), **yaml.safe_load(args.blind_config.read_text(encoding="utf-8"))}
    grids = cfg["quick_grids"] if args.quick else cfg["grids"]
    seed, tf = cfg["seed"], cfg["blind"]["test_fraction"]
    data = load_study_data(args.features, args.scenarios, args.env, args.env_data, args.meta)
    t0 = time.time()

    if args.stage == "seal":
        k = args.n_challenges or (4 if args.quick else cfg["blind"]["n_challenges"])
        digest = B.seal(data, cfg, k, args.out_dir)
        print(f"sealed {k} hidden rules. Labels: {args.out_dir / 'labels_blind.tsv'}")
        print(f"commitment (sha256): {digest}")
        print("Send this hash to your supervisor to timestamp it. Do not open the sealed file. Next: python scripts/25_blind_challenge.py analyze")
    elif args.stage == "analyze":
        print(f"analysing blind labels: {'quick' if args.quick else 'full'} grids, {int(tf * 100)}% locked test split", flush=True)
        res = B.analyze(data, cfg, grids, args.out_dir, tf, seed, args.jobs)
        print(f"\ndone in {(time.time() - t0) / 60:.1f} min. Choices recorded in {args.out_dir / 'preregistered_choice.tsv'}\n")
        print(res[["challenge", "chosen"] + [c for c in res.columns if c.startswith("test_")]].to_string(index=False))
        print("\nNothing about the rules has been looked at. Next: python scripts/25_blind_challenge.py reveal")
    else:
        tab, s = B.reveal(data, args.out_dir, tf, seed, args.force)
        pd.set_option("display.width", 200)
        print(tab[["challenge", "family", "nonadditive_fraction", "oracle_auc_test", "chosen", "chosen_auc", "best_in_hindsight", "best_auc",
                   "logreg_auc", "stacking_auc"]].to_string(index=False))
        print(f"\nhidden rules: {s['n']} ({s['families']})")
        print(f"mean oracle AUC on the test split: {s['mean_oracle']:.3f}; blind choice sits {s['mean_gap_chosen_to_oracle']:.3f} below it on average")
        print(f"blind choice within 0.01 AUC of the best model in hindsight: {s['chosen_best_or_within_0.01']:.0%} of rules; models chosen: {s['chosen_counts']}")
        print("mean regret (best model's AUC minus the strategy's; lower is better; +/- standard error):")
        for k, (m, se) in s["regret"].items():
            print(f"  {k:16s} {m:.4f} +/- {se:.4f}")
        fig_dir = args.out_dir.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        print("figure:", FB.fig_blind_reveal(tab, fig_dir))
        print("\nReminder: labels are simulated. The test split is small, so single-rule differences are noisy; read the averages.")


if __name__ == "__main__":
    main()
