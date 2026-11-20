"""Select flip-related attention heads with a cross-validated linear probe, for one model.

    python scripts/run_probe.py --run <main-run folder> --labels <answer-reading labels.csv> \
        --out outputs/probe/<model>

<main-run folder> holds config.json and attention.npz from scripts/run_pilot.py (main split);
labels.csv is the answer-reading output for the same run. Writes to --out:
  turns.csv      one row per used pushback turn: fold, flip, held-out probe score, mean share
                 over the "lower" heads, the "higher" heads and all heads (inputs for RQ2, RQ3, RQ5)
  heads.csv      per fold and head: coefficient, training difference, selected group
  stable_heads.csv  heads selected in at least 3 of 5 folds, with their group
  summary.json   settings, sizes, held-out AUC per fold
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sycoattn.analysis.probe import CONDITIONS, cross_validated_probe, eligible_turns, head_shares, stable_heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--group", default="evidence", help="span group used as features (evidence | answer_line)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--C", type=float, default=0.1, help="inverse L2 regularisation strength")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    rows = eligible_turns(pd.read_csv(args.labels))
    attention = np.load(args.run / "attention.npz")
    n_layers_heads = attention[f"{rows.dialogue_id.iloc[0]}::{args.group}"].shape[1:]
    X = head_shares(attention, rows, args.group)

    turns, heads, folds = cross_validated_probe(X, rows, args.top_k, args.C, args.folds, args.seed)
    stable = stable_heads(heads, n_layers_heads)

    args.out.mkdir(parents=True, exist_ok=True)
    turns.to_csv(args.out / "turns.csv", index=False)
    heads.to_csv(args.out / "heads.csv", index=False)
    stable.to_csv(args.out / "stable_heads.csv", index=False)
    summary = {
        "model": cfg["model"]["name"], "run": str(args.run), "labels": str(args.labels),
        "group": args.group, "conditions": list(CONDITIONS), "top_k": args.top_k, "C": args.C,
        "folds": args.folds, "seed": args.seed, "layers_heads": list(n_layers_heads),
        "rows": len(rows), "flips": int(rows.flip.sum()), "records": int(rows.record.nunique()),
        "probe_auc_mean": float(np.nanmean([f["probe_auc"] for f in folds])),
        "lower_auc_mean": float(np.nanmean([f["lower_auc"] for f in folds])),
        "higher_auc_mean": float(np.nanmean([f["higher_auc"] for f in folds])),
        "all_heads_auc_mean": float(np.nanmean([f["all_heads_auc"] for f in folds])),
        "stable_heads": len(stable), "stable_lower": int((stable.group == "lower").sum()),
        "stable_higher": int((stable.group == "higher").sum()), "per_fold": folds,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{summary['model']}: {summary['rows']} turns, {summary['flips']} flips, "
          f"AUC probe {summary['probe_auc_mean']:.3f}, lower heads {summary['lower_auc_mean']:.3f}, "
          f"higher heads {summary['higher_auc_mean']:.3f}, all heads {summary['all_heads_auc_mean']:.3f}; "
          f"stable heads {summary['stable_lower']} lower / {summary['stable_higher']} higher -> {args.out}")


if __name__ == "__main__":
    main()
