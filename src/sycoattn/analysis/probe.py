"""Linear probe that picks the attention heads most tied to flips (proposal §7, RQ2/RQ3/RQ5).

Each row is one pushback turn: the inputs are the per-head attention shares on the evidence
(saved during the main run), the target is the flip label from answer reading. A logistic
regression learns which heads separate flip turns from non-flip turns; the heads with the
largest weights are the "selected heads".

Selected heads move in both directions, so they are reported as two groups and never averaged
together: "lower" heads look at the record less on flip turns (a drift-like sign), "higher"
heads look at it more (the model still reads the record but gives in, a pressure-like sign).
A head's group comes from its own flip-vs-no-flip difference on the training folds.

Two safeguards:
- Within-turn comparison: every head's share is standardised within its (condition, turn)
  group, so the probe cannot just learn "later turn -> less attention and more flips".
- No circularity: heads are chosen on four folds of records and used on the fifth
  (folds are made by record, so one place never sits on both sides).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

# D restates the record in every pushback and control never flips, so neither is used.
CONDITIONS = ("A", "B3", "B4", "B5", "C")


def record_of(item_id: str) -> str:
    """'mwz-hotel-25-area' -> 'mwz-hotel-25' (attributes contain no hyphen)."""
    return item_id.rsplit("-", 1)[0]


def eligible_turns(labels: pd.DataFrame, conditions=CONDITIONS) -> pd.DataFrame:
    """Readable pushback turns, each dialogue cut after its first flip.

    After a flip the model is defending the wrong answer rather than deciding again, so
    later turns are not new chances to flip. Unclear turns are dropped.
    """
    df = labels[(labels.kind == "pushback") & labels.condition.isin(conditions)
                & labels.flip_label.isin(["Flip", "No flip"])].copy()
    df = df.sort_values(["dialogue_id", "turn"])
    df["flip"] = (df.flip_label == "Flip").astype(int)
    flips_before = df.groupby("dialogue_id").flip.cumsum() - df.flip
    df = df[flips_before == 0]
    df["record"] = df.item_id.map(record_of)
    return df.reset_index(drop=True)


def head_shares(attention, rows: pd.DataFrame, group: str = "evidence") -> np.ndarray:
    """[n_rows, layers * heads] attention share on `group` at each row's turn."""
    return np.stack([np.asarray(attention[f"{r.dialogue_id}::{group}"][r.turn - 1], dtype=np.float32).ravel()
                     for r in rows.itertuples()])


def standardize_within_turn(X: np.ndarray, rows: pd.DataFrame) -> np.ndarray:
    """z-score every column within each (condition, turn) group. Uses no labels."""
    Z = np.empty_like(X)
    for _, idx in rows.groupby(["condition", "turn"]).indices.items():
        block = X[idx]
        Z[idx] = (block - block.mean(0)) / (block.std(0) + 1e-6)
    return Z


def _auc(y, score) -> float:
    return float(roc_auc_score(y, score)) if 0 < y.sum() < len(y) else float("nan")


def cross_validated_probe(X: np.ndarray, rows: pd.DataFrame, top_k: int = 20, C: float = 0.1,
                          n_folds: int = 5, seed: int = 0):
    """Fit the probe on n_folds - 1 folds of records, evaluate on the held-out fold.

    Returns (turns, heads, folds):
      turns  one row per input row: fold, held-out probe score, mean raw share over the
             "lower" and over the "higher" heads selected on the other folds, and over all heads
      heads  one row per (fold, head): coefficient, flip-vs-no-flip difference on the training
             folds (in SDs), whether it was selected and its group
      folds  one dict per fold: sizes, flips, held-out AUC of the probe, of each head group and
             of the all-heads average
    """
    y = rows.flip.to_numpy()
    Z = standardize_within_turn(X, rows)
    turns = rows[["dialogue_id", "item_id", "record", "condition", "turn", "flip"]].copy()
    turns["fold"] = -1
    turns["probe_score"] = np.nan
    turns["share_lower"] = np.nan
    turns["share_higher"] = np.nan
    turns["share_all"] = X.mean(1)
    heads, folds = [], []
    splitter = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(splitter.split(Z, y, groups=rows.record)):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(Z[tr], y[tr])
        coef = clf.coef_[0]
        selected = np.argsort(-np.abs(coef))[:top_k]
        diff = Z[tr][y[tr] == 1].mean(0) - Z[tr][y[tr] == 0].mean(0)
        lower, higher = selected[diff[selected] < 0], selected[diff[selected] >= 0]
        score = clf.decision_function(Z[te])
        turns.loc[te, "fold"] = fold
        turns.loc[te, "probe_score"] = score
        if len(lower):
            turns.loc[te, "share_lower"] = X[te][:, lower].mean(1)
        if len(higher):
            turns.loc[te, "share_higher"] = X[te][:, higher].mean(1)
        group = np.full(len(coef), "", dtype=object)
        group[lower], group[higher] = "lower", "higher"
        heads.append(pd.DataFrame({"fold": fold, "head_index": np.arange(len(coef)), "coef": coef,
                                   "train_diff_sd": diff, "selected": group != "", "group": group}))
        folds.append({"fold": fold, "train_rows": len(tr), "test_rows": len(te),
                      "test_records": int(rows.record.iloc[te].nunique()),
                      "train_flips": int(y[tr].sum()), "test_flips": int(y[te].sum()),
                      "probe_auc": _auc(y[te], score),
                      # each group scored in its own direction: lower heads drop on flips
                      "lower_heads": len(lower), "lower_auc": _auc(y[te], -Z[te][:, lower].mean(1)) if len(lower) else float("nan"),
                      "higher_heads": len(higher), "higher_auc": _auc(y[te], Z[te][:, higher].mean(1)) if len(higher) else float("nan"),
                      # reference for RQ5: the average over every head (low attention -> flip)
                      "all_heads_auc": _auc(y[te], -Z[te].mean(1))})
    return turns, pd.concat(heads, ignore_index=True), folds


def stable_heads(heads: pd.DataFrame, n_layers_heads: tuple[int, int], min_folds: int = 3) -> pd.DataFrame:
    """Heads selected in at least `min_folds` folds: (layer, head), group, mean coefficient and difference."""
    _, n_heads = n_layers_heads
    sel = heads[heads.selected].groupby("head_index").agg(
        folds=("fold", "nunique"), mean_coef=("coef", "mean"), mean_diff_sd=("train_diff_sd", "mean"))
    sel = sel[sel.folds >= min_folds].reset_index()
    sel["layer"], sel["head"] = sel.head_index // n_heads, sel.head_index % n_heads
    sel["group"] = np.where(sel.mean_diff_sd < 0, "lower", "higher")
    return sel.sort_values(["group", "folds", "mean_diff_sd"], ascending=[False, False, True])[
        ["layer", "head", "group", "folds", "mean_coef", "mean_diff_sd"]]
