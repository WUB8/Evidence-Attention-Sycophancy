import numpy as np
import pandas as pd

from sycoattn.analysis.probe import (cross_validated_probe, eligible_turns, head_shares, record_of,
                                     stable_heads, standardize_within_turn)


def labels_for(dialogue, item, condition, flips):
    """flips: flip labels for turns 2.. (turn 1 is the question)."""
    rows = [dict(dialogue_id=dialogue, item_id=item, condition=condition, turn=1, kind="question",
                 flip_label="Not scored")]
    rows += [dict(dialogue_id=dialogue, item_id=item, condition=condition, turn=t, kind="pushback",
                  flip_label=f) for t, f in enumerate(flips, start=2)]
    return rows


def test_record_of():
    assert record_of("mwz-hotel-25-area") == "mwz-hotel-25"
    assert record_of("mwz-restaurant-19268-pricerange") == "mwz-restaurant-19268"


def test_eligible_turns_cut_after_first_flip_and_drop_others():
    labels = pd.DataFrame(
        labels_for("x|A", "mwz-hotel-1-area", "A", ["No flip", "Flip", "Flip", "No flip"])
        + labels_for("x|D", "mwz-hotel-1-area", "D", ["No flip"] * 4)
        + labels_for("y|C", "mwz-hotel-2-stars", "C", ["Unclear", "No flip", "No flip"]))
    rows = eligible_turns(labels)
    assert rows[rows.dialogue_id == "x|A"].turn.tolist() == [2, 3]   # stops at the first flip
    assert "D" not in set(rows.condition)                             # D is not used
    assert rows[rows.dialogue_id == "y|C"].turn.tolist() == [3, 4]   # unclear dropped
    assert rows.record.tolist()[:1] == ["mwz-hotel-1"]


def test_within_turn_standardization_removes_turn_level_differences():
    rows = pd.DataFrame({"condition": ["A"] * 4, "turn": [2, 2, 3, 3]})
    X = np.array([[0.9], [0.7], [0.2], [0.0]])                      # later turn: lower share
    Z = standardize_within_turn(X, rows)
    assert np.allclose(Z[:, 0], [1, -1, 1, -1], atol=1e-3)            # both turns centred the same way


def synthetic(n_records=60, layers=2, heads=4, lower=(1, 2), higher=(0, 1), seed=0):
    """On flip turns one head looks at the record less, another looks more; the rest is noise."""
    rng = np.random.default_rng(seed)
    labels, attention = [], {}
    for r in range(n_records):
        item = f"mwz-hotel-{r}-area"
        for cond in ["A", "C"]:
            flips = ["Flip" if rng.random() < 0.3 else "No flip" for _ in range(3)]
            labels += labels_for(f"{item}|{cond}", item, cond, flips)
            arr = rng.uniform(0.2, 0.4, size=(4, layers, heads)).astype(np.float16)
            for t, f in enumerate(flips, start=1):
                arr[t, lower[0], lower[1]] = 0.05 if f == "Flip" else 0.5
                arr[t, higher[0], higher[1]] = 0.6 if f == "Flip" else 0.1
            attention[f"{item}|{cond}::evidence"] = arr
    return pd.DataFrame(labels), attention


def test_probe_finds_both_head_groups_without_mixing_records():
    labels, attention = synthetic()
    rows = eligible_turns(labels)
    X = head_shares(attention, rows)
    assert X.shape == (len(rows), 8)
    turns, heads, folds = cross_validated_probe(X, rows, top_k=2, n_folds=5)
    # every record sits in exactly one held-out fold
    assert (turns.groupby("record").fold.nunique() == 1).all() and (turns.fold >= 0).all()
    assert all(f["probe_auc"] > 0.95 and f["lower_auc"] > 0.95 and f["higher_auc"] > 0.95 for f in folds)
    stable = stable_heads(heads, (2, 4), min_folds=5).set_index(["layer", "head"]).group.to_dict()
    assert stable == {(1, 2): "lower", (0, 1): "higher"}
    by_flip = turns.groupby("flip")[["share_lower", "share_higher"]].mean()
    assert by_flip.loc[1, "share_lower"] < by_flip.loc[0, "share_lower"]
    assert by_flip.loc[1, "share_higher"] > by_flip.loc[0, "share_higher"]
