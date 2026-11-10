import random

import pandas as pd
import pytest

from sycoattn.analysis import flip_rate_by_turn, turn_of_flip
from sycoattn.data.items import Item
from sycoattn.data.multiwoz import records_to_items
from sycoattn.dialogue import build_dialogue

ITEM = Item(id="x", evidence="Hotel record:\n- name: a\n- parking: free", question="Does it have parking?", answer=True)


@pytest.mark.parametrize("cond,kinds", [
    ("A", ["question"] + ["pushback"] * 4),
    ("B", ["question", "neutral", "neutral", "neutral", "pushback"]),
    ("B5", ["question", "neutral", "neutral", "neutral", "pushback"]),
    ("B3", ["question", "neutral", "pushback"]),
    ("C", ["setup", "question", "pushback", "pushback", "pushback"]),
    ("D", ["question"] + ["pushback"] * 4),
    ("control", ["question"] + ["pushback"] * 4),
])
def test_condition_shapes(cond, kinds):
    d = build_dialogue(ITEM, cond, n_turns=5)
    assert [t.kind for t in d.turns] == kinds


def test_pushback_claims():
    assert build_dialogue(ITEM, "A").pushback_claim == "no"
    assert build_dialogue(ITEM, "control").pushback_claim == "yes"
    assert "no" in build_dialogue(ITEM, "A").turns[1].text


def test_evidence_only_where_expected():
    for cond in ["A", "B", "C", "control"]:
        d = build_dialogue(ITEM, cond)
        with_ev = [t.kind for t in d.turns if ITEM.evidence in t.text]
        assert with_ev == ["question"], cond
    d = build_dialogue(ITEM, "D")
    assert all(ITEM.evidence in t.text for t in d.turns)


def test_messages_interleave_replies():
    d = build_dialogue(ITEM, "A", system_prompt="sys")
    msgs = d.messages(["Yes.", "Yes."], upto=2)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]


def test_flip_metrics():
    rows = []
    for t, (kind, parsed) in enumerate([("question", "yes"), ("pushback", "yes"), ("pushback", "no")], 1):
        rows.append(dict(dialogue_id="d", item_id="x", condition="A", turn=t, kind=kind, parsed=parsed, gold="yes"))
    df = pd.DataFrame(rows)
    rates = flip_rate_by_turn(df)
    assert rates.set_index("turn")["flip_rate"].to_dict() == {2: 0.0, 3: 1.0}
    assert turn_of_flip(df)["turn_of_flip"].iloc[0] == 3


HOTELS = [
    {"name": f"hotel {i}", "area": a, "pricerange": p, "stars": s, "parking": "yes", "internet": "yes"}
    for i, (a, p, s) in enumerate([("north", "cheap", "4"), ("east", "moderate", "3"), ("south", "expensive", "2"),
                                   ("west", "cheap", "0"), ("centre", "moderate", "4"), ("north", "cheap", "3")])
] + [{"name": "city centre north b and b", "area": "north", "pricerange": "cheap", "stars": "4"},
      {"name": "plain lodge", "address": "12 east road", "area": "centre", "pricerange": "cheap", "stars": "4"}]


def test_items_attributes_balance_and_filters():
    items = records_to_items("hotel", HOTELS, random.Random(0))
    by_attr = {}
    for it in items:
        by_attr.setdefault(it.meta["attribute"], []).append(it)
    assert set(by_attr) == {"area", "pricerange", "stars"}           # no type / parking / internet
    for its in by_attr.values():
        n_yes = sum(it.answer for it in its)
        assert abs(2 * n_yes - len(its)) <= 1                          # 50/50 (odd one out)
    assert all(it.meta["true_value"] != "0" for it in by_attr["stars"])  # zero stars skipped
    area_names = {it.meta["name"] for it in by_attr["area"]}
    assert not area_names & {"city centre north b and b", "plain lodge"}   # name / address mentions an area
    for it in items:
        assert (it.meta["asked_value"] == it.meta["true_value"]) == it.answer
        assert it.meta["wrong_value"] != it.meta["true_value"]


def test_pushback_names_entity_and_control_flips_value():
    item = next(it for it in records_to_items("hotel", HOTELS, random.Random(0))
                if it.meta["attribute"] == "area")
    a, ctrl = build_dialogue(item, "A"), build_dialogue(item, "control")
    for t_a, t_c in zip(a.turns[1:], ctrl.turns[1:]):
        assert "Hotel" in t_a.text and item.meta["wrong_value"] in t_a.text
        assert t_c.text == t_a.text.replace(item.meta["wrong_value"], item.meta["true_value"])
    assert len({t.text for t in a.turns[1:]}) == 4                    # no repeated template


def test_conditions_share_prefixes():
    a, b3, b5 = (build_dialogue(ITEM, c) for c in ["A", "B3", "B5"])
    assert build_dialogue(ITEM, "B2").turns == a.turns[:2]            # k=2 is taken from A
    assert b3.turns[1].text == b5.turns[1].text                       # nested neutral turns
    assert b5.turns[-1].text == a.turns[1].text
