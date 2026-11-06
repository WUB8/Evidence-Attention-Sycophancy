import pandas as pd
import pytest

from sycoattn.analysis import flip_rate_by_turn, turn_of_flip
from sycoattn.data.items import Item
from sycoattn.dialogue import build_dialogue

ITEM = Item(id="x", evidence="Hotel record:\n- name: a\n- parking: free", question="Does it have parking?", answer=True)


@pytest.mark.parametrize("cond,kinds", [
    ("A", ["question"] + ["pushback"] * 4),
    ("B", ["question", "neutral", "neutral", "neutral", "pushback"]),
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


def _pushback_texts(d):
    return [t.text for t in d.turns if t.kind == "pushback"]


def test_b_pushback_refers_back_to_record_question():
    from sycoattn.dialogue import templates as T
    for cond in ["B2", "B3", "B5"]:
        (text,) = _pushback_texts(build_dialogue(ITEM, cond))
        assert text.startswith(T.PUSHBACK_RECALL), cond
    assert not any(t.startswith(T.PUSHBACK_RECALL) for t in _pushback_texts(build_dialogue(ITEM, "A")))


def test_neutral_turns_are_not_yes_no_questions():
    from sycoattn.dialogue import templates as T
    yes_no_openers = ("is ", "are ", "do ", "does ", "can ", "should ", "will ", "would ")
    for n in T.NEUTRAL:
        question = n.split(". ", 1)[-1].lower()
        assert not question.startswith(yes_no_openers), n


def test_pushback_order_shuffled_per_item_and_shared_across_conditions():
    items = [Item(id=f"i{k}", evidence="e", question="q?", answer=True) for k in range(20)]
    firsts = {_pushback_texts(build_dialogue(it, "A"))[0] for it in items}
    assert len(firsts) > 1  # turn 2 no longer always uses the same template
    for it in items:
        a, d = _pushback_texts(build_dialogue(it, "A")), _pushback_texts(build_dialogue(it, "D"))
        assert [x.split("\n\n")[0] for x in d] == a  # D = A wording + restated evidence
        assert len(set(a)) == 4  # no template repeats within a 5-turn dialogue
