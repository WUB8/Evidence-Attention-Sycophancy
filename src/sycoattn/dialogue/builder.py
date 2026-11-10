"""Build multi-turn dialogues (proposal §6) with labelled spans.

Conditions (default n_turns = 5; turn numbers are 1-based user turns):
  A        evidence + question at t1, pushback at t2..tn             (distance and pressure grow together)
  B / Bk   evidence + question at t1, neutral turns, one pushback at tk
           (k defaults to n; vary k to vary distance at fixed pressure). B2 equals the
           first two turns of A, so we run B3..B5 and take k=2 from A.
  C        setup turn at t1, evidence + question at t2, pushback after (evidence closer, pressure grows)
  D        as A, with the evidence restated in every pushback turn
  control  as A, but the pushback claims the correct value (same templates)

Templates and neutral turns are drawn once per item (seeded by the item id), so all
conditions of an item share them: B3 and B4 use the first neutral turns of B5, and A,
C, D and control use the same pushback sentences in the same order.

Every user turn is a list of labelled segments, so the attention code can find the
evidence, question and pushback tokens after the chat template is applied.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..data.items import Item
from . import templates as T

# Segment labels
EVIDENCE = "evidence"
EVIDENCE_RESTATED = "evidence_restated"
QUESTION = "question"
PUSHBACK = "pushback"
NEUTRAL = "neutral"
SETUP = "setup"


@dataclass
class Segment:
    label: str
    text: str


@dataclass
class UserTurn:
    kind: str                       # question | pushback | neutral | setup
    segments: list[Segment]

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.segments)


@dataclass
class Dialogue:
    item: Item
    condition: str
    turns: list[UserTurn]
    system_prompt: str = ""
    pushback_claim: str = ""        # answer the pushback argues for ("yes"/"no")
    meta: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.item.id}|{self.condition}"

    def messages(self, replies: list[str], upto: int | None = None) -> list[dict]:
        """Chat messages through user turn `upto` (0-based, inclusive), with model
        replies filled in for the earlier turns."""
        upto = len(self.turns) - 1 if upto is None else upto
        msgs = [{"role": "system", "content": self.system_prompt}] if self.system_prompt else []
        for i in range(upto + 1):
            msgs.append({"role": "user", "content": self.turns[i].text})
            if i < upto:
                msgs.append({"role": "assistant", "content": replies[i]})
        return msgs


def _question_turn(item: Item) -> UserTurn:
    # Keep evidence and question as separate segments; the joined text matches T.QUESTION.
    return UserTurn("question", [
        Segment(EVIDENCE, item.evidence),
        Segment(QUESTION, f"Based on this record: {item.question}"),
    ])


def _claims(item: Item, value: str) -> dict[str, str]:
    """Present/past claim clauses that `value` is the item's attribute value."""
    attr = item.meta.get("attribute")
    if attr in T.CLAIM:
        name = T.pretty_name(item.meta["name"])
        present, past = (c.format(name=name, v=T.VALUE_TEXT.get(value, value)) for c in T.CLAIM[attr])
    else:
        present, past = (c.format(v=value) for c in T.GENERIC_CLAIM)
    return {"present": present, "past": past, "Present": present[0].upper() + present[1:]}


def _pushback_turn(template: str, claims: dict[str, str], restate: str | None = None) -> UserTurn:
    segs = [Segment(PUSHBACK, template.format(**claims))]
    if restate is not None:
        segs.append(Segment(EVIDENCE_RESTATED, T.EVIDENCE_RESTATE.format(evidence=restate)))
    return UserTurn("pushback", segs)


def build_dialogue(item: Item, condition: str, n_turns: int = 5, system_prompt: str = "",
                   seed: int = 0) -> Dialogue:
    rng = random.Random(f"{seed}|{item.id}")
    templates = rng.sample(T.PUSHBACK, k=min(len(T.PUSHBACK), n_turns - 1))
    neutral = rng.sample(T.NEUTRAL, k=min(len(T.NEUTRAL), n_turns))
    setup = rng.choice(T.SETUP)
    # value the pushback claims: wrong for A-D, true for control
    if condition == "control":
        claim = item.answer_str
        value = item.meta.get("true_value", claim)
    else:
        claim = "no" if item.answer else "yes"
        value = item.meta.get("wrong_value", claim)
    claims = _claims(item, value)

    if condition in ("A", "control"):
        turns = [_question_turn(item)] + [_pushback_turn(templates[i], claims) for i in range(n_turns - 1)]
    elif condition.startswith("B"):
        k = int(condition[1:]) if len(condition) > 1 else n_turns
        assert 2 <= k <= n_turns, f"pushback turn must be in [2, {n_turns}]"
        turns = [_question_turn(item)]
        turns += [UserTurn("neutral", [Segment(NEUTRAL, neutral[i])]) for i in range(k - 2)]
        turns += [_pushback_turn(templates[0], claims)]
    elif condition == "C":
        turns = [UserTurn("setup", [Segment(SETUP, setup)]), _question_turn(item)]
        turns += [_pushback_turn(templates[i], claims) for i in range(n_turns - 2)]
    elif condition == "D":
        turns = [_question_turn(item)]
        turns += [_pushback_turn(templates[i], claims, restate=item.evidence) for i in range(n_turns - 1)]
    else:
        raise ValueError(f"unknown condition: {condition}")

    return Dialogue(item=item, condition=condition, turns=turns,
                    system_prompt=system_prompt, pushback_claim=claim)


def build_dialogues(items: list[Item], conditions: list[str], n_turns: int = 5,
                    system_prompt: str = "", seed: int = 0) -> list[Dialogue]:
    return [build_dialogue(it, c, n_turns, system_prompt, seed) for it in items for c in conditions]
