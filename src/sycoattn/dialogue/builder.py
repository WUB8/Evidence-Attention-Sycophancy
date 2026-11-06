"""Build multi-turn dialogues (proposal §6) with labelled spans.

Conditions (default n_turns = 5; turn numbers are 1-based user turns):
  A        evidence + question at t1, pushback at t2..tn             (distance and pressure grow together)
  B / Bk   evidence + question at t1, neutral turns, one pushback at tk
           (k defaults to n; vary k to vary distance at fixed pressure)
  C        setup turn at t1, evidence + question at t2, pushback after (evidence closer, pressure grows)
  D        as A, with the evidence restated in every pushback turn
  control  as A, but the pushback argues for the correct answer

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


def _pushback_order(item: Item) -> list[int]:
    """Per-item shuffled order of pushback templates. Without it, turn t always used template
    t, so turn effects and wording effects were confounded. Seeded by item id, so every
    condition of the same item uses the same order (paired comparisons stay paired)."""
    order = list(range(len(T.PUSHBACK)))
    random.Random(f"pushback-order|{item.id}").shuffle(order)
    return order


def _pushback_turn(claim: str, i: int, agree: bool = False, restate: str | None = None,
                   order: list[int] | None = None, recall: bool = False) -> UserTurn:
    pool = T.AGREE_PUSHBACK if agree else T.PUSHBACK
    idx = order[i % len(order)] if order else i % len(pool)
    text = pool[idx % len(pool)].format(claim=claim)
    if recall:
        text = T.PUSHBACK_RECALL + text
    segs = [Segment(PUSHBACK, text)]
    if restate is not None:
        segs.append(Segment(EVIDENCE_RESTATED, T.EVIDENCE_RESTATE.format(evidence=restate)))
    return UserTurn("pushback", segs)


def build_dialogue(item: Item, condition: str, n_turns: int = 5, system_prompt: str = "",
                   rng: random.Random | None = None) -> Dialogue:
    rng = rng or random.Random(0)
    wrong = "no" if item.answer else "yes"
    neutral = rng.sample(T.NEUTRAL, k=min(len(T.NEUTRAL), n_turns))
    order = _pushback_order(item)
    turns: list[UserTurn] = []
    claim = wrong

    if condition == "A":
        turns = [_question_turn(item)] + [_pushback_turn(wrong, i, order=order) for i in range(n_turns - 1)]
    elif condition.startswith("B"):
        k = int(condition[1:]) if len(condition) > 1 else n_turns
        assert 2 <= k <= n_turns, f"pushback turn must be in [2, {n_turns}]"
        turns = [_question_turn(item)]
        turns += [UserTurn("neutral", [Segment(NEUTRAL, neutral[i])]) for i in range(k - 2)]
        # Same recall prefix for every k (B2 included), so only the distance varies across B_k.
        turns += [_pushback_turn(wrong, 0, order=order, recall=True)]
    elif condition == "C":
        # TODO(team): confirm the C design (how late the evidence goes, how many setup turns).
        turns = [UserTurn("setup", [Segment(SETUP, rng.choice(T.SETUP))]), _question_turn(item)]
        turns += [_pushback_turn(wrong, i, order=order) for i in range(n_turns - 2)]
    elif condition == "D":
        turns = [_question_turn(item)]
        turns += [_pushback_turn(wrong, i, restate=item.evidence, order=order) for i in range(n_turns - 1)]
    elif condition == "control":
        claim = item.answer_str
        turns = [_question_turn(item)]
        turns += [_pushback_turn(claim, i, agree=True, order=order) for i in range(n_turns - 1)]
    else:
        raise ValueError(f"unknown condition: {condition}")

    return Dialogue(item=item, condition=condition, turns=turns,
                    system_prompt=system_prompt, pushback_claim=claim)


def build_dialogues(items: list[Item], conditions: list[str], n_turns: int = 5,
                    system_prompt: str = "", seed: int = 0) -> list[Dialogue]:
    rng = random.Random(seed)
    return [build_dialogue(it, c, n_turns, system_prompt, rng) for it in items for c in conditions]
