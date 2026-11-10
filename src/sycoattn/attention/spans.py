"""Map labelled dialogue segments to token positions after the chat template is applied."""

from __future__ import annotations

from dataclasses import dataclass

from ..dialogue.builder import Dialogue

OWN_ANSWER = "own_answer"
ANSWER_LINE = "answer_line"   # the record line that holds the answer, e.g. "- area: centre"


@dataclass
class Span:
    label: str
    turn: int          # 0-based user turn the span belongs to
    start: int         # token index, inclusive
    end: int           # token index, exclusive


def render_with_spans(tokenizer, dialogue: Dialogue, replies: list[str], upto: int):
    """Render the dialogue through user turn `upto` with a generation prompt.

    Returns (input_ids: list[int], spans: list[Span]). Spans cover every user segment
    and every earlier assistant reply (label "own_answer"). For MultiWOZ items, the line of
    the original evidence that holds the answer gets an extra span ("answer_line"), which
    lies inside the evidence span.
    """
    msgs = dialogue.messages(replies, upto)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]

    # Pieces in the order they appear in the rendered text.
    pieces = []
    for t in range(upto + 1):
        for seg in dialogue.turns[t].segments:
            pieces.append((seg.label, t, seg.text))
        if t < upto:
            pieces.append((OWN_ANSWER, t, replies[t].strip()))

    def token_span(label, t, start, end):
        toks = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        return Span(label, t, toks[0], toks[-1] + 1)

    meta = dialogue.item.meta
    answer_line = f"- {meta['attribute']}: {meta['true_value']}" if "true_value" in meta else None

    spans, cursor = [], 0
    for label, t, piece in pieces:
        if not piece:
            continue
        start = text.find(piece, cursor)
        if start < 0:
            raise ValueError(f"segment not found in rendered chat: {piece[:60]!r}")
        end = start + len(piece)
        cursor = end
        spans.append(token_span(label, t, start, end))
        if label == "evidence" and answer_line and answer_line in piece:
            line_start = start + piece.index(answer_line)
            spans.append(token_span(ANSWER_LINE, t, line_start, line_start + len(answer_line)))
    return enc["input_ids"], spans
