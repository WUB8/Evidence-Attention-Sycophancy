import numpy as np

from sycoattn.attention.shares import attention_shares
from sycoattn.attention.spans import Span, render_with_spans
from sycoattn.data.items import Item
from sycoattn.dialogue import build_dialogue


def test_shares_exclude_sink_and_split_pushback():
    # 1 layer, 1 head, 10 tokens. Token 0 is a sink with most of the mass.
    attn = np.array([[[0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]]])
    spans = [Span("evidence", 0, 1, 3), Span("pushback", 1, 3, 5), Span("pushback", 2, 5, 7)]
    s = attention_shares(attn, spans, current_turn=2, sink_tokens=1)
    assert np.isclose(s["evidence"][0, 0], 0.1 / 0.5)
    assert np.isclose(s["pushback_earlier"][0, 0], 0.2 / 0.5)
    assert np.isclose(s["pushback_current"][0, 0], 0.1 / 0.5)


class CharTokenizer:
    """One token per character; the chat template just concatenates the messages."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return "".join(f"<{m['role']}>{m['content']}</s>" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def test_answer_line_span_inside_evidence():
    item = Item(id="x", evidence="Hotel record:\n- name: acorn\n- area: north\n- stars: 4",
                question="Is it in the south part of town?", answer=False,
                meta={"attribute": "area", "name": "acorn", "true_value": "north", "wrong_value": "south"})
    d = build_dialogue(item, "D")
    ids, spans = render_with_spans(CharTokenizer(), d, ["No.", "No."], upto=2)
    text = CharTokenizer().apply_chat_template(d.messages(["No.", "No."], 2))
    line = [s for s in spans if s.label == "answer_line"]
    ev = next(s for s in spans if s.label == "evidence")
    assert len(line) == 1                                   # only the original evidence, not restated copies
    assert text[line[0].start:line[0].end] == "- area: north"
    assert ev.start <= line[0].start and line[0].end <= ev.end
