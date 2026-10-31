import numpy as np

from sycoattn.attention.shares import attention_shares
from sycoattn.attention.spans import Span


def test_shares_exclude_sink_and_split_pushback():
    # 1 layer, 1 head, 10 tokens. Token 0 is a sink with most of the mass.
    attn = np.array([[[0.5, 0.05, 0.05, 0.1, 0.1, 0.05, 0.05, 0.05, 0.025, 0.025]]])
    spans = [Span("evidence", 0, 1, 3), Span("pushback", 1, 3, 5), Span("pushback", 2, 5, 7)]
    s = attention_shares(attn, spans, current_turn=2, sink_tokens=1)
    assert np.isclose(s["evidence"][0, 0], 0.1 / 0.5)
    assert np.isclose(s["pushback_earlier"][0, 0], 0.2 / 0.5)
    assert np.isclose(s["pushback_current"][0, 0], 0.1 / 0.5)
