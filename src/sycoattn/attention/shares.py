"""Per-turn attention shares by span (proposal §7).

We take the attention row of the last prompt token (the position that produces the
first answer token) in every layer and head, drop attention-sink tokens, renormalise,
and sum over each span group.
"""

from __future__ import annotations

import numpy as np
import torch

from .spans import OWN_ANSWER, Span

# Span groups reported per turn. "current" = the turn being answered.
GROUPS = ("evidence", "evidence_restated", "question", "pushback_current",
          "pushback_earlier", "own_answers", "neutral", "setup")


@torch.no_grad()
def last_token_attention(model, input_ids: list[int]) -> np.ndarray:
    """Attention from the last prompt token to every position: array [layers, heads, seq].

    Runs the prefix with a KV cache, then only the last token with output_attentions,
    so we never hold the full [seq, seq] attention maps for every layer.
    Requires the model to be loaded with attn_implementation="eager".
    """
    ids = torch.tensor([input_ids], device=model.device)
    prefix = model(ids[:, :-1], use_cache=True)
    out = model(ids[:, -1:], past_key_values=prefix.past_key_values, output_attentions=True)
    return torch.stack([a[0, :, -1, :] for a in out.attentions]).float().cpu().numpy()


def group_of(span: Span, current_turn: int) -> str:
    if span.label == "pushback":
        return "pushback_current" if span.turn == current_turn else "pushback_earlier"
    if span.label == OWN_ANSWER:
        return "own_answers"
    return span.label


def attention_shares(attn: np.ndarray, spans: list[Span], current_turn: int,
                     sink_tokens: int = 1) -> dict[str, np.ndarray]:
    """attn: [layers, heads, seq]. Returns {group: [layers, heads]} shares in [0, 1]."""
    a = attn.copy()
    a[..., :sink_tokens] = 0.0
    a /= a.sum(-1, keepdims=True).clip(min=1e-12)
    shares = {g: np.zeros(a.shape[:2], dtype=np.float32) for g in GROUPS}
    for s in spans:
        lo = max(s.start, sink_tokens)
        if lo < s.end:
            shares[group_of(s, current_turn)] += a[..., lo:s.end].sum(-1)
    return shares
