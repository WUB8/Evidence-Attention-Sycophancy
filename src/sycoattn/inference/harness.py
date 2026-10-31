"""Turn-by-turn inference: generate a reply at every user turn, parse it, and
(optionally) record attention shares at that turn."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..attention.shares import attention_shares, last_token_attention
from ..attention.spans import render_with_spans
from ..dialogue.builder import Dialogue
from .parser import parse_yes_no


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(name: str, dtype: str = "bfloat16", device: str = "auto"):
    device = pick_device(device)
    torch_dtype = torch.float32 if device == "cpu" else getattr(torch, dtype)
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch_dtype, attn_implementation="eager",
        device_map=device if device == "cuda" else None,
    )
    if device != "cuda":
        model.to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def generate(model, tok, input_ids: list[int], max_new_tokens: int = 64, do_sample: bool = False) -> str:
    ids = torch.tensor([input_ids], device=model.device)
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                         do_sample=do_sample, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_dialogue(model, tok, dialogue: Dialogue, max_new_tokens: int = 64, do_sample: bool = False,
                 with_attention: bool = True, sink_tokens: int = 1):
    """Returns (records, shares). records: one dict per turn. shares: one
    {group: [layers, heads]} dict per turn (empty list if with_attention=False)."""
    replies: list[str] = []
    records, shares = [], []
    for t, turn in enumerate(dialogue.turns):
        input_ids, spans = render_with_spans(tok, dialogue, replies, upto=t)
        if with_attention:
            attn = last_token_attention(model, input_ids)
            shares.append(attention_shares(attn, spans, current_turn=t, sink_tokens=sink_tokens))
        reply = generate(model, tok, input_ids, max_new_tokens, do_sample)
        replies.append(reply)
        records.append({
            "dialogue_id": dialogue.id,
            "item_id": dialogue.item.id,
            "condition": dialogue.condition,
            "turn": t + 1,
            "kind": turn.kind,
            "user": turn.text,
            "reply": reply,
            "parsed": parse_yes_no(reply) if turn.kind in ("question", "pushback") else None,
            "gold": dialogue.item.answer_str,
            "pushback_claim": dialogue.pushback_claim,
            "n_tokens": len(input_ids),
        })
    return records, shares


def stack_shares(shares: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """List over turns of {group: [L, H]} -> {group: [turns, L, H]}."""
    return {g: np.stack([s[g] for s in shares]) for g in shares[0]} if shares else {}
