"""Fallback dataset: BoolQ (passage + yes/no question)."""

from __future__ import annotations

from .items import Item


def build_boolq_items(max_items: int | None = None, seed: int = 0, split: str = "validation") -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split=split).shuffle(seed=seed)
    if max_items:
        ds = ds.select(range(min(max_items, len(ds))))
    return [
        Item(
            id=f"boolq-{split}-{i}",
            evidence=ex["passage"],
            question=ex["question"].rstrip("?").capitalize() + "?",
            answer=bool(ex["answer"]),
            source="boolq",
        )
        for i, ex in enumerate(ds)
    ]
