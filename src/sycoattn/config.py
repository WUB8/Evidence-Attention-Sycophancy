"""YAML config with dotted command-line overrides, e.g. --set model.name=Qwen/Qwen2.5-7B-Instruct."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides or []:
        key, value = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg
