"""Run the dialogues turn by turn, saving replies, parsed answers and attention shares.

    python scripts/run_pilot.py --config configs/default.yaml --set data.max_items=5

Outputs in <output_dir>/<model>/<run_name>/:
  turns.jsonl       one row per (dialogue, turn)
  attention.npz     {dialogue_id::group: [turns, layers, heads]}
  flip_rates.csv    flip rate per condition and turn
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sycoattn.analysis import flip_rate_by_turn
from sycoattn.config import load_config
from sycoattn.data.items import load_items
from sycoattn.dialogue import build_dialogues
from sycoattn.inference.harness import load_model, run_dialogue, stack_shares


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.name=Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--no-attention", action="store_true")
    ap.add_argument("--run-name", default=time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)

    d = cfg["data"]
    kwargs = ({"domains": tuple(d["domains"]), "split": d.get("split", "main")}
              if d["source"] == "multiwoz" else {})
    items = load_items(d["source"], max_items=d.get("max_items"), seed=d.get("seed", 0), **kwargs)
    dialogues = build_dialogues(items, cfg["dialogue"]["conditions"], cfg["dialogue"]["n_turns"],
                                cfg["dialogue"]["system_prompt"].strip(), seed=d.get("seed", 0))

    model, tok = load_model(cfg["model"]["name"], cfg["model"]["dtype"], cfg["model"]["device"])
    out_dir = Path(cfg["output_dir"]) / cfg["model"]["name"].replace("/", "__") / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"{len(items)} items x {len(cfg['dialogue']['conditions'])} conditions "
          f"= {len(dialogues)} dialogues on {model.device} -> {out_dir}")

    records, attn = [], {}
    with (out_dir / "turns.jsonl").open("w") as f:
        for dlg in tqdm(dialogues):
            recs, shares = run_dialogue(
                model, tok, dlg,
                max_new_tokens=cfg["generation"]["max_new_tokens"],
                do_sample=cfg["generation"]["do_sample"],
                with_attention=not args.no_attention,
                sink_tokens=cfg["attention"]["sink_tokens"],
            )
            for r in recs:
                f.write(json.dumps(r) + "\n")
            records += recs
            for g, arr in stack_shares(shares).items():
                attn[f"{dlg.id}::{g}"] = arr.astype(np.float16)

    if attn:
        np.savez_compressed(out_dir / "attention.npz", **attn)
    rates = flip_rate_by_turn(pd.DataFrame(records))
    rates.to_csv(out_dir / "flip_rates.csv", index=False)
    print(rates.to_string(index=False))


if __name__ == "__main__":
    main()
