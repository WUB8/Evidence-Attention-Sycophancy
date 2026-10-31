"""Build yes/no QA items and save them to data/processed/items_<source>.jsonl.

    python scripts/prepare_data.py --config configs/default.yaml
"""

import argparse
from collections import Counter

from sycoattn.config import load_config
from sycoattn.data.items import load_items, save_items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. data.max_items=200")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)["data"]

    kwargs = {"domains": tuple(cfg["domains"])} if cfg["source"] == "multiwoz" else {}
    items = load_items(cfg["source"], max_items=cfg.get("max_items"), seed=cfg.get("seed", 0), **kwargs)
    out = f"data/processed/items_{cfg['source']}.jsonl"
    save_items(items, out)

    print(f"saved {len(items)} items to {out}")
    print("labels:", dict(Counter(it.answer_str for it in items)))
    print("\nexample:\n" + items[0].evidence + "\nQ: " + items[0].question + "  A: " + items[0].answer_str)


if __name__ == "__main__":
    main()
