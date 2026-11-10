"""Yes/no items from MultiWOZ database records.

Each item uses one database record (e.g. a hotel) as the evidence and asks a yes/no
question about one of its attributes. Exactly half the questions for each (domain,
attribute) are true for the record and half are false (a different value of the same
attribute), within each split.

Design choices (see reports/writeup):
- Only clear multi-valued attributes are asked about. Dropped: hotel internet/parking
  (97% / 88% "yes"), hotel and attraction type (overlapping or name-revealing labels),
  restaurant food (23 overlapping cuisines). Dropped fields stay in the evidence.
- Zero-star hotels are not asked about stars ("0 stars" is ambiguous).
- An item is dropped if the record's name or address contains a value of the asked
  attribute (e.g. "pizza hut city centre", "regent street city centre" for area): the
  answer would then sit outside its own line, or the text would contradict it
  ("east road" for a place in the centre).
- Records are split into a pilot set (for tuning templates, prompts, the parser) and a
  main set (reported results), stratified by domain.
"""

from __future__ import annotations

import json
import random
import ssl
import urllib.request
from pathlib import Path

import certifi

from .items import Item

DB_URL = "https://raw.githubusercontent.com/budzianowski/multiwoz/master/db/{domain}_db.json"
RAW_DIR = Path("data/raw/multiwoz")
DOMAINS = ("hotel", "restaurant", "attraction")

# attribute -> question template. "{v}" is the value being asked about.
CATEGORICAL = {
    "hotel": {
        "area": "Is it in the {v} part of town?",
        "pricerange": "Is it in the {v} price range?",
        "stars": "Does it have a {v}-star rating?",
    },
    "restaurant": {
        "area": "Is it in the {v} part of town?",
        "pricerange": "Is it in the {v} price range?",
    },
    "attraction": {
        "area": "Is it in the {v} part of town?",
    },
}
# values never asked about (and never used as the wrong value)
SKIP_VALUES = {"stars": {"0"}}

N_PILOT_RECORDS = 40   # pilot records, split across domains in proportion to their size

EVIDENCE_FIELDS = {
    "hotel": ["name", "type", "area", "pricerange", "stars", "parking", "internet", "address", "phone"],
    "restaurant": ["name", "food", "area", "pricerange", "address", "phone"],
    "attraction": ["name", "type", "area", "address", "phone"],
}


def download_db(domain: str, raw_dir: Path = RAW_DIR) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{domain}_db.json"
    if not path.exists():
        # certifi's CA bundle: python.org builds on macOS ship without system certs.
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DB_URL.format(domain=domain), context=ctx) as r:
            path.write_bytes(r.read())
    return path


def load_db(domain: str, raw_dir: Path = RAW_DIR) -> list[dict]:
    return json.loads(download_db(domain, raw_dir).read_text())


def render_record(domain: str, rec: dict) -> str:
    lines = [f"{domain.capitalize()} record:"]
    for k in EVIDENCE_FIELDS[domain]:
        if k in rec and rec[k] not in ("", "?", None):
            lines.append(f"- {k}: {rec[k]}")
    return "\n".join(lines)


def text_reveals(rec: dict, values: set[str]) -> bool:
    """True if the record's name or address contains one of the attribute's values as a word."""
    text = f"{rec.get('name', '')} {rec.get('address', '')}".lower().replace(",", " ")
    return bool(set(text.split()) & values)


def split_records(records_by_domain: dict[str, list[dict]], n_pilot: int,
                  rng: random.Random) -> dict[str, set[str]]:
    """Record-level pilot/main split, stratified by domain. Returns {domain: pilot names}."""
    total = sum(len(r) for r in records_by_domain.values())
    pilot = {}
    for d, recs in records_by_domain.items():
        k = round(n_pilot * len(recs) / total)
        pilot[d] = {r["name"] for r in rng.sample(recs, k)}
    return pilot


def records_to_items(domain: str, records: list[dict], rng: random.Random,
                     split: str = "all") -> list[Item]:
    """Yes/no items for one domain's records, balanced 50/50 per attribute."""
    items = []
    for attr, tmpl in CATEGORICAL.get(domain, {}).items():
        skip = SKIP_VALUES.get(attr, set())
        values = sorted({str(r[attr]) for r in records if attr in r} - skip)
        eligible = [r for r in records if attr in r and str(r[attr]) not in skip
                    and not text_reveals(r, set(values))]
        # exactly half "yes" (the odd one out is random)
        rng.shuffle(eligible)
        n_yes = len(eligible) // 2 + (rng.random() < 0.5 if len(eligible) % 2 else 0)
        for i, rec in enumerate(eligible):
            true_v = str(rec[attr])
            truth = i < n_yes
            others = [x for x in values if x != true_v]
            asked = true_v if truth else rng.choice(others)
            items.append(Item(
                id=f"mwz-{domain}-{rec.get('id', rec['name'])}-{attr}",
                evidence=render_record(domain, rec), question=tmpl.format(v=asked), answer=truth,
                source="multiwoz",
                meta={"domain": domain, "attribute": attr, "name": rec["name"], "split": split,
                      "asked_value": asked, "true_value": true_v,
                      # value the pushback claims: the asked value for "no" items,
                      # some other value for "yes" items
                      "wrong_value": rng.choice(others) if truth else asked},
            ))
    return items


def build_multiwoz_items(max_items: int | None = None, seed: int = 0,
                         domains: tuple[str, ...] = DOMAINS, split: str = "main") -> list[Item]:
    """split: "main" (reported runs), "pilot" (tuning only) or "all"."""
    rng = random.Random(seed)
    # the split always uses all three domains, so it doesn't change with `domains`
    records = {d: load_db(d) for d in DOMAINS}
    pilot = split_records(records, N_PILOT_RECORDS, rng)
    items = []
    for d in domains:
        groups = {"pilot": [r for r in records[d] if r["name"] in pilot[d]],
                  "main": [r for r in records[d] if r["name"] not in pilot[d]]}
        for name, recs in groups.items():
            if split in (name, "all"):
                items += records_to_items(d, recs, rng, split=name)
    rng.shuffle(items)
    return items[:max_items] if max_items else items
