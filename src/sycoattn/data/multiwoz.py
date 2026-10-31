"""Yes/no items from MultiWOZ database records.

Each item uses one database record (e.g. a hotel) as the evidence and asks a yes/no
question about one of its attributes. Half the questions are true for the record and
half are false (a different value of the same attribute), so the labels are balanced.
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

# attribute -> question template. "{v}" is the value being asked about.
CATEGORICAL = {
    "hotel": {
        "area": "Is it in the {v} part of town?",
        "pricerange": "Is it in the {v} price range?",
        "type": "Is it a {v}?",
        "stars": "Does it have a {v}-star rating?",
    },
    "restaurant": {
        "area": "Is it in the {v} part of town?",
        "pricerange": "Is it in the {v} price range?",
        "food": "Does it serve {v} food?",
    },
    "attraction": {
        "area": "Is it in the {v} part of town?",
        "type": "Is it a {v}?",
    },
}
# attribute -> question; yes/free count as "yes"
BOOLEAN = {
    "hotel": {
        "parking": "Does it have parking?",
        "internet": "Does it have internet?",
    },
}

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


def records_to_items(domain: str, records: list[dict], rng: random.Random) -> list[Item]:
    values = {a: sorted({str(r[a]) for r in records if a in r}) for a in CATEGORICAL.get(domain, {})}
    items = []
    for rec in records:
        evidence = render_record(domain, rec)
        for attr, tmpl in CATEGORICAL.get(domain, {}).items():
            if attr not in rec or len(values[attr]) < 2:
                continue
            true_v = str(rec[attr])
            truth = rng.random() < 0.5
            v = true_v if truth else rng.choice([x for x in values[attr] if x != true_v])
            items.append(Item(
                id=f"mwz-{domain}-{rec.get('id', rec['name'])}-{attr}",
                evidence=evidence, question=tmpl.format(v=v), answer=truth,
                source="multiwoz", meta={"domain": domain, "attribute": attr, "asked_value": v},
            ))
        for attr, q in BOOLEAN.get(domain, {}).items():
            if attr not in rec:
                continue
            items.append(Item(
                id=f"mwz-{domain}-{rec.get('id', rec['name'])}-{attr}",
                evidence=evidence, question=q, answer=rec[attr] in ("yes", "free"),
                source="multiwoz", meta={"domain": domain, "attribute": attr},
            ))
    return items


def build_multiwoz_items(max_items: int | None = None, seed: int = 0,
                         domains: tuple[str, ...] = ("hotel", "restaurant", "attraction")) -> list[Item]:
    rng = random.Random(seed)
    items = []
    for d in domains:
        items += records_to_items(d, load_db(d), rng)
    rng.shuffle(items)
    return items[:max_items] if max_items else items
