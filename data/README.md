# Data

Nothing in `raw/` or `processed/` is committed. Regenerate with:

```bash
python scripts/prepare_data.py
```

- `raw/multiwoz/` — MultiWOZ database files (`{hotel,restaurant,attraction}_db.json`), downloaded from
  https://github.com/budzianowski/multiwoz/tree/master/db on first use.
- `processed/items_<source>.jsonl` — yes/no items: `{id, evidence, question, answer, source, meta}`.

BoolQ (fallback) is loaded through Hugging Face `datasets` (`google/boolq`).
