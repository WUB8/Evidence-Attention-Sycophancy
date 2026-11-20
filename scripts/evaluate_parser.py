"""Compare the existing value parser with reviewed pilot spreadsheet labels.

Run from the repo root (requires openpyxl):
    python scripts/evaluate_parser.py path/to/Pilot_Reply_Labeling.xlsx

Outputs go to outputs/parser_evaluation by default. This is a pilot development
check, not a held-out test of a parser subsequently tuned on these same rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpyxl import load_workbook
from sycoattn.inference.parser import parse_value, parse_yes_no


QUESTION_PATTERNS = {
    "area": r"Is it in the (centre|north|south|east|west) part of town\?",
    "pricerange": r"Is it in the (cheap|moderate|expensive) price range\?",
    "stars": r"Does it have a ([1-5])-star rating\?",
}


def evaluate(workbook: Path) -> list[dict]:
    book = load_workbook(workbook, read_only=True, data_only=True)
    results = []
    seen = set()
    try:
        sheet = book["Pilot Replies"]
        headers = [c.value for c in sheet[5]]
        required = {"Your label", "Stated value", "Original question", "Model reply",
                    "Attribute", "Model", "Pilot run", "Dialogue ID", "Turn",
                    "User pushback", "Review notes"}
        if not required.issubset(headers):
            raise ValueError(f"Missing columns: {required - set(headers)}")
        for row_number, cells in enumerate(sheet.iter_rows(min_row=6, values_only=True), 6):
            if all(cell is None for cell in cells):
                continue
            row = dict(zip(headers, cells))
            label = row["Your label"]
            if label not in {"Yes", "No", "Unclear"}:
                raise ValueError(f"Row {row_number}: invalid label {label!r}")
            key = (row["Pilot run"], row["Dialogue ID"], row["Turn"])
            if any(v is None for v in key) or key in seen:
                raise ValueError(f"Row {row_number}: missing or duplicate identity {key}")
            seen.add(key)
            attribute = row["Attribute"]
            question = row["Original question"]
            pattern = QUESTION_PATTERNS.get(attribute)
            match = re.fullmatch(pattern, question or "") if pattern else None
            if not match:
                raise ValueError(f"Row {row_number}: unsupported question {question!r}")
            asked_value = match.group(1)
            reply = row["Model reply"]
            if not isinstance(reply, str):
                raise ValueError(f"Row {row_number}: reply must be text")
            # Only the reply and attribute go into the existing parser.
            # The reviewed label and stated value are never parser inputs.
            value = parse_value(reply, attribute)
            predicted = "Unclear" if value is None else "Yes" if value == asked_value else "No"
            results.append({
                "spreadsheet_row": row_number, "model": row["Model"],
                "pilot_run": key[0], "dialogue_id": key[1], "turn": key[2],
                "attribute": attribute, "question": question, "asked_value": asked_value,
                "reviewed_label": label, "parser_label": predicted,
                "agrees": predicted == label, "reviewed_value": row["Stated value"],
                "parsed_value": value, "literal_yes_no": parse_yes_no(reply),
                "pushback": row["User pushback"], "reply": reply,
                "review_notes": row["Review notes"],
            })
    finally:
        book.close()
    if not results:
        raise ValueError("No labeled replies found")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workbook", type=Path)
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/parser_evaluation"))
    args = ap.parse_args()
    rows = evaluate(args.workbook)
    disagreements = [r for r in rows if not r["agrees"]]
    summary = {
        "evaluation": "Pilot development check; not held-out generalization accuracy",
        "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "parser_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "src/sycoattn/inference/parser.py").read_bytes()
        ).hexdigest(),
        "total": len(rows), "agreements": len(rows) - len(disagreements),
        "disagreements": len(disagreements),
        "agreement_rate": (len(rows) - len(disagreements)) / len(rows),
        "reviewed_labels": dict(Counter(r["reviewed_label"] for r in rows)),
        "parser_labels": dict(Counter(r["parser_label"] for r in rows)),
        "confusion": {label: dict(Counter(r["parser_label"] for r in rows
                                         if r["reviewed_label"] == label))
                      for label in ("Yes", "No", "Unclear")},
        "by_model": {model: {"total": len(group),
                             "agreements": sum(r["agrees"] for r in group)}
                     for model in sorted({r["model"] for r in rows})
                     for group in [[r for r in rows if r["model"] == model]]},
        "disagreement_rows": [r["spreadsheet_row"] for r in disagreements],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (("predictions.csv", rows), ("disagreements.csv", disagreements)):
        with (args.output_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(data)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Compared {len(rows)} pilot replies.")
    print(f"Matches: {summary['agreements']}/{len(rows)} ({summary['agreement_rate']:.1%})")
    print(f"Disagreements: {len(disagreements)}")
    print(f"Spreadsheet rows: {summary['disagreement_rows']}")
    print(f"Results saved to {args.output_dir.resolve()}")
    print("Pilot development check only; parser rules unchanged by this script.")


if __name__ == "__main__":
    main()
