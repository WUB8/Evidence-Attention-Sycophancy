# Pilot parser baseline

Compared the unchanged value parser from commit `5238e84` with the reviewed
200-reply `Pilot_Reply_Labeling.xlsx` using `scripts/evaluate_parser.py`.

- Agreement: 191/200 (95.5%).
- Reviewed labels: 81 Yes, 116 No, 3 Unclear.
- Disagreements: spreadsheet rows 13, 22, 25, 28, 67, 109, 121, 124, 139.

The parser receives only reply text and the attribute. Its extracted value is
compared with the value asked about in the original question to produce Yes or No.
An unparsed value maps to Unclear. Literal `Final answer` Yes/No is saved only as a
diagnostic because it can refer to the pushback instead of the original question.

## What needs investigation

| Rows | Observed difference |
| --- | --- |
| 13, 67, 139 | Prose and Value line conflict. Reviewed label is Unclear, but the parser trusts the Value line. |
| 22, 28, 109, 124 | Value line contains a monetary range. Parser returns Unclear even though prose states the price category. |
| 25 | Reply describes the record as expensive, then offers to find a cheap alternative. Parser treats both categories as candidate answers. |
| 121 | Reply says luxury and gives a monetary range. Reviewed answer to the cheap question is No; luxury is outside the parser's canonical categories. |

## Revised parser

After the rule changes, agreement on the same 200 replies is 200/200 (100%).
All 3 reviewed Unclear replies remain Unclear. The 49 parser tests pass, including
21 new cases covering contradictions, price amounts, reported claims, questions,
conditional alternatives, proximity, and the literal luxury category.

The changes check Value lines against asserted prose, fall back from monetary
price lines to explicit prose categories, and exclude several common forms of
non-answer mentions. Luxury is preserved as a distinct stated category. Amounts
alone are not mapped to price categories. No row IDs, entity names, or particular
monetary ranges are hard-coded in the parsing rules.

The original outputs are preserved in `outputs/parser_evaluation/baseline/` and
the revised outputs are in `outputs/parser_evaluation/revised/`. The spreadsheet
labels were not changed. This is still a rule-based heuristic: unusual wording,
negation, or references to different entities can remain difficult. These changes
have not yet been applied to the complete main runs.

This is a development sample, with approximately equal sampling across three
models. The aggregate is not a frequency-weighted estimate for the main runs.
Any improvement after tuning on these rows is development agreement, not an
independent held-out test. Flip rates have not been recomputed by this evaluation.

The full local row-level outputs are in `outputs/parser_evaluation/` (git-ignored).
The JSON summary includes workbook and parser SHA-256 hashes for reproducibility.
