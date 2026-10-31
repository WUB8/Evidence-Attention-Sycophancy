# Evidence Attention and Sycophancy in Multi-turn Conversations

CS6120 final project, Group 8: Mohammad Shahid Mahmood, Bijun Wu, Shiv Patel, Tuyet Tu.

**Question.** When an LLM gives up a correct, evidence-supported answer under user pushback,
is it because its attention drifted away from the evidence (distance), or because pushback
piled up (pressure)? We build 5-turn yes/no QA dialogues that vary distance and pressure
independently, track per-turn attention to the evidence in open-weight models, and relate it to
when the model flips.

## Layout

```
configs/default.yaml        experiment config (model, data, conditions, generation)
src/sycoattn/
  data/                     QA items: MultiWOZ DB records -> yes/no questions; BoolQ fallback
  dialogue/                 dialogue builder (conditions A, B/Bk, C, D, control) + templates
  inference/                turn-by-turn harness (HF Transformers, eager attention) + answer parser
  attention/                span -> token mapping; per-turn attention shares by span
  analysis/                 flip rate per turn, Turn of Flip
scripts/prepare_data.py     build items -> data/processed/
scripts/run_pilot.py        run dialogues, save replies / attention shares / flip rates
scripts/cluster_env.sh      conda env on the cluster
slurm/run_pilot.sbatch      cluster job
tests/                      unit tests (no model download needed)
```

## Setup

Local (macOS / Linux, Python ≥ 3.10):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Cluster:

```bash
source scripts/cluster_env.sh   # creates the conda env the first time
sbatch slurm/run_pilot.sbatch   # Qwen2.5-7B-Instruct pilot on 50 items
```

Llama-3.1-8B-Instruct is gated: accept the license on Hugging Face, then run `huggingface-cli login`.

## Running

```bash
python scripts/prepare_data.py                                  # 50 MultiWOZ items by default
python scripts/run_pilot.py --set data.max_items=3              # smoke test with Qwen2.5-0.5B (~1 GB)
python scripts/run_pilot.py --set model.name=Qwen/Qwen2.5-7B-Instruct data.max_items=50
```

Any config field can be overridden with `--set key.sub=value`. Outputs go to
`outputs/<model>/<run-name>/`:

| file | contents |
|---|---|
| `turns.jsonl` | one row per dialogue turn: user text, reply, parsed answer, gold answer |
| `attention.npz` | `"<item>\|<cond>::<group>"` → `[turns, layers, heads]` attention share |
| `flip_rates.csv` | flip rate per condition and turn (initially-correct dialogues only) |

## Conditions (5 user turns)

| | turns | varies |
|---|---|---|
| **A** | evidence+question, then 4 pushbacks | distance and pressure together |
| **B / Bk** | evidence+question, neutral turns, 1 pushback at turn k | distance (pressure fixed) |
| **C** | setup, evidence+question, then 3 pushbacks | pressure (evidence closer) |
| **D** | as A, evidence restated in every pushback | restating intervention |
| **control** | as A, pushback toward the correct answer | rules out random changes |

## Attention measurement

At each turn we run the prompt through the model and take the attention row of the **last prompt
token** (the position that produces the first answer token) in every layer and head. We drop
attention-sink tokens (`attention.sink_tokens`), renormalise, and sum over span groups:
`evidence`, `evidence_restated`, `question`, `pushback_current`, `pushback_earlier`,
`own_answers`, `neutral`, `setup`. The prefix runs with a KV cache and only the last token
requests attentions, so memory stays small even for 8B models.

## Answer parsing

The system prompt asks the model to end with `Final answer: Yes/No`, and the parser uses that
line first. Don't trust a leading "Yes" after pushback: replies like *"Yes, … it is not in the
north"* agree with the user rather than answer the question.

## Team workflow

`main` should always run. Work on a branch and merge through a pull request.

```bash
git checkout main && git pull                 # start from the latest main
git checkout -b mohammad/dialogue-templates   # one branch per task: <name>/<task>
# ... edit, then check nothing broke:
pytest
git add -A && git commit -m "Add 50 neutral follow-up turns"
git push -u origin mohammad/dialogue-templates
```

Then open a pull request on GitHub, ask one teammate to review, and merge once it's approved.

- Keep pull requests small: one task from the timeline each.
- Don't commit data, model weights or `outputs/`; `.gitignore` already excludes them.
  Share results by uploading a run's `flip_rates.csv` or a short summary to the PR.
- Put new experiment settings in `configs/` instead of editing defaults in code.
- If `main` moved while you worked: `git checkout main && git pull`, then
  `git checkout <your-branch> && git merge main`, fix any conflicts, and push again.

## Status / TODO (see proposal timeline)

- [x] Environment, MultiWOZ item builder, dialogue builder, harness, parser, attention shares, flip metrics
- [ ] Expand and validate neutral turns (~150) and pushback templates — `dialogue/templates.py`
- [ ] Confirm condition C design — `dialogue/builder.py`
- [ ] Hand-check ~200 parsed answers; extend the parser
- [ ] Item filtering (drop items the model gets wrong at turn 1)
- [ ] Linear probe for sycophancy heads (held-out split)
- [ ] Attention-masking intervention (RQ4)
- [ ] Per-turn logistic model (statsmodels) for RQ2
