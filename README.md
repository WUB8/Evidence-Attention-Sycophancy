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

Cluster: see [Running on Explorer](#running-on-explorer) below.

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

## Running on Explorer

Northeastern's Explorer cluster, course account `cs6120.202710`.

**Log in.** Since 2026-11-06 Explorer only accepts the SSH key Research Computing generated
for you. Download the two key files from Open OnDemand (Files → Home Directory → `.ssh`), move
them into `~/.ssh/`, `chmod 600` the private one, and add to `~/.ssh/config`:

```
Host explorer
    HostName login.explorer.northeastern.edu
    User <your-username>
    IdentityFile ~/.ssh/<your-username>-<date>
```

Then `ssh explorer`. Details: https://rc.northeastern.edu/2026/10/23/ssh-instructions-for-explorer/

**First time only.** Clone the repo, then build the environment on a compute node (not the
login node, which is for light work only):

```bash
git clone https://github.com/WUB8/Evidence-Attention-Sycophancy.git && cd Evidence-Attention-Sycophancy
srun -p short -A cs6120.202710 --mem=8G -t 01:00:00 --pty /bin/bash
source scripts/cluster_env.sh       # ~10 min the first time; later it only activates
hf auth login                       # only for gated models (Llama); accept the license on HF first
exit
```

**Submit a run** from the repo root. `logs/` must exist before `sbatch`, because Slurm opens
the log file before the script runs:

```bash
mkdir -p logs
MODEL=Qwen/Qwen2.5-14B-Instruct CONDITIONS='[A,C,B2,B5]' sbatch --export=ALL slurm/run_pilot.sbatch
```

Set `MODEL`, `CONDITIONS` and `ITEMS` (default 50) in front of `sbatch` as shown; putting them
inside `--export=` breaks the commas in `[A,C,B2,B5]`. Without them the job runs
Qwen2.5-7B-Instruct on all five conditions.

**Check on it:**

```bash
squeue -u $USER --start                           # PENDING (Priority/Resources) or RUNNING
tail -f logs/sycoattn-pilot-<jobid>.out           # progress bar, errors
sacct -j <jobid> -X -o JobID,State,ExitCode,Elapsed   # after it ends
```

Results go to `outputs/<model>/pilot-<jobid>/`. Copy what you need to your laptop with
`scp -r explorer:Evidence-Attention-Sycophancy/outputs/<model>/pilot-<jobid> outputs/<model>/`.

**What the job asks for.** `gpu` partition, one H200 (141 GB), 64 GB RAM, 4 h. The `gpu`
partition allows one GPU per job and at most 8 h (`sbatch -t 08:00:00 ...` for slow models).
Models up to ~30-40B fit in bf16 on one H200; 70B does not. To use an A100 80 GB instead:
`sbatch --gres=gpu:a100:1 --constraint=a100@80g ...`. See free GPUs with
`sinfo -p gpu -O "Gres:22,GresUsed:28,StateCompact:8,NodeList:16"`.

**Where things live.** Everything big goes in `/scratch/$USER` (home is capped at 75 GB):
the conda env (`conda_envs/sycoattn`, ~6 GB), pip/conda caches and the Hugging Face cache
(`hf_cache`, ~30 GB for a 14B model, including your HF login token). Scratch is purged
monthly and not backed up: a missing env or model is rebuilt / re-downloaded by the next job,
but copy results you want to keep elsewhere. To delete a big folder quickly, rename it first,
then `rm -rf` it; Open OnDemand cannot delete non-empty folders efficiently.

**Gotchas we hit**
- The PyPI torch wheel targets CUDA 13, but Explorer's driver supports CUDA ≤ 12.8, so
  `cluster_env.sh` installs `torch==2.14.1` from PyTorch's cu126 index. The job stops with
  "torch cannot see the GPU" if this ever breaks; check `nvidia-smi` for the driver's CUDA version.
- If you change the torch pin, delete `/scratch/$USER/conda_envs/sycoattn` so the next job rebuilds it.
- After a run, check how many replies have an empty `parsed` in `turns.jsonl`: scoring relies on
  the model writing "Final answer: Yes/No", and models differ in how well they follow it.

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
