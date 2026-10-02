# HANDOFF: step-budget curve for distillation (warmup-stable-decay)

Branch `worktree-distill-wsd`, worktree `.claude/worktrees/distill-wsd`, cut from `main` at `c3f7499` on 2026-10-02.
**No code is written yet.** This file is the vision and the starting map. Delete it when the work lands (as the
stage-5 grid agent did with its HANDOFF).

## A job is running: do not disturb it

A rung-C distillation is training **from the main checkout** (`/home/luke/projects/buzzdetect-training`), launched with
`tools/launch_job.sh` (pid 1603211 when this was written; log `05_distill/data/main_rungC.log`):

    main.py --teacher v4-ft-ps-e60-moderate --rung C --steps 28000 \
      --runs "yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human fast32h16:a0.50:classes=ins_buzz+ambient_rain+human"

It re-reads `05_distill/*.py` at every stage, so **never edit or merge into the main checkout's `05_distill/` while it
runs** (05_distill/CLAUDE.md, "Never edit scripts here while a chain runs"). Develop here, in this worktree. The 4 GB GPU
fits one job: until the run finishes, test on CPU (`CUDA_VISIBLE_DEVICES=''`, niced, tiny `--steps`, the `test_` prefix).
Merge to main only after it has finished (check `ps`, bracket the pattern: `pgrep -af "[d]istill_train.py"`). Its two
runs land as `fe_C_yamnet_a0.50_s1_select_c-buzz-rain-human` and `fe_C_fast32h16_a0.50_s1_c-buzz-rain-human`, 28,000
steps, ordinary cosine schedule. They are one data point for the question below, not the answer.

## The question

Every distillation run uses a fixed 7,000 steps at batch 512 with a cosine learning-rate decay, and the schedule is a
function of the total step count. Two consequences:

1. **The ladder was confounded.** Rungs A, B and C (about 50, 200, 800 h) got about 19, 5 and 1.2 passes over their
   data. "B to C did not advance" (headline 0.625 to 0.608) therefore says little about data. See `LADDER.md`,
   `FRONTENDS.md`.
2. **No run shows whether 7k steps is too few or too many.** Held-out error (`mae_live` in `curve.json`'s `val`) is still
   falling at step 7,000 in every run (about 2-4% over the last 1,000 steps), but cosine forces that, so it is not proof of
   undertraining. Training loss also rises with rung at a fixed budget (0.105, 0.112, 0.119 for A, B, C), which is the
   undertraining signature. Nothing has ever run past 7,000 steps except the C runs above.

Classic early stopping does not fit: there is no overfitting signal (unlabeled audio, teacher-logit targets), the cosine
schedule makes "stop early" mean "stop at a high learning rate", and the real metric (`sensitivity_exclquiet` at fpr 0.005 over the
5 rotating folds) is noisy (about 0.01-0.02 per run) and costly. Bracketing fixed schedules (7k, 28k, 70k) answers it
but costs 5-10 h per model at 70k.

## The vision: warmup-stable-decay (WSD)

Train one run at a **constant learning rate** after warmup, checkpoint at chosen steps (for example 7k, 14k, 28k, 70k),
and from each checkpoint run a **short decay** to zero (about 10-15% of the steps so far) on a copy. Each decayed
branch is a properly annealed model for that budget, so the branches give the whole steps-versus-held-out curve for
about the cost of one long run plus a few short decays. Read the knee off the curve. A stopping rule falls out: stop
the stable phase when two successive decayed branches agree within noise.

Also budget in **passes over the data, not steps**, so rungs compare fairly: `--epochs` (or a `--passes` helper that
converts with the shard's frame count; `distill_train.py` already prints `steps x batch / total = N passes`).

Open design choices, yours to make (state them in the commit and `README.md`):

- Branch evaluation: the cheap held-out proxies (`mae_live`, buzz lost/gained on V) every branch; the full headline
  (`export` then `eval_folds.py`) only for the branches worth the cost. Do not stop on the noisy headline alone.
- Whether branches are separate runs under derived names (`<name>_wsd<step>`) or one run dir with several `curve`
  entries. Separate runs fit the existing export/eval/record stages with no changes.
- Warmup length and the stable learning rate. 1e-3 is the cosine peak today; WSD often tolerates a bit more, but
  verify on one student before trusting it.
- Optimizer state at a branch: copy the checkpoint's Adam slots (as a resume does) rather than resetting.

## Where to start

- `05_distill/distill_train.py`:
  - schedule: `CosineDecay(a.lr, a.steps, alpha=0.01)` (about line 396);
  - checkpointing: `save_ckpt` / `load_resume_state` (about lines 254-290), which writes every `--eval-every` steps;
  - `RESUME_KEYS` (about line 246) is how a rerun is matched to a run dir, so a new `--schedule`/`--decay-from` option
    must go in it or a mix-up will resume wrongly;
  - the loop writes `curve['val']` entries every eval, and a `final` one at the end.
- `05_distill/main.py`: `--steps`, the run-spec grammar (`<fe>:<arch>[:<init>][:classes=..][:lam=..]`), and `run_name`.
  A schedule option has to reach `distill_train.py` and be recorded.
- `05_distill/ladder_record.py record`: the ladder row has `steps` but no schedule field; add one so WSD rows are never
  mixed with cosine rows in `frontier` tables (`tools/human/frontier_svg.py` filters to rung B seed 1 and would
  otherwise plot them as ordinary points).
- Result artifacts: `05_distill/data/<teacher>/{runs,models,eval,ladder.jsonl}`. Per-run `curve.json`.

## What I already know (so you need not redo it)

- Per-run curves exist for all 47 rung-B students. On the headline, A to B went 0.51 to 0.625; the A repeat spread on
  `lost_pct` is 1.3 points; headline seed noise is about 0.01-0.02.
- Narrower trunks fit worse (YAMNet front end, buzz only: final training loss 0.119 at `a0.25`, 0.106 at `a0.50`), so
  more data helps the `a0.50` students more than the `a0.25` ones; steps help both.
- Pool sizes (after the 2026-10-02 replan that blacklisted the SeeNote test set): A 3,026 slices (50.0 h), B 12,099
  (200.0 h), C 48,395 (800.0 h), D 168,073 (2,778.5 h); V 6,056. Slices start once per hour of each file, so D is about
  1/60 of the audio on the drive: data supply is not the limit, disk is (rung C pack 46 GB).
- Speed: about 2.4 steps/s on the 4 GB card for an `a0.50` student; `wall.txt` per run has the rest.

## Verification plan

1. CPU smoke test on a tiny budget with the `test_` prefix: a WSD run plus one branch produces a `curve.json` and a
   resumable checkpoint; kill mid-stable-phase and rerun to check the resume.
2. Equivalence check: a WSD run decayed at step N should land near a cosine run of N steps (same student, same seed)
   on `mae_live` and `lost_pct`; if it is clearly worse, the stable learning rate is wrong.
3. Then, once the C job is done and the GPU is free: one long constant-rate run on one frontier student
   (suggest `yamnet:a0.25:select:classes=ins_buzz+ambient_rain+human`, 17 min per 7k steps at rung B), branches at 7k,
   14k, 28k, 70k; compare with the C job's 28k cosine point.
4. Report the curve with the noise floor; say plainly if it is flat past some budget or still rising at the last one.

## Constraints from CLAUDE.md worth repeating

- Long jobs through `tools/launch_job.sh` and one `tools/watch_job.sh <pid> <log>` Monitor; do not poll.
- Tools always run from the main checkout; this worktree's `tools/` is only for editing.
- Do not touch `01_annotate/`. Ask before editing existing models or sets; `test_` models are free.
