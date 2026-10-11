# HANDOFF: ps-d8-aves (batch 30, experiment 2 of 4)

Batch 30 status at park (2026-10-11 05:05): **1 of 4 logged** (`fused-blocks`, distill arm). Remaining:
`ps-d8-aves` (this, CV running), `conv-stft` (code ready, waits for the GPU; its own HANDOFF.md),
and a fourth not yet chosen (see conv-stft's handoff for the suggestion).

AVES is under LOOP.md's injunction; Luke allowed this one test on 2026-10-11 ("I'll allow your cheap test
this once, then finish up your batch like usual"). Do not start a second AVES run (no repeat, no AVES2).

## The job
- pid **3556926**, log `.local/worktrees/ps-d8-aves/train.log`. Adopt it: `tools/watch_job.sh --adopt 3556926`,
  then one `tools/watch_job.sh` Monitor.
- Stage 3 CV, 8 folds, 16.6 min per fold, started 04:33; ETA ~06:47. Fold 1 (`1_29`) 0.600, buzz logit SD 1.33.
- **If it is still running, report progress and stop** (re-arm; park again only if > 1.5 h remain).

## When it finishes (exit 0)
1. Check every fold line's `buzz logit SD` in train.log (0.57-1.3 is normal for this recipe).
2. `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 ps-d8-aves` from this worktree (control: the
   same-recipe draw of 2026-10-10, 0.497; other draws 0.509, 0.515). If ± SD is blank the frame grids
   differ: say so, the embedder was meant to share ps-depth8's grid (0.96 s frames).
3. Fill notes.md (Hypothesis is written; add Changes, Results, Conclusion). Changes: embedder
   `yamnet_trunk_pitchshift_depth8_aves` (in main, commit 1a78e6a), extraction 31 min on GPU
   (`BUZZDETECT_NO_GPU=1 ... 02_set/main.py --set medium --embedder yamnet_trunk_pitchshift_depth8_aves
   --workers 0`), then the ps-depth8 recipe unchanged.
4. If the result is null or negative: before logging it as an embedder verdict, run one offline readout
   check (LOOP.md "A new embedder doesn't inherit the probe config"), and say what was measured. The head
   here is a Dense at lr 2e-4 for 30 epochs over [2048 tail code, 2304 AVES columns].
5. From main: `tools/finish_experiment.sh ps-d8-aves --summary "..." --model
   .local/worktrees/ps-d8-aves/models/ps-d8-aves --baseline-model ps-depth8-r3 --hypothesis "..." --trust
   <clean|caveated> --conclusion "..."`. Delete this HANDOFF.md first (finish refuses a worktree with one).
   IDEAS.md item 29 (AVES2/BEATs) stays: it was not run. Mention in the conclusion that it is the same
   question with a ~1.5x faster encoder, still under the injunction.

## If it died
Relaunch from this worktree (finished folds are skipped):
`TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 tools/launch_job.sh train.log -- 03_train/main.py --name ps-d8-aves --set medium --embedder yamnet_trunk_pitchshift_depth8_aves --translation general --epochs 30 -y`
An OOM in fold 1's first step: add `TRUNK_BATCH=512 TRUNK_ACCUM=2`.
