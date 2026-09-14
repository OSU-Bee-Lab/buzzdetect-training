# HANDOFF — decimate-lead (IDEAS item 19)

The extraction is **not running**. `tools/agent_loop.sh` killed it on purpose
(2026-09-14 14:59) when Luke stopped batch 3: the loop kills a batch's jobs when
the batch ends. Relaunch it first thing.

## Relaunch

From the worktree root:

```bash
tools/launch_job.sh extract.log -- 02_set/main.py --set medium \
  --embedder yamnet_pitchshift_decimate_aves_mid --workers 1
```

That starts a notifier too: it pings you on an error, when the job ends, and
every 50 min. There is nothing to arm; wait for the pings.

Reruns resume incrementally: idents already extracted are skipped, and a
killed ident is rebuilt. The embedder code hasn't changed since the first run
(`embedders/yamnet_pitchshift_decimate_aves_mid/`, smoke-tested and committed to
main, unit-checked byte-identical against `yamnet_pitchshift_aves_mid` on the
unshifted/AVES blocks; see notes.md's Changes section).

**Where it got to.** Three runs so far, all stopped by loop cleanup, none by a
crash: 7/82 idents in the first, 9 cached idents skipped plus 3 in the second,
and 7 more in the third (killed mid-way through
`Chia - Bee Audio 2022 Original/7-13-22_SouthCharleston/3`). Per-ident time
varies a lot with ident size (0.75 to 20 min), so don't extrapolate from one.

## When it finishes

1. The DONE ping says `exit 0`.
2. Launch the CV from the worktree root; its notifier pings once per fold:
   ```bash
   tools/launch_job.sh train.log -- 03_train/main.py --name decimate-lead \
     --set medium --embedder yamnet_pitchshift_decimate_aves_mid \
     --translation general -y
   ```
3. Read results against **both** matched controls, per notes.md's Hypothesis:
   ```bash
   python tools/results.py yamnet_pitchshift_aves_mid decimate-lead   # the tiled-shift lead (main comparator)
   python tools/results.py yamnet_pitchshift_decimate decimate-lead   # optional: vs the plain decimated shift
   ```
4. Fill in notes.md's Results/Conclusion sections. **Read
   `diagnostics/2026-09-14_trill_1_114/README.md` first** — it predicts this
   run's `1_114` behavior (AVES blocks confuse trill for buzz; the decimated
   shift alone is only a marginal improvement over the tiled shift on that
   axis, so don't be surprised if `1_114` stays down here too). Falsifier from
   IDEAS item 19: `1_114` AND `1_95` must both rise above
   `yamnet_pitchshift_aves_mid`, headline no worse than -0.027.
5. `tools/finish_experiment.sh decimate-lead --summary "..." \
     --model .local/worktrees/decimate-lead/models/decimate-lead \
     --baseline-model yamnet_pitchshift_aves_mid --hypothesis "..." \
     --trust clean --conclusion "..." --commit-also IDEAS.md` (only pass
   `--commit-also IDEAS.md` if `git diff IDEAS.md` shows nothing but item 19's
   deletion).
6. Delete this HANDOFF.md in the same commit that records the result (or a
   follow-up commit) — it's a resumption aid, not part of the permanent record.

## If it fails

Read the tail of `extract.log` for the traceback before relaunching.

## Already committed and pushed to main (not just this branch)

- `embedders/yamnet_pitchshift_decimate_aves_mid/` (new embedder, IDEAS item 19)
- `embedders/yamnet_aves_mid_avesshift/` (new embedder, IDEAS item 22 — built
  and smoke-tested, NOT yet extracted or run; next experiment after this one,
  same falsifier logic as item 22's IDEAS entry)
- `diagnostics/2026-09-14_trill_1_114/` (free diagnostic for item 18, already
  complete with README — no further action needed on it)

This branch (`exp/decimate-lead`) holds this experiment's `notes.md` and this
`HANDOFF.md`, plus a merge of main (2026-09-14) for the current tooling.
