# trunk-ft-large

## Hypothesis

Not a lever test. This asks what it takes to run the `trunk3-ft-1e5` recipe (YAMNet layers 13-14 fine-tuned,
backbone LR 1e-5, head LR 2e-4, batch 1024, 60 epochs) on the `large` set (same annotations as `medium`, framehop
0.2 instead of 1, so ~5x the frames), what breaks, how long a CV takes, and the marginal gain over medium.
Control: `trunk3-ft-1e5` (medium, 0.398). No matched control exists: frame grid, frames per epoch (so gradient
steps per epoch) and the labels' overlap handling all differ, so a delta is a rough marginal gain, not a lever effect.
`large` is otherwise a final-confirmation step; Luke asked for this run explicitly on 2026-09-20.

## Problems met and fixes (in order)

1. `large` was stale: annotations dated Sep 1, medium's Sep 11. Re-ran `02_set/sets/large/build.R` (copies medium's
   files); the fingerprint change invalidated snips and embeddings.
2. Worktree `audio` symlink is tracked in git with a Mac path (`/Volumes/Expansion/...`), so source audio was
   unreadable and 80 idents were skipped. Relinked to `/media/server storage/experiments` (not committed).
3. One large snip (`wooster/1_143/240726_1446/snip_105210.000_105570.004.flac`) had a corrupt FLAC header from the
   Sep 1 cut (frames = 2^63-1), giving `psf_fseek() failed` and "did not extract". Replaced with medium's copy of
   the same range; scanned all 2846 large snips, no others bad.
4. `train.py`'s per-fold cleanup (`del model; clear_session(); gc.collect()`) dropped only `model`, not `data`
   (`TrainingData.train_tf`), and `_load_data` loaded every training fold's raw float32 embeddings into one list
   before converting any of it to the packed float16 array — so on `medium`'s small pool this never mattered, but
   on `large`'s ~40-fold pool (raw ~18 GB + packed ~9 GB) the process was OOM-killed by the kernel every single
   fold (confirmed via `journalctl -k`, anon-rss 23.1 GB against a 23 GB box). Fixed in two parts: `data` is now
   explicitly cleared between folds, and `_load_data` packs and frees each source fold's embeddings as it loads
   rather than after the whole pool is resident (`_pack_fp16`/`_assemble_fp16`). Cost the CV 6 retries and a
   crash-recovery lap before the fix landed; the 8th fold and the shipped-model run both completed cleanly with
   the fix in place. Also: `chain_large.sh` had no `|| exit 1` after its retry wrapper, so it printed
   `CHAINLARGE-DONE` even with fold 8/8 untrained when retries exhausted the first time -- fixed to fail loudly.
5. First deploy attempt failed on "no destination": `setup_worktree.sh` never symlinked the gitignored
   `paths.local.json` into new worktrees. Fixed there too; linked by hand here to unblock this run.

## Results

`sensitivity_exclquiet` at fpr 0.005, fold identities matched exactly across `medium` and `large` (same
deployments); ± SD not computed (one-off check, not a lever test -- see Hypothesis).

| fold | baseline (medium, cv-baseline-v3-refresh) | frozen trunk (medium, trunk3-frozen) | fine-tune (medium, trunk3-ft-1e5) | **fine-tune (large, trunklarge-ft-1e5)** | delta: large vs medium fine-tune |
|---|---|---|---|---|---|
| 1_29 | 0.443 | 0.452 | 0.485 | 0.438 | -0.047 |
| 53 | 0.440 | 0.424 | 0.554 | 0.547 | -0.007 |
| willard (1_11) | 0.368 | 0.347 | 0.570 | 0.532 | -0.038 |
| wooster (1_143) | 0.459 | 0.454 | 0.524 | 0.585 | +0.061 |
| 1_150 | 0.231 | 0.231 | 0.296 | 0.273 | -0.023 |
| 1_95 | 0.042 | 0.052 | 0.087 | 0.106 | +0.019 |
| 1_37 | 0.399 | 0.382 | 0.446 | 0.459 | +0.013 |
| 1_114 | 0.252 | 0.229 | 0.221 | 0.186 | -0.035 |
| **total (headline)** | **0.329** | **0.321** | **0.398** | **0.391** | **-0.007** |

Inclusive reading (`sensitivity`, quiet buzz included): baseline 0.268, frozen 0.262, medium fine-tune 0.326,
large fine-tune 0.317 (delta -0.009). Same read as exclquiet.

Shipped model (`--skip-cv`, 49 epochs from the consensus val_loss curve, all 43 folds pooled, 404755 frames)
deployed to buzzdetect (`trunklarge-ft-1e5`, ONNX parity max|d|=4.07e-02, top-class agreement 1.0000).

## Conclusion

**One-off assumption check, not a loop experiment** -- run once at Luke's explicit request to see whether the
`trunk3-ft-1e5` recipe (and the whole pipeline) holds up on `large`, not to search for a lever. Headline moves
-0.007 (exclquiet) / -0.009 (inclusive), both well inside this era's ~0.012-0.014 headline-delta noise floor
(`docs/judging-results.md`) -- a null result at the mean, not a regression. Per-fold, the two hard folds split:
`1_150` down (-0.023), `1_95` up (+0.019, still near-chance at 0.106); neither move is outside single-fold
eval-sampling SD (0.03-0.13), so nothing here says `large` fixes or breaks the hard folds. `1_114` moved most
(-0.035) and is also a hard fold, worth a second look if `large` gets revisited. `wooster`'s +0.061 is the
biggest single mover but is a rich, background-tier-heavy fold, consistent with this era's general
rich-folds-move-more pattern rather than anything specific to `large`.

**The assumption holds**: fine-tuning transfers to `large` about as well as it worked on `medium`, no material
degradation and no hidden failure mode the bigger set uncovered in the *result*. It did surface three real
pipeline bugs (per-fold host-RAM leak, a chain script's false-completion signal, a missing worktree symlink) --
all fixed in this branch and generalized where the fix belonged (`setup_worktree.sh`, `chain_large.sh`), not just
patched for this run. `large` stays a final-confirmation step, not a loop target; this doesn't change that.
