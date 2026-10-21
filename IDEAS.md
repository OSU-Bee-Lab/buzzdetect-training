# Experiment Ideas

**Untried candidate experiments, nothing else.** When you run an idea, delete its
section entirely; the verdict lives in the run's log entry and `notes.md`. No
strikethrough, "DONE" or summary left behind. If only part is tested, trim to the
untested remainder. Facts, history and provenance go elsewhere:
`docs/training-proposer-facts.md`, `docs/distill-proposer-facts.md`, or the run's
`notes.md`.

Results: `03_train/log.jsonl`, `05_distill/log.jsonl`. Protocol: `LOOP.md`.

---

# Queue — training arm

Ranked best-first by expected value on the headline and the hard folds, cost second.

## 32. Microphone impulse-response / random-EQ augmentation (real audio, re-extracted)

*Evidence: untagged proposal. Device-IR augmentation (Morocutti et al. 2023, arXiv 2305.07499; DCASE 2023-24 Task 1 reports, using the 66 MicIRP vintage-mic IRs) is the strongest single lever for unseen recording devices in acoustic scene classification. E2 `trunk-ft-1e5-aug` (negative, doubled pool) was a different augmentation, and no recorder-response augmentation has run in any era.*

Convolve a random half of each training fold's snips with a random mic IR. If MicIRP's licence doesn't fit, use a no-download alternative: a random parametric EQ, with spectral tilt of ±3 dB/oct and 1-2 peaking filters of ±6 dB between 200 Hz and 4 kHz. Extract the result as an augment dir through `02_set` (a real embedder on real, transformed audio), then train depth8 with `--augment`. Parameters are drawn at random, not fit on pooled statistics, so the CLAUDE.md augmentation rule holds. Check RAM first: `v4-ft-pshud` came close with 4 views, and `TRUNK_STREAM` does not stream augmented embeddings. Prediction: the folds whose recorders differ most from the pool move, so check `DEPLOYMENTS.md` before reading folds. The hard folds' confusers (aircraft, trill) are not a device effect, so they are not the expected movers. Falsifier: headline within 0.02 and no tier up.

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building, and its inference rate against LOOP.md's AVES injunction. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

## 36. Seam-free two-octave view

*Evidence: E4, weak. `ps2-depth8` (−0.017 ± 0.014) used a 4x tile / 0.24 s source window, and its loss was largest at 1_114, the tile-seam fold (`ps-untiled`, `ps-sharedtiled`). Its notes name a seam-free two-octave view as the only form left untested.*

Build the third view from a 4x-longer source window centred on the frame (neighbouring *audio* via `context_frames`, which does not leak), resampled 4:1, rather than tiling 0.24 s. Falsifier: below the three-draw mean again, which closes the two-octave line.

## 37. BN statistics re-estimation: pooled, site-adaptive, and shrunk between them

*Evidence: untagged proposal. AdaBN (Li et al., arXiv 1603.04779) re-estimates BatchNorm statistics on target-domain audio without labels. Counter-evidence: Boudiaf et al. (arXiv 2302.06658) found most source-free adaptation methods underperform the unadapted model on bioacoustic shifts. `ps-bntrain` (BN trained on the pool) was a +0.022 two-draw mean at 30 epochs that did not compound. (a) pooled AdaBN ran as `bn-pooled` (2026-10-20): on ps-depth8 weights trained with frozen AudioSet BN it collapsed the headline 0.481 -> 0.019, so post-hoc statistics misalign weights that co-adapted to the old ones; (b) and (c) estimate post hoc too, so on frozen-BN fine-tunes expect the same, and they would need a model trained with BN tracking (`ps-bntrain`).*

The tail's BN layers are frozen at AudioSet statistics. All variants are offline on saved fold weights (`TRUNK_SAVE_WEIGHTS=1`, exp/ps-d8-soup; ps-d8-soup-m1..m3 have them), and none changes the labels or the weights, only the BN means and variances. They change features, not a monotone map of the score, so the metric can move. Run them on saved weights and compare against the unadapted rescore:

- **(b) Site-adaptive.** Recompute them on the held-out fold's own audio (labels unused, but its frames are scored with statistics it shaped). Failure mode to measure: the statistics absorb whatever dominates the fold, so check 1_95 (the jet flyover) and buzz-rich folds first, and read the per-loudness-tier sensitivity for signal contrast lost.
- **(c) Shrunk.** Blend the pooled statistics (tool: `soup.py --bn-pool`, exp/bn-pooled) as a prior with (b), weighted by the amount of audio the site has, which limits the small-sample and flyover problems.

**(b) and (c) use the eval deployment's audio at test time. In deployment that means a two-pass or streaming per-recording inference, with a per-site score distribution that interacts with the user-set threshold: a policy call for Luke, not one for the loop.** Luke has said to log all three and decide once the models come in. Run (a) freely, and surface (b) and (c) results to him before any design work toward deploying them.

---

# Queue — distillation arm

Ranked best-first by expected movement of the speed / sensitivity frontier,
cost second.
