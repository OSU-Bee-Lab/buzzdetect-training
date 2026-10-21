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

## 38. Buzz pitch subclasses as auxiliary outputs beside the scored `ins_buzz`

*Evidence: E4, one draw. `binary-d8` (2026-10-21) collapsed the 14 non-buzz classes and lost −0.116 at epoch 30 (−0.026 incl. quiet at the cross-fold epoch, `tools/epoch_rule_audit.py`), mostly on 1_150, 1_95 and 1_114: on a trainable trunk the extra classes' supervision shapes features the buzz boundary uses. Buzz is the one class still collapsed. Its pitch tags are well populated in medium (`ins_buzz_medium` 1777, `_low` 681, `_high` 249 annotations, plus `_quiet`/`_loud` variants).*

Keep `ins_buzz` as the scored output, trained on every buzz frame exactly as now, and add `buzz_low` / `buzz_medium` / `buzz_high` outputs as extra multi-label targets. The metric, threshold and headline are unchanged; only the supervision grows. Pitch only: loudness subclasses would train the model to separate quiet buzz from loud, against detecting quiet buzz. Untagged-pitch buzz (`ins_buzz_pollination`, `_rasp`, `_RECLASSIFY`) gets `ins_buzz` alone.

Needs one code change: a translation maps each label to one class, so let a `to` value name several (e.g. `ins_buzz|buzz_low`, split in `dataset.translate_labels`) and add a translation built that way in `medium/translate.R`. The multi-hot loss already handles several targets per frame. Check `build_weights` doesn't move `ins_buzz`'s loss weight with the new class count (`PIN_BUZZ_TRANSLATION`, as `binary-d8` did). Run on the ps-depth8 cache (no extraction) against a matched `general` control. Prediction: headline ≥ +0.02 with `untagged` or a hard fold up. Falsifier: within ~0.02. A frozen-probe arm is a cheap second read: there the subclasses add capacity (the max over three buzz hyperplanes), not just supervision, so it tests a different mechanism.

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
