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

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building, and its inference rate against LOOP.md's AVES injunction. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

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

**Queued by Luke on 2026-10-22 while the distillation injunction (CLAUDE.md) stands. The injunction still holds:
don't run anything here until it lifts.**

## 40. Buzz-only students (`classes=ins_buzz`) against their three-class twins

*Evidence: E3, contaminated. In the 2026-10-02 quarantine (`05_distill/log.contaminated_2026-10-02.jsonl`:
rung B, 7k-step cosine, seed 1, trained on test-set audio), 15 matched pairs scored buzz-only minus
`ins_buzz,ambient_rain,human` at a mean of +0.010 ± 0.016 SD. The deltas ran from +0.041 (fast32lo a0.25) and
+0.037 (fast32lo a0.375) down to −0.014 (twofast32 a0.50). The gains sat mostly on the narrower, cheaper
students; the yamnet-trunk pairs were within ±0.007. Contamination keeps the direction but not the numbers
(`docs/distill-proposer-facts.md`). Every clean run since 2026-10-04 used the three-class subset, so buzz-only
has never had a clean read.*

Retrain the clean three-class frontier students as buzz-only (`<frontend>:<arch>[:<init>]:classes=ins_buzz`).
Match the teacher, rung, seed and WSD budgets of an existing `_c-buzz-rain-human` row, so the twin is already
logged. Take two or three frontier points, cheapest first (fast32h16 / b64w16h16 a0.25), where the old gap was
largest. Watch 1_95: class subsets helped there before (0.134 → 0.200), but dropping rain and human also drops
the last confounder supervision, and jets set most of the threshold false positives. Read `eval_folds.py probe`
next to the headline. Check that buzzdetect copes with a one-column model before any deployment. Prediction:
headline ≥ +0.02 on the cheap students at equal speed. Falsifier: every pair within ~0.02 (one seed each), or
1_95 falls.
