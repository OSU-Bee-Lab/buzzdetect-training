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

## 31. Cross-fold MixStyle on the cached layer-7 map (deployment-style augmentation)

*Evidence: MixStyle (Zhou et al., ICLR 2021, arXiv 2104.02008) and Freq-MixStyle are standard for device shift in DCASE Task 1 (Morocutti et al., arXiv 2305.07499; DG-SED arXiv 2407.03654). `mixstyle-dep` (2026-10-20) ran the within-fold variant (partners from the frame's own deployment, p=0.5, Beta(0.1,0.1), `TRUNK_MIXSTYLE`): -0.031 +- 0.014 vs ps-depth8-r3, 1_95 -0.045 +- 0.021, 1_150 -0.074 +- 0.044, `background` +0.038 the only gain.*

The within-fold run mixed only within-deployment style, which is not what differs between deployments. The method's point is mixing across deployments: partners from the rotation's other training folds (never the val/test fold; see `03_train/CLAUDE.md`). Extend `TRUNK_MIXSTYLE`'s partner draw to the whole training pool. Since the within-fold version already cost ~0.03, try a gentler setting too (p=0.25, or mixing only the std). Falsifier: within ~0.02 of the three-draw depth8 mean (0.507), flat or falling 1_95 / 1_114, or the same `loud`/`untagged` loss within-fold showed.

## 32. Microphone impulse-response / random-EQ augmentation (real audio, re-extracted)

*Evidence: untagged proposal. Device-IR augmentation (Morocutti et al. 2023, arXiv 2305.07499; DCASE 2023-24 Task 1 reports, using the 66 MicIRP vintage-mic IRs) is the strongest single lever for unseen recording devices in acoustic scene classification. E2 `trunk-ft-1e5-aug` (negative, doubled pool) was a different augmentation, and no recorder-response augmentation has run in any era.*

Convolve a random half of each training fold's snips with a random mic IR. If MicIRP's licence doesn't fit, use a no-download alternative: a random parametric EQ, with spectral tilt of ±3 dB/oct and 1-2 peaking filters of ±6 dB between 200 Hz and 4 kHz. Extract the result as an augment dir through `02_set` (a real embedder on real, transformed audio), then train depth8 with `--augment`. Parameters are drawn at random, not fit on pooled statistics, so the CLAUDE.md augmentation rule holds. Check RAM first: `v4-ft-pshud` came close with 4 views, and `TRUNK_STREAM` does not stream augmented embeddings. Prediction: the folds whose recorders differ most from the pool move, so check `DEPLOYMENTS.md` before reading folds. The hard folds' confusers (aircraft, trill) are not a device effect, so they are not the expected movers. Falsifier: headline within 0.02 and no tier up.

## 33. One-way partial-AUC auxiliary loss at low FPR

*Evidence: untagged proposal. Zhu et al., "When AUC meets DRO" (ICML 2022, arXiv 2203.00176; LibAUC's SOPA-s) optimise one-way pAUC over the top-β negatives directly. The headline is sensitivity at FPR 0.005, a single point on that curve. Closed precedents: margin/ranking losses against a **named** confuser (`mech-margin`, `pairwise-rank`, E3) and OHEM (`tail-loss`, E2). This differs from both: no class is named, and it ranks positives against whichever negatives currently score highest.*

Add λ·L_pAUC to the BCE: per batch, the squared hinge (margin 1) between each `ins_buzz` positive's logit and the top ceil(0.05·n_neg) negative logits. Positives are `_quiet` included, matching how training already treats them. λ=0.3 on depth8 ran as `pauc-l03` (batch 38): -0.091 ± 0.015 vs ps-depth8-r3, largest losses on 1_114/1_150/1_95, no logit collapse. Remaining rung: λ=0.1 on depth8. Watch the `buzz logit SD` line: E2's `tail-loss` collapsed the class output, and a dose that does the same is an artifact, not a verdict. Falsifier: monotone decline with dose, as `pairwise-rank` showed. If that happens, close the low-FPR-loss family.

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building, and its inference rate against LOOP.md's AVES injunction. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

## 34. Weight-space averaging: model soup of depth8 draws, then WiSE-FT

*Evidence: untagged proposal. Model soups (Wortsman et al., ICML 2022, arXiv 2203.05482) average fine-tuned weights that share a pretrained init and often match the ensemble at one model's cost. WiSE-FT (Wortsman et al., CVPR 2022, arXiv 2109.01903) interpolates fine-tuned weights back toward the pretrained ones and gains OOD robustness. In this log, the 3-draw logit ensemble is the largest clean step (+0.049, `ps-d8-ensemble`), and its source is decorrelated negatives. `ps-ema` (within-run averaging) was null, which is different from averaging separate runs.*

Fold weights are not saved today (`folds/<fold>/` holds predictions only), so the first step is a knob that saves each fold's tail and head. Train three depth8 draws with it (ens3's cost), then score offline on each fold's cache, with no new training: (a) the uniform soup of the three, against the logit ensemble of the same three and against the best member; (b) WiSE-FT on the soup, with the tail interpolated toward the pretrained YAMNet layers 8-14 at α 0.25 / 0.5 / 0.75 and the head kept. A soup below its best member means the constant-LR, random-head runs left the shared basin (a shared head warm-start, `lp-ft`, was itself negative, so it is not the fix). Gains here matter for the standard tier's graph: one tail instead of three.

## 35. Layer-wise LR decay across the trainable tail

*Evidence: untagged proposal. LLRD is the default for fine-tuning AST/BEATs-style audio models and BEiT. The depth ladder here saturates at 8, `ps-depth4` ties it at ~2x compute, and depth8 overfits past 30 epochs, so the lower blocks may be moving too far at a uniform 1e-5.*

Scale each block's backbone LR by γ^(14 − layer), with the top block at 1e-5, using `_make_adam_multilr` per block instead of one backbone multiplier. Run one rung at γ=0.7 on depth4, the deepest tail, where the effect should be largest; depth8 serves as the comparator. A hyperparameter on paper, but it decides whether trainable depth past 8 is usable, which is structural. Falsifier: depth4+LLRD within 0.02 of the depth8 three-draw mean.

## 36. Seam-free two-octave view

*Evidence: E4, weak. `ps2-depth8` (−0.017 ± 0.014) used a 4x tile / 0.24 s source window, and its loss was largest at 1_114, the tile-seam fold (`ps-untiled`, `ps-sharedtiled`). Its notes name a seam-free two-octave view as the only form left untested.*

Build the third view from a 4x-longer source window centred on the frame (neighbouring *audio* via `context_frames`, which does not leak), resampled 4:1, rather than tiling 0.24 s. Falsifier: below the three-draw mean again, which closes the two-octave line.

## 37. BN statistics re-estimation: pooled, site-adaptive, and shrunk between them

*Evidence: untagged proposal. AdaBN (Li et al., arXiv 1603.04779) re-estimates BatchNorm statistics on target-domain audio without labels. Counter-evidence: Boudiaf et al. (arXiv 2302.06658) found most source-free adaptation methods underperform the unadapted model on bioacoustic shifts. `ps-bntrain` (BN trained on the pool) was a +0.022 two-draw mean at 30 epochs that did not compound.*

The tail's BN layers are frozen at AudioSet statistics. All variants are offline once #34's saved fold weights exist, and none changes the labels or the weights, only the BN means and variances. They change features, not a monotone map of the score, so the metric can move. Run them on the same saved weights and compare against the unadapted rescore:

- **(a) Pooled.** Recompute the statistics on the rotation's k-1 training folds only, never the held-out fold. This ships as static weights, with no inference change and no policy call, and the loop can run it. Expect it to overlap `ps-bntrain`; say so when reading the result.
- **(b) Site-adaptive.** Recompute them on the held-out fold's own audio (labels unused, but its frames are scored with statistics it shaped). Failure mode to measure: the statistics absorb whatever dominates the fold, so check 1_95 (the jet flyover) and buzz-rich folds first, and read the per-loudness-tier sensitivity for signal contrast lost.
- **(c) Shrunk.** Blend (a) as a prior with (b), weighted by the amount of audio the site has, which limits the small-sample and flyover problems.

**(b) and (c) use the eval deployment's audio at test time. In deployment that means a two-pass or streaming per-recording inference, with a per-site score distribution that interacts with the user-set threshold: a policy call for Luke, not one for the loop.** Luke has said to log all three and decide once the models come in. Run (a) freely, and surface (b) and (c) results to him before any design work toward deploying them.

---

# Queue — distillation arm

Ranked best-first by expected movement of the speed / sensitivity frontier,
cost second.
