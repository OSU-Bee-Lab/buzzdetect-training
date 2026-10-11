# Experiment Ideas

**Untried candidate experiments, nothing else.** When you run an idea, delete its
section entirely; the verdict lives in the run's log entry and `notes.md`. No
strikethrough, "DONE" or summary left behind. If only part is tested, trim to the
untested remainder.

Results: `03_train/log.jsonl`, `05_distill/log.jsonl`. Protocol: `LOOP.md`.
Training proposers read `docs/training-proposer-facts.md` first; the
distillation arm's facts are below.

---

# Queue — training arm

Ranked best-first by expected value on the headline and the hard folds, cost second.

*(Added 2026-10-09 from a literature pass; neither has a log entry. Both are off the pitch-shift line.)*

## 30. More ensemble members, and a cheaper source of them

*Evidence: `ps-d8-ensemble` (2026-10-11), this era: the logit mean of three ps-depth8 draws is 0.558 against members at 0.497-0.515 (+0.049 ± 0.011), a disjoint depth4 + depth10 pair 0.559, five members 0.575. Thresholds drop on every fold; negatives correlate 0.62 between draws, buzz 0.88.*

Two open questions, each one run. (a) Where the curve flattens: two more plain ps-depth8 draws give a 5-draw same-recipe mean to set against the mixed-depth five (0.575); `tools_exp/ensemble_predictions.py` on `exp/ps-d8-ensemble` scores any member list in seconds. (b) Whether one training run can supply decorrelated members: K independent tails (layers 8-14 + head, different init order / batch order) on the shared frozen stem, trained in one job and averaged at the logit. Same training compute as K runs but one extraction pass and one deployable graph; time it against a single ps-depth8 pass, since layers 1-7 are shared. Falsifier for (b): the K-tail mean lands at a single draw's headline (the tails co-adapt or the decorrelation needs independent data order). EMA within a run did not do it (`ps-ema` -0.008).

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building. Costs an extraction, so it comes after 28. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

---

# Queue — distillation arm

Ranked best-first by expected movement of the speed / sensitivity frontier,
cost second. `python 05_distill/ladder_record.py frontier` is the current state;
comparable (clean) students only, contaminated-era readings are leads.

## D1. An ensemble teacher

*Evidence: `ps-d8-ensemble` (training arm, 2026-10-11): averaging 3 draws of the fine-tuned recipe is +0.049 ± 0.011 on `medium`'s CV, 5 members +0.066. Untested on `moderate` and untested through a student.*

A student's speed does not depend on its teacher's cost, so the ensemble gain is free at the lite tier if it survives distillation. Train N (3) draws of the teacher's recipe on `moderate`, deploy one graph that averages their logits (and pick one member's code, or the mean code, for the code-regression loss: say which), and distil the frontier's anchor students from it against their existing rows. It is a new teacher: its own `<distill_cache>/<teacher>/` targets (the `_mel` cache is shared), its own log `key`, so the matched controls must be re-judged under it or compared by headline only. Costly (N teacher trainings plus a target re-cache); a training-arm confirmation on `moderate` comes first and needs Luke's go-ahead for the big-set runs. Falsifier: the student of the ensemble lands within seed noise (0.02) of the same student of one member.

## D2. A 64-band, hop-16 front end between `yamnet` and `fast32h16`

*Evidence: `matmul-stft` (2026-10-11): with the Pad gone, the frontier has a hole between yamnet a0.25_d8 (0.713 at 3.24x) and twofast32 a0.50 (0.691 at 4.81x), and frame rate is the largest speed term (front end alone: yamnet 3.6x, fast32 6.7x, fast32h16 9.2x). Hop 10 -> 16 ms cost ~0.035 on fast32 (0.687 -> 0.650 at a0.50); 64 -> 32 bands and the 25 -> 16 ms window are not separated.*

Register `yam64h16` (25 ms window, 64 bands, 125-7500 Hz, hop 256; 60 frames) and distil `a0.25_d8` and `a0.50_d8` on it at rung C (WSD to 56000, buzz+rain+human), exported with `mmt`. Bench it with random weights first (`bench_arch.py --seconds 200`, `+mmt`): it should land near 4-5x. Costs a `cache_fe` pass and ~25-45 GB of rung-C shards on the root disk (149 GB free on 2026-10-11). Gain: a student above the line from 0.713 at 3.24x to 0.691 at 4.81x by more than 0.02. Falsifier: it lands on that line (frame rate, not bands or window, is what the fast front ends lose to).

## Standing facts a distillation proposer needs

Detail and evidence for each: `05_distill/FRONTENDS.md` (closed 2026-10-06).
Speeds are x YAMNet at 200 s on the GTX 1650; headlines are as in the training arm.

- **Export with `--fe-impl mmt`; every speed measured before 2026-10-11 included a
  Pad node that cost more than the student** (`matmul-stft`). A `Pad` (or
  `Concat`) over a 200 s waveform takes 11-20 ms on the CUDA EP; the rest of a
  fast student's call is ~4 ms. `mmt` (Slice + Reshape + MatMul, tail padded
  apart) is the same numbers and no retraining: every rung-C student has an
  `_mmt` row with an identical headline, 1.5-2.7x faster. The code is on
  `exp/matmul-stft` (not merged): bring it over as a patch.
- **Without the Pad, front end and trunk both set the speed.** Front end alone,
  x YAMNet at 200 s: fast32h16 9.2, fast32h32 8.4, twofast32 6.9, fast32 6.7,
  yamnet 3.6. Frame rate (hop 16 ms against 10 ms) is the large term; trunks on
  fast32h16 now span 5.6x (a0.75_d8) to 7.3x (a0.25). The older readings (FFT
  length sets the speed, shrinking the trunk buys little, a ~2.7x ceiling) were
  the Pad. Band count and band placement (`lo` twins) were not re-timed.
- **Time whole graphs; don't read per-node profiles on the GPU.** An onnxruntime
  per-node profile on the CUDA EP reports asynchronous kernel launches, not
  costs (it put ~60% in depthwise convs; fused full-conv trunks then timed
  slower at equal width). Run `bench_arch.py` first, with `--seconds 200` (its
  default is 20; every table here quotes 200 s). To find where a graph's time
  goes, `prefix_bench.py` (on `exp/matmul-stft`) times prefixes of it; that is
  what found the Pad.
- **No narrow buzz band.** Buzz frames differ from clean negatives nearly flat
  across frequency (`band_profile.py`); no reason to expect a band-limited front
  end to win big.
- **Distil only `ins_buzz`, `ambient_rain`, `human`:** +0.05 at every width on
  fast32h16 (rung B), same speed. `lam=0` (no code regression) hurts. Every live
  frontier student is such a subset student.
- **Seed noise on the headline is ~0.01-0.02**; the repeat rule's bar is 0.02.
  Read single-seed gaps under ~0.03 as ties.
- **Steps vs data.** At equal steps rung C ≈ rung B; past ~14k steps the gain
  needs C's data (B overfits: 0.616 at 14k, 0.588 at 56k; C keeps climbing to
  0.650). YAMNet-trunk students are flat from 7k; fast-front-end students keep
  learning to 28-56k. A 7k WSD branch matches a 7k cosine run.
- **hit@K tracks the headline** (r 0.94 over buzz+rain+human students) and is
  calibration-blind, which is why it is the WSD stop signal. hit% at logit 0 is
  as good a proxy but also reads the student's calibration offset.
- **The clean frontier (2026-10-11, `_mmt` rows), rung C:** yamnet a0.50_d8
  0.735 at 2.60x, yamnet a0.25_d8 0.713 at 3.24x, fast32 a0.50_d10 0.696 at
  4.55x, twofast32 a0.50 0.691 at 4.81x, fast32h16 a0.75_d8 0.656 at 5.57x,
  fast32h16 a0.50_d8 0.646 at 6.67x, fast32h16 a0.25 0.612 at 7.28x. All far above
  the floor (0.207) and the CV baseline (0.414). Comparison points: teacher
  honest rotation 0.574; teacher ONNX through the harness 0.692 (inflated, trained
  on those folds). Students inherit fold knowledge through the teacher, so beating
  0.574 is not "better than the teacher".
- **Contamination:** every rung-B student and the yamnet rung-C runs before
  2026-10-04 trained on test-set audio; their rows are in
  `05_distill/log.contaminated_2026-10-02.jsonl`. Directions survive (one retrain
  landed inside seed noise), numbers are not comparable.
- **Jets at `1_95`:** 36-76% of the threshold-setting false positives in every
  student; none rejects them. Class subsets help on that fold (0.134 to 0.200,
  fast32h16 a0.50). `eval_folds.py probe` scores it (a training fold, not
  held-out). The teacher has not been probed.
- **Eval is slightly pessimistic for long windows** (each frame scored alone,
  zero-padded), and the headline is the student ONNX's, not the full pipeline's.
- **Untried:** pruning dead channels, non-uniform widths, a learned (conv) front
  end, int8 (needs modern hardware to time).

---

# Low priority — deploy speed, not accuracy

**Luke, 2026-09-25: low priority, "mostly fun".** Neither item aims at the
headline. buzzdetect analyses are IO-bound on slow drives, but on NVMe the
network's compute should be a material share of the time. Pick these up only
when the queue above is empty or blocked, never ahead of it. **Profile first:**
time a deploy run on NVMe split into decode/front end vs ONNX inference. If
inference is a small share, both items stop there. In stage 2 the GPU sat ~10%
busy, held back by the CPU (CLAUDE.md).

## 31. Take the waveform Pad out of the deployed graphs

*Evidence: `matmul-stft` (2026-10-11), measured by prefix timing at 200 s on the GTX 1650: the `Pad` over the waveform is 11.0 of 50.6 ms in `yamnet_large_general` and 2 x 10.8 of 140 ms in `v4-ft-ps-e60-moderate`; YAMNet's whole front end is ~24 ms, of which the pad-free MatMul form needs ~4 on the same spectrogram. The teacher's octave-up resampling Conv is another 54 ms.*

`04_deploy`'s exported front end (YAMNet's `WaveformFeatures`) pads the whole waveform before the STFT. Replace it with `frontends.make_features_layer(get('yamnet'), 'mmt')` from `exp/matmul-stft` (equal to the STFT form within 2e-5 on the log-mel, patch counts identical at every length tried), or do it as an ONNX pass in `04_deploy/onnx_passes.py`. Parity as in the int8 work: detections flipped in both directions at logit > 0, `ins_buzz` first, then raw logit difference. Expected: YAMNet-headed models ~1.3x from the Pad alone, more with the MatMul STFT; the two-pass teacher ~1.2x. Not low-value "fun": it is a measured, exact-parity speedup of the standard tier, and it also moves the denominator of every "x YAMNet" in the distillation log, so re-time the reference when it lands. Needs Luke's nod only because it changes shipped graphs.

## 26. Post-training int8 quantization of the shipped ONNX graph

*Evidence: untagged proposal. Background: per-channel PTQ of MobileNetV1
usually costs little; per-tensor PTQ of depthwise convs is the known failure.*

Quantize the exported model (`04_deploy/export_onnx.py` output) with
onnxruntime's static quantization. Use per-channel weights, calibrated on
training-pool frames. Nothing gets installed into `buzzdetect-train`, and
nothing retrains. Measure two things. **Speed:** CPU and GPU, fp32 vs int8, on
NVMe. **Fidelity:** score every rotating fold with both graphs and compare the
headline, per-fold sensitivity, **and each fold's threshold**. `1_95`'s
threshold is set by one jet flyover, so rounding noise can move a threshold
while mean agreement looks fine.

*Escalation, only if PTQ loses fidelity:* quantization-aware training (fake-
quant nodes in fp32 training). `tensorflow_model_optimization` does not
support Keras 3. Any QAT path needs a separate env (TF pin hazard) or a
PyTorch/ONNX route, so scope it before building.

*Falsifier:* if PTQ gives under ~1.5x end-to-end on NVMe, close it. If it moves
any fold's sensitivity beyond that fold's eval-sampling SD, don't ship it.

**First pass done, 2026-09-28** (`diagnostics/2026-09-28_int8-ptq/`): plain static int8 flips too many
detections (+36/-75 at best, weights-only +4/-32, fp16 0/0). Speed untimed: this box has no
AVX2/VNNI. Open: skip-layer sweep, AdaRound / bias correction, per-fold sensitivity, then time on
modern hardware (Luke offered).

---

# Needs Luke

## 25. `1_37`'s threshold is set by `ambient_background` — a 22-frame listen list

*Evidence: **E4** census above, one draw.*

20 of the 22 negatives setting `1_37`'s threshold (chicory, 0.410 on the lead)
are annotated `ambient_background`, and the fold's threshold (-1.00) is second
worst. Two readings with opposite fixes: they are unannotated faint buzz (an
annotation question, which only Luke can answer), or chicory's background is
genuinely buzz-like (a representation question, item 15's territory). Build
the list — ident, snip, offset, raw activation — with the census recipe in
`docs/training-proposer-facts.md`, commit it under `diagnostics/`, and ask Luke to listen. No
training. Do not relabel anything.

---

# Parked

- **[E3] Seed averaging inside a run.** Shrinks error on every future
  experiment for ~2-3x compute. Era-boundary decision; **Luke declined for
  now** (2026-09-10). Re-raise at the next cutover.
- **[policy] `large`-set confirmation. Forbidden without Luke asking.**
- **[E3] Frame length isolated from embedder** (YAMNet at a 5 s effective frame,
  `overlap_event_s` held absolute). Item 2 asks the motivating question more
  cheaply.
- **[E2] Context width k=2.** Its E2 negative's mechanism failed to reproduce,
  but context has now landed rich-folds-only three times (`context-frames-fix`,
  `yamnet-aves-context`, `pitchshift-context`), so widening it is a lever on
  the wrong folds. Code on `refs/archive/context-width`.
- **[E2 + literature] Sub-frame pooling on `yamnet_trunk`** (log-sum-exp over
  time, or keep the frequency axis). E2 option 1 was −0.007, and attentive or
  finer pooling has been measured **not** to help CNN encoders
  (arXiv:2605.10494). Item 21 tests the pooling hypothesis where it is expected
  to work. The `yamnet_trunk` cache no longer exists.
- **[throughput only] Split framing from embedding in `--workers`.**

# Ruled out — do not re-propose

Full reasoning is in `03_train/log.jsonl` and each branch's `notes.md`.

- **Site-adversarial / site-confusion terms on the fine-tuned tail's pooled
  code** (`site-adv-w01/w03/w10`, `site-conf-w01`, `site-neg-w01`,
  `site-ent-ramp`): six runs, four shapes of the term (gradient reversal, KL to
  uniform on all frames, the same on non-buzz frames, bounded entropy ramped
  from 0). Each lost the class output before the site head left ~15% (chance
  2.1%; unopposed it reads the training fold at 84%). The ramped run kept the
  class head only while the weight was <= 0.02, where site accuracy was still
  ~0.5. Not a weight-tuning question; a different mechanism would be a new idea.
- **Per-site score calibration / label-free threshold selection** — inert by
  construction.
- **`1_150` as an annotation-quality problem** — Luke listened 2026-09-09:
  "Most of them are very quiet, but still legitimate targets."
- **Leave-one-concept-out as the cause of the hard folds** — r = 0.013, ruled
  out twice. Check roles before reviving it.
- **`--monitor val_sens` in any form**, and **buzz-only epoch selectors** — the
  selector does not move the score; the budget does.
- **Margin or ranking losses against a named confuser** (`mech-margin`,
  `pairwise-rank`) — monotone negatives on the fold they targeted. Applies to
  trill as much as to the jet.
- **Dropout on YAMNet-derived blocks** — null on plain YAMNet and on
  `yamnet_pitchshift`.
- **Per-block input standardisation** — negative on `yamnet_aves` at a fixed
  budget.

**Caveat that belongs to every hard-fold claim here.** Seven `ins_buzz`
annotations spanning one 300 s file supply 63% of mustard's and 52% of
Fit+Fast's buzz seconds. Their `buzz_frames` counts overstate their independent
sample size.
