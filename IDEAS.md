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

## 28. Pseudo-labelled unlabeled audio from training deployments

*Evidence: untagged proposal. MAST (arXiv 2609.15221) gets +0.12 to +0.22 mAP under cross-site shift from two-stage self-training.*

Annotation covers 24 snips per fold; the rest of each recording is unlabeled. Score the unlabeled audio of the **training** folds with the current model, take high-confidence positives and negatives as pseudo-labels, and retrain with them added. Confident negatives are the cheap win: they widen the negative tail where the 0.5% FPR threshold is read. First check how much unlabeled audio exists per training fold and that nothing from the scored fold is touched (`03_train/CLAUDE.md`: no augmentation or mix may cross a fold boundary). Using the *held-out* fold's unlabeled audio is a protocol change and needs Luke.

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building. Costs an extraction, so it comes after 28. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

---

# Queue — distillation arm

Ranked best-first by expected movement of the speed / sensitivity frontier,
cost second. `python 05_distill/ladder_record.py frontier` is the current state;
comparable (clean) students only, contaminated-era readings are leads.

## Standing facts a distillation proposer needs

Detail and evidence for each: `05_distill/FRONTENDS.md` (closed 2026-10-06).
Speeds are x YAMNet at 200 s on the GTX 1650; headlines are as in the training arm.

- **Speed is set by the front end, and within it by FFT length.** YAMNet's own
  front end alone runs ~2.1x YAMNet, which caps every YAMNet-front-end student
  (a0.25 ≈ 1.6x). fft 256 ~2.7-3.1x, fft 512 ~1.65-2.0x, fft >= 1024 is slower
  than YAMNet; a lower frame rate (hop 16/32 ms) helps. Shrinking the trunk buys
  little: 14x fewer MACs (a1.00 to a0.25) is 2.0x. Band count barely matters, and
  band placement (`lo` twins) ties its twin.
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
- **The clean frontier (2026-10-05), rung C at the rule's stop:** yamnet a0.25
  0.707 at 1.62x, twofast32 a0.50 0.690 at 1.76x, fast32 a0.50 0.680 at 2.17x,
  fast32h16 a0.50 0.639 at 2.51x, fast32h16 a0.25 0.612 at 2.75x. All far above
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
