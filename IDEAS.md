# Experiment Ideas

**A queue of untried ideas. Candidate experiments, nothing else — closed items
do not belong here in any form.** Results go in `log.jsonl`, protocol in
`LOOP.md`, closed eras in `archive/`. When you run an idea, **delete its
section entirely** — the verdict, the mechanism and the "don't rerun this" live
in that run's `log.jsonl` entry and its `notes.md` on `exp/<slug>`. **A
strikethrough heading, "DONE"/"ANSWERED" writeup, or `## Closed:` summary left
in its place is the same violation as not deleting it** — it still costs every
future agent an uncached re-read for a verdict that's already durably recorded
elsewhere. This file grew to 1047 lines, 46% of it closed material, before that
rule was enforced, and it has been re-violated since (writeup-in-place, not
outright non-deletion) — see `CLAUDE.md`'s Invariants for the standing note. If
only part of an idea is tested, trim to the untested remainder; don't narrate
the tested part here.

**State as of 2026-09-14.** Anchor **`cv_baseline_v3` = 0.330** (bare linear
probe on frozen YAMNet, `--epochs 400`). Provisional era lead
**`pitchshift-aves-mid` = 0.422** — `[yamnet(t), yamnet(octave-up(t)), AVES
layers 6/9/12 mean-pooled]`, 4352-d, linear. Its repeat draw
(`pitchshift-aves-mid-repeat`) was in flight when this was written; read its log
entry before building on 0.422. Confirmed levers this era: pitch-shift up an
octave (+0.069), AVES middle layers (+0.026 to +0.032), `--hidden 1024` on
`yamnet_aves` (+0.027), removing the pitch-shift tile seam (headline-flat, but
`1_95`/`1_114` up both draws). Context stacking is honest at ~+0.02 to +0.03
and **three times** landed on rich folds only.

**Read the epoch rule before you read any number here.** The fixed `--epochs`
budget is the default and selects no epoch. **A different budget N is a
different rule**, so still run a matched control.

## Which era a number came from

An untagged claim is a proposal, not a measurement.

| tag | era | how to treat it |
|---|---|---|
| **E1** | `archive/2026-06_fixed-test` — 29 runs | Different metric, different data, two revisions ago. **Not a verdict.** Re-establish or don't cite. |
| **E2** | `archive/2026-08_cv-medium-v1` — 30 runs | Right metric, wrong data and roster. Directions survive; numbers don't. |
| **E3** | `archive/2026-09-08_cv-medium-v2` — 24 runs | Right metric family, but pre-revision data, `Dropout(0.2)`, and a mix of three epoch rules. Directions survive; numbers don't. |
| **E4** | current `log.jsonl` — 2026-09-11 on | Directly comparable. |

`temporal-context` was a clear E1 negative and, rerun as `context-stack` in E2,
became the largest gain in the log. **E1/E2/E3 negatives are leads, not
answers.**

## Standing facts a proposer needs

- **MDE is ~0.027 on the headline** at n=1. Anything smaller is unreadable from
  one run. Prefer dose ladders to single comparisons.
- **A single fold resolves ~0.09, not ~0.027.** `buzz_frames` is not the sample
  size; folds hold 14-42 buzz *events*. `tools/eval_sampling_sd.py <control>
  --other <exp>` gives each fold's delta its own SD; training stochasticity adds
  a smaller ~0.016-0.026 per fold delta on top (`docs/judging-results.md`).
  Propose against the headline; use folds to ask *where* an effect lives.
- **Context/averaging levers have a standing prior against them: four for
  four, they lift rich folds and leave hard folds flat-to-down.**
  `context-verify`/`context-stack` (+0.085, "three times landed on rich folds
  only" per the era-open summary above), `asym-context-yamnet` (+0.046, 6/8
  folds but both `1_95`/`1_114` flat-to-down even with an explicit-contrast
  design), and `trunk-ft-v3`'s `trunkctx-ft-1e5` (flat, +0.002 over plain
  fine-tuning) all show it. Mechanism: this era's hard folds (`1_95`, `1_114`,
  `1_150`) fail on isolated ~1 s events; averaging or concatenating a wider
  window washes out exactly that signal while flattering `1_29`/`53`'s
  254 s-median sustained drones. This is backwards from the era's goal (hard
  folds are the target, not rich-fold noise). A new context/broadcast/mean-pool
  proposal (Perch included — a naive mean-pooled window has the same
  structure) must argue past this prior: prefer a design that preserves the
  transient (max-pool, attention, or an explicit local-vs-baseline contrast)
  over another plain average, and treat `1_95`/`1_114`/`1_150` as the falsifier,
  not the headline.
- **Never select an epoch on the held-out fold.** Any rule that reads the
  scored fold's own curve reports max-over-epochs of the graded statistic. A
  cross-fold rule stays available offline via `tools/honest_epoch.py`.
- **`_quiet` buzz leaves the score, not the training pool.** Headline is
  `sensitivity_exclquiet`; every run carries per-tier columns. **A lever that
  moves neither `untagged` nor `loud` is not a detection gain.** `1_29` and
  `53` carry 254 s-median `_background` drones no other fold has — read their
  `untagged` column, not their headline.
- **Per-site *score* transforms are inert under this metric.** Each fold's
  threshold is set on its own negatives, so any strictly increasing per-fold
  transform leaves `sens@fpr0.005` bit-identical. Do not queue calibration,
  temperature scaling, per-site z-scoring of scores, or label-free thresholds.
- **Any input normalisation moves the optimum** (four for four:
  `input-standardization`, `standardize-blocks`, `recorder-center`,
  `yamnet-aves-standardize` — the last negative at a fixed budget). A linear
  readout already rescales blocks itself.
- **Reading neighbouring cache rows leaks; reading neighbouring audio does
  not.** Frames inside one snip almost always share a label. Anything that
  lets a frame see adjacent cached *rows* (score smoothing, embedding stacking)
  inherits `context-stack`'s inflation. Embedders that need neighbouring audio
  set `context_frames` (on `main` since `context-frames-fix`) and read it from
  the contiguous buffer.
- **Controls go stale with the annotations.** Loudness tagging is ongoing, so a
  control trained days earlier scores a different buzz-event pool
  (`aves-p3` was caveated for exactly this). If the set's annotations moved
  since your control ran, rerun the control under the same `--name` suffix
  rather than joining across the change.
- **A clean extraction currently fails** `read_fold_roles` on
  roster drift (`…/2026-07-27/1_99` is in `folds.csv`
  but not `annotations.csv`). The known workaround is an empty directory of
  that name under the new embedder's cache (`pitchshift-decimate-up`
  notes.md). Do not edit `01_annotate/`.
- **Embedders on disk**: real encoders `yamnet`, `aves`, `perch` (Perch needs
  TF 2.21 — extract in `.local/venv-perch-extract`, **never** `pip install`
  into `buzzdetect-train`). Derived: `yamnet_aves`, `yamnet_aves_mid`,
  `yamnet_aves_p3`, `yamnet_context`, `yamnet_context_aves`,
  `yamnet_pitchshift`, `yamnet_pitchshift_decimate`,
  `yamnet_pitchshift_context`, `yamnet_pitchshift_aves_mid`,
  `yamnet_combined`, `yamnet_doublerate`, `yamnet_bandpass`. There is **no**
  BEATs/EAT/BirdMAE/CLAP/BirdNET embedder — an idea naming one is proposing to
  write and validate a new embedder. `buzzdetect-train` already has
  `torch 2.13`, `torchaudio 2.11`, `transformers 5.16`, `librosa 0.10.2`.
- **Disk**: 110 GB free (2026-09-14); the 4352-d lead cache is 1.4 GB, so a
  per-token AVES cache (tens of GB at most) is affordable. Check `df` anyway.

### The hard folds, re-measured on the lead (2026-09-14)

**Low thresholds are good**: a fold's threshold is where its own negatives sit
at 0.5% FPR, so a *high* threshold means negatives are scoring like buzz. On
`pitchshift-aves-mid`, four folds lag, and they fail in **three different
ways** — never treat them as one item.

| fold (crop) | lead sens | threshold | what sets it (negatives above threshold) | kind |
|---|---|---|---|---|
| `1_95` (blueberry) | 0.170 | **-0.97** | 33 frames, plane-heavy mix (`mech_plane` 9 alone, more in mixtures) | **FP: aircraft** |
| `1_114` (senna) | 0.210 | **-0.72** | **21 of 21 are `ins_trill`** | **FP: orthopteran trill** — new |
| `1_37` (chicory) | 0.410 | **-1.00** | **20 of 22 are `ambient_background`** | **FP: "background"** — new |
| `1_150` (apple) | 0.468 | -2.07 (good) | diffuse, 23 frames | **positives**: genuine low-SNR buzz |

Every other fold sits at -1.57 to -2.07. `1_114`'s trill confusion **explains
this era's fold pattern at `1_114`**: pitch-shift was +0.181 there, the tile
seam's removal +0.034/+0.043, and adding AVES layers to pitch-shift took it
back down -0.204. Those counts are one draw on 21-34 frames each — a
mechanism lead, not a measurement to optimise.

**How to reproduce the census** (it bit the agent who first ran it): the
thresholds in `folds_sx.csv` are on the **raw activation** scale, as is
`predictions.csv`, but `predictions.csv` has no label column. `surprisal/<fold>/
*_surprisal.csv` has `label`, but its `activation_ins_buzz` is **sigmoid(raw)**
and its rows are in a **different order**. The sorted values match exactly, so
take `logit` of the surprisal activation and compare that to the threshold,
using only rows whose `label` does not contain `ins_buzz`.

---

# Queue

Ranked best-first by expected value on the headline and the hard folds, cost second.

## 27. Distill frontier: fill the front end × trunk × class-subset grid (stage 5)

*Special request from Luke, 2026-09-30. Run it as **one batch**: a single
`05_distill/main.py` invocation, not split across loop iterations, and ahead of
everything else in the queue. Not an autoresearch CV experiment: it touches
only `05_distill/`, never stage 3.*

The grid is front end (`yamnet`, `twofast32`, `fast32`, `fast32lo`, `fast32h16`)
× trunk (`a0.25`, `a0.375`, `a0.50`) × class subset (`ins_buzz+ambient_rain+human`,
`ins_buzz` alone): 30 cells, rung B, seed 1, teacher `v4-ft-ps-e60-moderate`.
Seven are already run, so 23 remain (9 rain+human, 14 buzz alone). The YAMNet
front end uses `select` init, as in the existing rows. Why: the frontier is
partly an artefact of which cells were tried (see `05_distill/FRONTENDS.md`).

```bash
tools/launch_job.sh 05_distill/data/main_stage5_grid.log -- \
  <python> 05_distill/main.py --teacher v4-ft-ps-e60-moderate --rung B --runs "\
fast32:a0.25:classes=ins_buzz+ambient_rain+human fast32lo:a0.25:classes=ins_buzz+ambient_rain+human \
twofast32:a0.25:classes=ins_buzz+ambient_rain+human yamnet:a0.25:select:classes=ins_buzz+ambient_rain+human \
fast32:a0.375:classes=ins_buzz+ambient_rain+human fast32lo:a0.375:classes=ins_buzz+ambient_rain+human \
twofast32:a0.375:classes=ins_buzz+ambient_rain+human yamnet:a0.375:select:classes=ins_buzz+ambient_rain+human \
fast32lo:a0.50:classes=ins_buzz+ambient_rain+human \
fast32h16:a0.25:classes=ins_buzz fast32:a0.25:classes=ins_buzz fast32lo:a0.25:classes=ins_buzz \
twofast32:a0.25:classes=ins_buzz yamnet:a0.25:select:classes=ins_buzz \
fast32:a0.375:classes=ins_buzz fast32lo:a0.375:classes=ins_buzz twofast32:a0.375:classes=ins_buzz \
yamnet:a0.375:select:classes=ins_buzz \
fast32h16:a0.50:classes=ins_buzz fast32:a0.50:classes=ins_buzz fast32lo:a0.50:classes=ins_buzz \
twofast32:a0.50:classes=ins_buzz yamnet:a0.50:select:classes=ins_buzz"
```

**ETA ~10.5 h (range 9-13 h)** from `wall.txt` of the finished runs: a0.25 ~17 min,
a0.375 ~23 min, a0.50 ~28-60 min. The YAMNet-front-end runs are doubled (its one
subset run took 56 min at a0.50). `c-buzz` has one timing sample (22.6 min at
a0.375). One GPU job at a time: start it only when no other GPU job is running.
Check `main.py --dry-run` with the same `--runs` first: it should list 23
pending and skip none of the seven done. Resumable: rerun the same command.
When done, `ladder_record.py frontier` and `frontier_svg.py` show the filled
grid; report it to Luke, with per-cell times, before proposing anything further.

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

**First pass, 2026-09-28** (`diagnostics/2026-09-28_int8-ptq/`, target
`v4-ft-ps-e60-moderate`, the true best): plain per-channel static int8 flips too
many detections. Scored as detections gained/lost at logit > 0 on the exported
graph (Luke's parity metric), `ins_buzz` versus fp32 on 1,124 reference positives:
entropy calibration all convs +36/-75, min/max +27/-276, weights-only int8
+4/-32, fp16 sibling 0/0. So ~3% is the weight-rounding floor and activations add
the rest; depthwise convs are not the main source. Speed unmeasured: this box has
no AVX2/VNNI (int8 ran 0.45-0.54x here). Convs are 71% of CPU time and the STFT
front end 19%, so ~1.9x end-to-end is the ceiling. Open, in order: fix the
skip-layer sensitivity sweep (mislabeled in the first pass), AdaRound / bias
correction, re-derive `activation_centers` on the quantized outputs, then
sensitivity-at-fixed-FPR per fold, then time it on modern hardware (Luke offered).
Not closed: the falsifier's speed clause has not been tested.

---

# Needs Luke

## 25. `1_37`'s threshold is set by `ambient_background` — a 22-frame listen list

*Evidence: **E4** census above, one draw.*

20 of the 22 negatives setting `1_37`'s threshold (chicory, 0.410 on the lead)
are annotated `ambient_background`, and the fold's threshold (-1.00) is second
worst. Two readings with opposite fixes: they are unannotated faint buzz (an
annotation question, which only Luke can answer), or chicory's background is
genuinely buzz-like (a representation question, item 15's territory). Build
the list — ident, snip, offset, raw activation — with the census recipe in the
standing facts, commit it under `diagnostics/`, and ask Luke to listen. No
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

Full reasoning is in `log.jsonl` and each branch's `notes.md`.

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
