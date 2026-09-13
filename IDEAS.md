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

**State as of 2026-09-11.** The anchor is **`cv_baseline_v3` = 0.330**
(excl-quiet; 0.269 inclusive) over 8 rotating folds — a bare linear probe on
frozen YAMNet, no dropout, `--fixed-epochs 400`. Everything else below is still
an E3 claim, i.e. from the era now archived in
`archive/2026-09-08_cv-medium-v2/`. Three things moved at the cutover — the
annotations were revised again (loudness tagging, still in progress), scoring
split into `sensitivity_exclquiet` (the headline) and `sensitivity`, and
`--fixed-epochs` became the default rule with dropout off. **Every float here
is now an E3 lead, not a target.**

**The first jobs of the era, in order.** They are not in the queue below because
they are not ideas, they are the anchor:

1. ~~`cv_baseline_v3`~~ — **done 2026-09-11: 0.330 excl-quiet / 0.269 inclusive,
   8/8 folds.** It is the comparator for everything below, and the only number
   in this file you can beat.
2. ~~1a~~ — **done 2026-09-11, CORRECTED 2026-09-13 (`exp/context-frames-fix`):
   `yamnet_context` is worth +0.030 ± 0.015, not the +0.085 `context-verify`
   reported.** That run trained on a cache `extract.py` had built by embedding
   label-bucketed frames, so each frame's "neighbours" were same-label frames —
   `context-stack`'s leak, now fixed. Honestly extracted: 0.330 → 0.360, and
   the shape changes — `untagged`/`loud` move +0.018/+0.015 (not +0.090/+0.092,
   so the falsifier fires), the surviving gain is `background`-tier at the four
   rich folds, and **all four hard folds are flat or down**. `context-verify` is
   now `artifact` and `hidden-context-verify` `caveated`. See `log.jsonl`.
3. ~~1b~~ — **done 2026-09-11 (`exp/yamnet-aves-verify`): `yamnet_aves`
   confirmed at almost the same magnitude as its E3 lead, 0.330 → 0.354
   (+0.024), 7/8 folds up. `1_95` moved again (+0.046, 0.052 → 0.098) — the
   4th confirmation across two eras that this is the only intervention to
   touch that fold. `untagged`/`background` tiers broadly up; `loud` dipped
   but on a thin tier (115-122 frames). `1_114` down, no mechanism. See
   `log.jsonl` for the per-fold table.
4. ~~1c~~ — **done 2026-09-12, all three legs (`hidden-head-verify`,
   `hidden-context-verify`, `hidden-aves-verify`): `--hidden 1024`'s effect is
   representation-dependent.** Null and costs `1_150` on plain YAMNet and
   `yamnet_context` (that leg is `caveated` — both its arms used the leaky
   context cache, see item 2 above); on `yamnet_aves` it is +0.022 (inside MDE) with both hard
   folds up together (1_150 +0.061, 1_95 +0.051) — see `log.jsonl` for the
   per-fold tables and mechanism notes on each.

Per-tier sensitivity is **already built in** — it is not an idea and not a
queued experiment. Every run prints it and every `folds_sx.csv` carries it.

**Read the epoch rule before you read any number here.** The audit of
2026-09-11 (`exp/pairwise-rank:notes/new-era-audit.md`) found the last era's
distortion was **undertraining, not leakage**: `val_loss` early stopping
carried no measurable selection optimism (-0.002 mean over 17 runs) but stopped
`1_150` at epoch 5-32 on every embedder tried, scoring it on a barely-trained
probe. That is fixed at the source now — `--fixed-epochs` is the default and
selects no epoch at all. The rule that survives: **a different budget N is a
different rule**, so still run a matched control.

## Which era a number came from

An untagged claim is a proposal, not a measurement.

| tag | era | how to treat it |
|---|---|---|
| **E1** | `archive/2026-06_fixed-test` — 29 runs | Different metric, different data, two revisions ago. **Not a verdict.** Re-establish or don't cite. |
| **E2** | `archive/2026-08_cv-medium-v1` — 30 runs | Right metric, wrong data and roster. Directions survive; numbers don't. |
| **E3** | `archive/2026-09-08_cv-medium-v2` — 24 runs | Right metric family, but pre-revision data, `Dropout(0.2)`, and a mix of three epoch rules. Directions survive; numbers don't. |
| **E4** | current `log.jsonl` — 2026-09-11 on | Directly comparable. The only numbers you can beat — and as of now there are none. |

`temporal-context` was a clear E1 negative and, rerun as `context-stack` in E2,
became the largest gain in the log. **E1/E2/E3 negatives are leads, not
answers.**

## Standing facts a proposer needs

- **MDE is ~0.027 on the headline** at n=1 (`probe-grid`: baseline SD 0.0095
  over n=3). Anything smaller is unreadable from one run. Prefer dose ladders to
  single comparisons.
- **A single fold resolves ~0.09, not ~0.027** — roughly 4x coarser than the
  headline, so a per-fold delta is an investigation, not a result. The reason is
  that `buzz_frames` is not the sample size: a buzz event spans many frames, and
  the folds hold 5-23 events each (`1_29`: 1972 scored frames, 14 events; `53`:
  900 in 9, one of them 79% of the total). `folds_sx.csv` carries
  `buzz_events_exclquiet`, and `tools/eval_sampling_sd.py <control> --other
  <exp>` gives each fold's delta its own SD in seconds — 0.017-0.052 from eval
  sampling, with training stochasticity larger again on top. Propose against the
  headline; use the folds to ask where an effect lives, and say "unsure" when
  the fold cannot answer. Measured 2026-09-13 (`hidden-aves-verify` r1/r2).
- **Never select an epoch on the held-out fold.** `main` no longer offers a way
  to — there is no `--monitor` flag, and `--fixed-epochs` selects nothing. The
  hazard is reintroducing one: any rule that reads the scored fold's own curve
  reports `max`-over-epochs of the statistic it is graded on. `monitor-leakage`
  marked three E3 entries `artifact` over this, and the inflation was
  **budget-dependent** (+0.021 over a 400-epoch argmax, ~+0.006 over 150), so
  quote it with its budget if you ever quote it. A cross-fold rule stays
  available offline via `tools/honest_epoch.py`.
- **The stopping rule was worth more than most levers, and it was not leakage.**
  Removing it measured **+0.031** (`yamnet_context`) and **+0.040**
  (`yamnet_aves` linear) in E3. It is now the default, so the live version of
  this fact is: **the budget N is a free parameter nobody has tuned**, and it
  moves the score at least that much.
- **`_quiet` buzz leaves the score, not the training pool.** The headline is
  `sensitivity_exclquiet`. `folds_sx.csv` also carries the inclusive figure and
  a per-tier breakdown (`sensitivity_loud` / `_normal` / `_untagged` /
  `_quiet`), all at the same per-fold threshold, and every run prints them.
  **Read the tiers before proposing anything aimed at a hard fold** — they say
  whether `1_150` and `1_95` are hard because their buzz is faint or hard on
  audible buzz too, which is the difference between an SNR problem and a
  discrimination problem, and several of the ideas below assume one or the
  other. Loudness tagging is in progress, so most frames still read `untagged`
  and the headline's denominator shrinks as it proceeds.
- **`main` now carries `--fixed-epochs` and `--dropout`, but not the rest.**
  `--hidden` lives on `exp/yamnet-aves-head-fixed`; `exp/yamnet-aves-context`
  has `--context-frames` / `--context-dims` (train-time context stacking over an
  existing cache, no extraction); `--standardize` on `exp/standardize-blocks`;
  `--epoch-rule xfold` on `exp/xfold-epoch`; the probe hyperparameter flags on
  `exp/probe-grid`. Branch from the one that already has your flag rather than
  re-implementing it — but note those branches predate the era cutover, so
  rebase onto `main` or you inherit the old default head and the old scoring.
- **Cost anchors for *budgeting a run you have not launched*** (mtimes of
  `folds/*/summary.json`, `medium`, CPU, `--fixed-epochs 150`): `yavf_h0`
  (1792-d linear) **~160 s/fold, ~13 min CV**; `yavf_h1024` (1792-d + 1024
  hidden) **~1110 s/fold, ~92 min CV**. Roughly linear in epochs and in input
  width. **This is not evidence about a run you have** — LOOP.md's first-fold
  gate still applies to every launch, including the nine-minute ones.
- **Only three real embedders exist on disk**: `yamnet`, `aves`, `perch`, plus
  the derived concats (`yamnet_combined`, `yamnet_context`, `yamnet_aves`,
  `yamnet_context_aves`, `yamnet_harmonic`, `yamnet_gain4`) and two unrun
  variants (`yamnet_bandpass`, `yamnet_doublerate`). There is **no** BirdNET,
  BEATs, AST, PANNs, CLAP or Nighthawk embedder — any idea naming one is
  proposing to write and validate a new embedder, not to flip a flag.
- **Per-site *score* transforms are mathematically inert under this metric.**
  Each fold's threshold is chosen on that fold's own negatives, so any strictly
  increasing per-fold transform of the scores moves the threshold with them and
  leaves `sens@fpr0.005` bit-identical. Score calibration, per-site
  z-normalisation, temperature scaling, label-free threshold selection — a
  large and active literature — **cannot move the headline**. (Per-site *input*
  normalisation can, and did move it: `recorder-center`, negative.) Do not
  queue a calibration idea; if score portability ever becomes the goal, it is a
  different metric and a conversation with Luke.
- **Don't quote a noise figure from this file** — run
  `tools/eval_sampling_sd.py <model dir>` (seconds, no training). Current
  per-fold bootstrap SD 0.010-0.037, headline 0.007-0.012. `1_150`'s
  run-to-run movement (0.055) is *training* stochasticity, not eval sampling.
- **The hard folds are the target**, and **they fail in opposite ways — never
  treat them as one item.** `1_150` (apple, Lynd) and `1_95` (blueberry, Big D's)
  pin 40% of the headline near zero. A gain concentrated there is the result,
  not a caveat on it.

  | | `1_150` | `1_95` |
  |---|---|---|
  | failure | **positives** — genuine low-SNR quiet buzz | **false positives** — one jet flyover |
  | negatives, p99.5 | 0.139 — normal | **0.391 — ~3x every other fold** |
  | buzz frames, mean/median logit | **0.071 / 0.064 — ≈ its own background** | 0.102 / 0.068 |
  | buzz/non-buzz mean-logit gap | — | **0.513**, against 1.578 at `1_29` |
  | frames setting the threshold | diffuse ambient | **32 of 35 `mech_auto`** |
  | fold threshold | -1.819 | **-0.401** — every other fold is -1.59 to -1.90 |
  | what has failed on it | early stopping was the bug; `--fixed-epochs` gave +0.160 to +0.172 | `harmonic-comb`, `mech-margin`, `shared-trunk-head`, `context-embedder` |
  | live leads | items 3, 4, 5 | items 2, 8, 15 |

  `context-embedder` pushed `1_150`'s threshold *down* (-1.819 → -2.107) without
  helping, so a better representation alone does not touch it — which is why
  item 3 attacks the *pooling* rather than the encoder. `yamnet_aves` is the only
  intervention ever to move `1_95` (+0.014 / +0.024 over two draws); it is still
  near chance.
- **Read `best_epoch` alongside every delta.** If your treatment moved it, the
  delta is confounded with a stopping change — re-score both arms at a common
  epoch (`tools/honest_epoch.py`, no training) before believing it.
  `recorder-center` produced a textbook +0.082 hard-fold gain at `1_150` that
  was entirely its `best_epoch` going 26 → 213; under one rule on both arms it
  is -0.007. **The question is whether the treatment could have selected on the
  reported metric.**
- **Any input normalisation moves the stopping epoch.** Three for three
  (`input-standardization` E2, `standardize-blocks` E3, `recorder-center` E3).
  Pair the next one with a fixed epoch budget from the start.
- **Frames inside one snip almost always share a label.** Any lever that lets a
  frame see its cached *neighbours' rows* — score smoothing, embedding
  stacking — inherits `context-stack`'s inflation (`artifact`, +0.050 → +0.022
  honest when rebuilt from real neighbouring **audio** instead). Reading
  neighbouring audio at extraction time is honest; reading neighbouring cache
  rows is not. **This rule was silently violated for a whole era**: the
  `context_frames` extraction path both context embedders' docstrings describe
  was never on `main`, so `yamnet_context` was built from label-bucketed rows
  and every number resting on it was inflated by ~0.055 (`context-frames-fix`,
  2026-09-13). It is fixed on `exp/context-frames-fix`; **merge that before
  extracting `yamnet_context_aves` again**, whose cache is honest only because
  it was built on the unmerged branch and has not been rebuilt since.

---

# Queue

Ranked. **Run item 1 first** — every comparison after it is budget-limited by
an unknown amount until it lands, and it is cheap.

## 5b. Up-shift by real time-window decimation, not tile — self-contained, like `yamnet_context`

*Evidence: **untagged proposal** (Luke, 2026-09-13); the down direction needed
nothing beyond a `samplerate` bump (see below), and a second correction same
day settled how the up direction should read its wider window.*

`yamnet_pitchshift`'s x2/x4 rungs create a real artifact: to shift up, they
resample the 0.96 s frame down to fewer samples then **tile** it back to
15360, which repeats the same compressed content twice with a hard seam at
the midpoint. `yamnet_pitchshift_down` avoids that on the way down for a
structural reason, not a cleverer trick: it declares a higher native
`samplerate` (32000, vs the pipeline's usual 16000) so one 0.96 s frame
already contains 30720 real samples, and the down-shift just **feeds the
first 15360 of them to YAMNet as-is, no resample call** — same sample count,
a genuinely shorter real span, YAMNet told it's the full 0.96 s. No wider
buffer, no extraction change beyond the `samplerate` bump; that embedder is
already extracting.

**The up direction genuinely needs more real audio than one 0.96 s frame
contains, and the honest way to say that is: the frame is 1.92 s, full stop —
not "0.96 s with a wider context."** `yamnet_context` widens by concatenating
three *separate, faithful* 0.96 s embeddings end-to-end; each 1024-d component
of its output still represents only its own true window, and the label logic
never has to answer for content it wasn't told about. This construction is
different in kind: it merges frame i and i+1's real, contiguous raw *samples*
(30720, a genuine 1.92 s when they're truly adjacent) and decimates that down
to 15360 *before* YAMNet ever runs — one opaque 1024-d vector that doesn't
correspond to any single 0.96 s interval. Row spacing can still match the
existing 0.96 s grid (so frame counts line up with the 2048-d cache for
concatenation), but whatever label that row gets under the standard grid
covers only the first half of what the embedding actually heard — a buzz
sitting entirely in the borrowed second half still shapes the vector without
being the row's own label. That is a real, if different, version of the same
dilution concern `framelength-changes-labels` already raised for a literal
wider frame (`yamnet_doublerate`'s failure mode) — say so up front rather than
waving it off with the `yamnet_context` comparison, which doesn't have this
problem. Still self-contained (no `extract.py` change — `embed()` already
receives a buffer spanning several consecutive real frames, same as every
other embedder here) and it inherits the boundary-clamp caveat `yamnet_context`
already measured (92-98% real neighbours on medium/yamnet_aves), but the
label-mismatch is new and specific to merging audio pre-embedding, and belongs
in the write-up if this gets run.

*Falsifier:* if the x2/x4 rungs' hard-fold pattern (particularly the mechanism
read on `1_95`/`1_114`) doesn't change once the tile-seam is removed, the seam
wasn't doing the work and the decimation construction is a purity improvement,
not a result mover — worth knowing either way before spending the extraction
surgery on a third up-shift family.

## 1e. Fix the epoch budget — DEMOTED 2026-09-11, premise weakened

*Evidence: **E3** — `exp/pairwise-rank:notes/new-era-audit.md`, recommendation 5.
**Partly superseded by `cv_baseline_v3`.***

> **Read this first.** The urgency below came from every E3 run being capped at
> 150 and still climbing. The anchor runs at **400** and is *not* obviously
> short: per-fold peaks scatter 116-398 and
> `mean(last 21 epochs) - mean(e300-320)` is -0.003 to +0.013 across all eight
> folds. So the era's default budget is defensible as it stands, and this drops
> below the re-verifications. It is still worth one run eventually — 400 was
> chosen by argument, not measurement, and a wide hidden head (1c) may well move
> the optimum — but it is no longer the thing blocking everything else.

The E3 evidence, for whoever runs it:

**Every `--fixed-epochs 150` run on disk was still rising at its cap.** Pooled
mean sensitivity over the last 21 epochs minus e100-120: `yavf_h0` +0.009,
`yavf_h256` +0.005, `yavf_h1024` **+0.014**, `yavf_h1024_r2` +0.011,
`yavx_h1024` +0.008. Pooled argmax is the final epoch for `yavf_h0` and
`yavx_h1024`. The e100 cost on `yavf_h1024` is -0.026 — 1.6x what the
400-epoch `xfold_context` run predicted — and whether e250 suffices for a
hidden-layer config is **untested**: no run has gone past 150 on one.

So the era's best numbers are all deflated by an unknown amount, and the
`yamnet-aves-context` amendment shows the deflation is **not equal between
arms** (over its last 40 epochs the control gained +0.019 and the arm +0.008),
which means a budget too short can invert a comparison, not just shrink it.

Run `yavf_h1024` at `--fixed-epochs 400`. One run yields the whole pooled curve,
so it answers both "does it plateau" and "what budget should everything after
this use". Do `yavf_h0` at 400 as well if the box is free — it is ~36 min.

*Cost:* no extraction. ~1110 s/fold at e150 scales to ~49 min/fold at e400,
~4.1 h total — **right on the handoff line**, so take the first-fold
measurement seriously and be ready to write `HANDOFF.md`. `--fixed-epochs 300`
(~3.1 h, ~37 min/fold) is the safe variant and still triples the evidence.

*Falsifier:* if the pooled curve's argmax is at or before e150 and the e150→e400
gain is under 0.01, the budget is settled at 150, the era's numbers stand as
written, and every later idea can keep using 150. (Note the audit's warning: a
*plain-YAMNet* fixed-budget run gaining +0.02-0.04 is the **expected** result,
not a sign something is wrong — an earlier document's falsifier had this
backwards.)

## 2. Perch as a third block — the only representation measured to separate the jet

*Evidence: **E3** for the separation numbers (`diagnostics/2026-09-09_jet_1_95/README.md`),
**E3** for the concat mechanism (`yamnet-aves`, +0.025 clean, two draws),
**untagged** for the composition.*

The held-out jet probe, run on all three cached embedders with a same-fold
non-jet control — statistic is P(a wave-1 jet frame outranks a random `1_95`
buzz frame), lower is better:

| embedder | P(jet > buzz) | control | ratio |
|---|---|---|---|
| yamnet | 0.860 | 0.332 | **2.6x** |
| aves | 0.558 | 0.377 | 1.5x |
| **perch** | **0.239** | 0.206 | **1.2x** |

Perch is the only representation in which a jet frame is not preferentially
mistaken for buzz — a measured, mechanism-level reason to expect it to move the
fold nothing else has. The literature agrees independently: Perch 2.0
(arXiv:2508.04665, 2025) trains on 14,597 species including insects, and the
2026 comparative review (Ecological Informatics, arXiv:2508.01277) finds it the
**best linear-probe transfer encoder on BEANS**, with transfer to mosquitoes,
bats and marine mammals it never trained on. A shallow probe is the regime
Perch wins in.

**`perch-probe`'s verdict is not evidence against this.** It died of a
frame-length confound this design removes: `overlap_event_s = framelength_s *
overlap_event_prop`, so Perch's 5 s frame demanded 1.0 s of overlap against
YAMNet's 0.192 s, 31% of buzz events are shorter, and per-fold dilution
predicted the per-fold delta exactly (1.22x -> -0.195; 2.9x -> +0.131 twice).

**2a — `perch-broadcast`, no extraction.** Both caches are framed by the same
`extract.py` over the same snips, so 1.0 s frame *j* of `yamnet_aves` lies
inside 5 s Perch frame *j // 5*. Join by index at train time:
`[yamnet_aves(j), perch(j//5)]` = 3328-d. Grid, frame count and every label are
**identical to `yamnet_aves`**, so it joins `yavf_h1024` directly. Verify the
index map first — `frametimes.csv` is not on disk, `extract.py`'s per-chunk
padding needs checking, and the last `j//5` must be clamped.

**2b — `yamnet_aves_perch`, only if 2a pays.** Perch on a 5 s window *centred*
on each 1.0 s frame, so neighbouring frames stop sharing a Perch vector.

*Cost:* 2a zero extraction, one CV at 3328-d h1024 = ~1.85x `yavf_h1024`
(~2.8 h at e150; re-budget from item 1). 2b needs a Perch re-extraction at ~19x
perch-probe's audio throughput, in `.local/venv-perch-extract` (TF 2.21 —
**never** `pip install` into the pinned conda env); measure the first ident,
expect hours. Control `yavf_h1024`/`_r2`, already on disk.

*Falsifier:* **`1_95` is the test, not the headline.** If `1_95` does not move
above its ~0.02 bootstrap SD while the headline rises, this is another
rich-fold gain of the `yamnet-aves-context` shape — say so and stop at 2a.

*Second caveat:* broadcasting one Perch row across five frames adds ~2.5 s of
real context each side, so a rich-fold-only gain may be `context-embedder`
again rather than Perch. Check the per-fold signature against `yamnet_context`
before crediting the embedder.

## 3. Sub-frame pooling of the AVES block — RE-AIM BEFORE RUNNING

> **Its target moved on 2026-09-11.** This was written to rescue `1_150`'s
> *quiet* positives, and quiet buzz is now **out of the headline** — catching it
> earns nothing. Do not run this as written. The mechanism (a mean over ~49
> tokens washing out a short faint event) is still sound and still worth
> testing; re-aim it at `untagged` and `loud` frames in the folds that are
> actually weak on audible buzz, and say up front which tier you expect to move.
> Check the anchor's tier row per fold before writing the hypothesis.

*Evidence: **untagged proposal**, resting on **E3** (`yamnet-aves` clean) and a
checkable property of `embedders/aves/embedder.py`.*

`embedders/aves/embedder.py` takes wav2vec2's last-layer output `(B, T', 768)`
and does `.mean(dim=1)` — a mean over ~49 tokens of 20 ms each. `1_150` is a
**positives** problem (genuine low-SNR quiet buzz; do not reopen it as an
annotation question), and 31% of buzz events are shorter than 1 s. A buzz
occupying 200 ms of the frame has its evidence divided by ~5 before the probe
sees it. Mean pooling is a matched filter for a *stationary* signal and the
worst pooling for a *transient* one.

Replace the single mean with `[mean, max, std]` over the token axis: 2304-d
instead of 768-d, so `yamnet_aves` becomes 3328-d. Max-over-tokens is the
parameter-free form of what the literature now says frozen transformer probes
leave behind: attentive pooling over patch tokens takes AudioMAE 84.5 -> 97.2
AUROC on BEANS and BEATs 94.1 -> 98.0 (arXiv:2508.01277, 2026), and multi-layer
attentive probing adds ~0.08 accuracy on BEANS over last-layer linear probing
(arXiv:2605.10494, 2026). Those are bird/multi-taxa benchmarks with far
stronger labels than ours — treat the **magnitudes** as inapplicable and the
**direction** as well supported. A learned attention pool is the follow-up, but
it needs the token sequence cached (~49x, ~12 GB — infeasible), which is
exactly why the three-statistic version is the right first move.

*Cost:* one AVES re-extraction under a new name (`yamnet_aves_p3`), ~1.0 h on
GPU with `--workers 0` + `BUZZDETECT_NO_GPU=1` for the TF half — see the
AVES-GPU note, and do **not** use `CUDA_VISIBLE_DEVICES=""`, which breaks the
torch path. Then one CV at 3328-d h1024 (~2.8 h at e150). Control `yavf_h1024`.

*Falsifier:* if `1_150` does not move above 0.037 (its bootstrap SD) in the
right direction, the dilution story is wrong and the extra dims are being read
as capacity. Check the `[mean, max]`-only variant offline on the new cache
before spending a second CV.

## 4. AVES middle layers — the closed lead reopened on measured grounds

*Evidence: **E3**, but the entry that closed it is `caveated` and superseded on
mechanism (`aves-probe` -> `aves-readout`).*

`aves-probe` closed the middle-layer sweep on CV magnitude: "a middle layer
would have to find ~0.15 sensitivity the final layer has nowhere." **That gap
was ~2.5x head artifact** — `aves-readout` recovered 0.074 -> 0.194 offline on
the *same* embeddings, and `aves-mlp-head` then showed the probe was
undertrained under a rule tuned for a sparse non-negative code. The closure was
computed against a gap that has since shrunk by more than the gap it denied.

The literature contradicts the current choice specifically: for **SSL**
encoders the *middle* layers carry the most transferable information, upper
layers only for the supervised pretraining task, and for mixed-taxa and
non-bird targets it is spread across layers (arXiv:2605.10494, 2026). AVES is
wav2vec2 — SSL, 12 layers — and we read layer 12 only, for a non-bird target.

Cheapest honest form: cache mean-pooled layers **6, 9 and 12** (2304-d) and let
the probe weight them, rather than sweeping one layer at a time.
`extract_features` already returns every layer; the current code discards
eleven of them at no saving.

*Cost:* one AVES re-extraction (~1.0 h; the forward pass is unchanged, only
what is kept), then one CV at 3328-d. Control `yavf_h1024`.

*Falsifier:* if the learned per-block weight norm (inspect `W`, as
`harmonic-comb` did) puts >=80% on the layer-12 block and the headline is inside
MDE, the final layer is the right read and this closes for real.

**Do not run 3 and 4 in the same arm** — both produce a 3328-d
`yamnet_aves`-shaped cache and would be indistinguishable. Run whichever fits
the box first; the loser is still worth a run.

## 7. Annotation triage by embedding search into genuinely unannotated audio

*Evidence: **untagged proposal**, narrowed 2026-09-13
(`diagnostics/2026-09-13_annotation-triage/`) after the surprisal-only half of
this idea turned out to be feasible for free and got run.*

**The activation-ranking half of this idea is done, and it was cheaper than
expected: zero extraction, using `<model>/surprisal/**/*_surprisal.csv`
directly.** `tools/annotation_triage.py <model dir>` ranks already-annotated
held-out frames by `activation_ins_buzz`, both directions. The high-activation
non-buzz list is dominated (27/30, all of the top 23) by `1_95`'s jet minutes —
the pre-registered correctness check — so the ranking is sound. See the
diagnostic's README for the full read, including why List B skews toward
`1_29`'s long `_background` spans.

**What that script cannot do, and what's still open:** it only ever ranks
frames that are already annotated (the surprisal file only covers held-out
*labeled* frames). `02_set` extracts embeddings **only for annotated audio** —
pickle filenames are the label combination present in that snip — so there is
currently no cached embedding for a single frame of genuinely unlabeled audio,
anywhere on disk. The original goal, a true "what to annotate next" reading
list built from nearest-neighbour search into audio nobody has heard yet,
needs one extraction pass over full-day recordings first (not the free op this
idea assumed). That extraction, plus the vector-search half over its output,
is the remaining, untested piece — scope it as its own item when full-day raw
audio access is worth spending an extraction on.

## 8. night-negatives — blocked on a data decision from Luke

*Evidence: **untagged proposal**; the disk survey is **E3**, measured 2026-09-10.*

**An experiment may not add or edit an annotation effort unilaterally.** The
shipped model has a tight false-positive spike near midnight that can dwarf the
diel trend in the focal crop (`Luke - Diel Drivers/2026-05-06` — one of the two
near-chance folds). Every nighttime detection is false by construction, so
labels generate at any volume. These are **in-domain hard negatives from the
deployment's own recorder** — the thing every `aug-*` experiment tried to
synthesize and failed at — fold-safe for free, and they scale with deployments
rather than annotation labour. The PAM literature is blunt that negatives
capturing a site's real acoustic complexity are *the* lever on false-positive
rate, and the published `buzzdetect` paper (J. Insect Sci. 2025) names passing
planes and cars as the specific failure mode.

**What is on disk:** `02_set/sets/medium/build.R:11-15` sources exactly two
efforts, `Even Sample` and `2025-06-04 original annotations` — **neither night
effort is in the set**. `01_annotate/2025-06-07 night detections/` has 278
combined rows, **hand-labeled with real identities**, but no `folds.csv`, so it
violates `MAKE.R`'s contract and cannot be sourced (warns, does not fail).
`01_annotate/2026-05-26 Automatic Annotations/` has a complete `combine.R`
(1500 night frames, `fold='auto'`, `role='train'`) but its `data/` inputs are
not synced, so it produces nothing.

**The "what label?" question is already answered by the 2025 hand labels:**
`animal_frog_tree` 127, `ins_trill` 60, `mech_hum_traffic` 26,
`mech_auto_truck` 23, `ambient_rustle` 13, `mech_auto_car` 12,
`ambient_background` **2**. `ambient_background` — which `2026-05-26` assigns
to all 1500 — is correct for **0.7%** of them; 46% are tree frog and 22%
`ins_trill`, classes the probe meets in daylight. So: a distinct
`auto_night_negative` class with its own translation row, or the shipped
model's per-class argmax. Not "call it all background".

**Corroboration for `1_95`.** Rebasing `cv_baseline`'s FPs onto Even Sample's
snips: 22 of 42 threshold-setting negatives fall in `260506_1924_s95760`
(22:00-22:05) and 11 in `_s88560` (20:00-20:05) — 79% in 2 of 24 snips, 32 of
42 `mech_auto`. Dropping just those snips' negatives takes the fold
0.034 -> 0.097. `tools/night-positives/README.txt` documents the *same* event
independently with `model_general_v3`; audio at
`tools/night-positives/260507_2200.mp3`. Two unrelated models, two eras apart,
failing in the same ten minutes.

**Blocking question for Luke:** is
`data/raw/Luke - Various Opportunistic Recordings/` syncable to this machine?
The automatic path needs it. **How much** is a dose-response (0x / 1x / 4x the
fold's existing negatives), not one volume.

## 10. Asymmetric context — feed the contrast, not the concat

*Evidence: **untagged proposal**, resting on **E3**'s `context-embedder`
(+0.061 honest) and the decoupled-head arithmetic.*

A linear readout of `[e_{t-1}, e_t, e_{t+1}]` can only take a fixed weighted
sum — it **cannot** compute "this frame stands out from its own neighbours",
because the weights are fixed at training time and the contrast is a per-frame
quantity. Yet that contrast is close to the definition of a buzz: a transient
against a locally stationary background. Feed the difference explicitly:
`[e_t, e_t - mean(e_{t±1})]` (2048-d), or
`[e_t, mean(e_{t±k}), e_t - mean(e_{t±k})]` (3072-d, matching the incumbent on
parameter count). It adds **no information and no capacity** — only changes
what is linearly available, which makes it the one input-side move that
directly attacks `mech-margin`'s and `shared-trunk-head`'s shared conclusion
that a linear readout of frozen YAMNet has no room here.

**Demoted 2026-09-11:** `yamnet-aves-context` found context and AVES **overlap**
(context was +0.058 on plain YAMNet, +0.011 to +0.015 on top of `yamnet_aves`)
and all three hard folds went down. Run this as a YAMNet-side question over the
`yamnet_context` cache; do not expect it to add to the current best config.

*Cost:* **no extraction** — an input transform in `03_train` over the existing
cache (`exp/yamnet-aves-context` already has `--context-frames` /
`--context-dims` for this shape). One CV.

*Falsifier / caveat:* the difference channel is signed and zero-centred where
YAMNet's code is 89.6% exact zeros and non-negative — a larger regime change
than `recorder-center`'s median shift, which moved only 86 of 1024 dims and
still shifted `best_epoch` up to 8x. Fixed budget from the start or the number
is confounded exactly as `recorder-center`'s was.

## 13. Context width k=2 — its negative rests on a mechanism that failed to reproduce

*Evidence: **E2** negative (`context-width`), whose stated cause **failed in
E3**; demoted 2026-09-11 by `yamnet-aves-context`.*

`context-width` took k=1 → k=2 (5120-d) from +0.022 to **-0.021** and blamed
dilution of short isolated buzz at willard. **That mechanism is dead on the
current data**: `context-embedder` *gained* +0.037 at willard, and frame
sensitivity bucketed by event span runs the wrong way for dilution (<1 s
**+0.046**, >5 s **+0.026** — short and isolated gains *more*).

But the reason to run it has weakened. `yamnet-aves-context` showed context and
AVES overlap heavily, so widening context is a lever on a representation the era
has moved past, and it would need its own matched `yamnet_context` control at a
fixed budget — **two CVs, ~3 h each at 5120-d, plus ~11 min / 1.6 GB of
extraction**. Run it only if items 2-5 all come back null and context is the
last representation lever standing. `embedders/yamnet_context_k2/` and
`extract.py`'s `context_frames` support are on `refs/archive/context-width`.

## 14. Within-frame pooling on `yamnet_trunk` — subframe-head options 2 and 3

*Evidence: **E2** — option 1's negative is an E2 negative: a lead, not a closed
door. Superseded in priority by item 3, which is the same idea on a better
representation.*

`yamnet_trunk` caches `layer12_pointwise_conv_relu` at **(6, 4, 512)** — 6 time
steps, 4 frequency bands per 0.96 s frame — and GAP averages all of it away.
Option 1 (time-max/freq-mean) was -0.007, folds split 5/6. Untried: **(2)**
log-sum-exp over time instead of a hard max, if the max's gradient path was the
problem; **(3)** keep the frequency axis — pool over time only, flatten to
2048-d. Both need the `yamnet_trunk` cache, which no longer exists for `medium`.
**Item 3 tests the same hypothesis on AVES, where the pooling is over 49 tokens
rather than 6 and the literature support is direct — do that one first.**

## 15. PCEN as a parallel block — speculative, high ceiling, near known failures

*Evidence: **untagged proposal**, and the two nearest E1 relatives are
catastrophic negatives. Read the caveat before ranking it above item 14.*

Per-Channel Energy Normalization (Lostanlen et al., PLOS ONE 2019 / DCASE 2019,
arXiv:1911.00417) replaces log-mel compression with an adaptive per-channel AGC
whose time constant defines "background". Its published effect is the `1_95`
failure mode exactly: **50x fewer false alarms near-field, 5x far-field** on
avian and marine data, by converting heterogeneous soundscapes into
near-additive white noise. A ~90 s jet drone is background at a 1-2 s time
constant; a ~1 s buzz is not.

**Why this is not `yamnet-bandpass` / `yamnet-mask`:** those *removed*
information and corrupted the distribution YAMNet's convolutions expect. PCEN
removes nothing — it is a per-channel gain — but it still changes the input
distribution, so a replacement frontend should be expected to fail the same
way. **The only defensible form is a parallel block:** PCEN-frontend YAMNet
concatenated with the unmodified 1024-d block, so the probe can ignore it.
Weigh against `harmonic-comb`, the last handcrafted spectral block tried:
inconclusive, and **+0.002 twice at `1_95`, byte-stable across two
nondeterministic runs**. Handcrafted spectral features have not paid here;
PCEN's claim is different in kind (adaptive background subtraction, not a fixed
feature), which is why it is queued and not archived.

*Cost:* one YAMNet-class extraction + one CV at 2048-d. `librosa.pcen` is in
the pinned env (librosa 0.10.2) — **no new package, no new venv**.

*Falsifier:* `1_95`'s `mech_auto` share of threshold-setting FPs. `mech-margin`
drove that share *up* (26.5% -> 39.3%) while appearing to work. If the share
does not fall below ~20%, PCEN is not doing the thing it is here to do,
whatever the headline says.

---

# Parked

- **[E3] Seed averaging inside a run.** Averaging 2-3 seeds per fold *within* one
  run shrinks the error on every future experiment for ~2-3x the compute of one
  run — cheaper than repeating whole experiments, and it compounds. It is why
  `probe-grid` needed 13 runs to read 8 levers. **Era-boundary decision** (it
  changes what "a run" means). Raised 2026-09-10; **Luke declined for now** —
  still hunting for effects large enough to clear the current floor. Re-raise at
  the next cutover. Related: `cv_baseline` is a fair draw (0.218 vs a 3-run mean
  of 0.2177), so nothing in `log.jsonl` is biased by a lucky denominator.
- **[E2] Trunk fine-tuning.** `trunk-ft-1e5` (YAMNet layers 13-14 at lr 1e-5) was
  +0.046 at 9/11 folds, the largest clearly-outside-noise result in the archive,
  and depth is settled: frozen 0.216 → 14-only 0.234 → **13-14 0.262** → 12-14
  0.229, a true interior optimum. Parked because the **standing injunction bans
  unfreezing backbone layers** (it is about *runtime*, and does not restrict head
  depth). If it is ever unparked: **do not let the frozen ranking choose the
  embedder for it.** YAMNet's frozen advantage is substantially an artifact of
  *what it is* — the penultimate layer of a supervised classifier whose 521
  AudioSet classes include `Buzz`, `Bee, wasp, etc.`, `Insect`, `Mosquito`. Once
  the representation can be reshaped, "already linearly separable" stops being
  worth anything; expect the AVES and Perch gaps to narrow and do not rule out an
  inversion. Hyperparameters found on ~11.5k frozen parameters are uninformative
  priors for ~95M unfrozen ones. AVES base fully unfrozen is likely infeasible on
  a 4 GB GTX 1650 (Adam moments alone ~1.1 GB at fp32). Measure one epoch before
  budgeting.
- **[policy] `large`-set confirmation. Forbidden without Luke asking explicitly.**
  One-time final confirmation after the structural search on `medium` concludes.
  `large` is the same annotations and folds at 5x frame density, and its
  `folds_sx.csv` is **not comparable to any `medium` number** — frame density
  moves the negative population the FPR threshold rests on. The valid test is a
  matched control on `large`, i.e. two long CV runs.
- **[E3] Frame length, isolated from embedder — the `perch-probe` follow-up.**
  The experiment worth running is **YAMNet re-extracted at a 5 s effective
  frame**, one variable. Hold **`overlap_event_s` absolute**, not
  `overlap_event_prop`, or the control reproduces the confound. Not comparable
  to `cv_baseline`, so this is two CV runs. **Item 2 is the cheaper way to ask
  the question that motivated it** (does Perch help, with the frame rule held
  fixed) and should be run first.
- **[E2] Patience 20-25 instead of 50, and [E2] `restore-on-sens`' other half.**
  Both moot under `--fixed-epochs`. Patience is answerable offline by replaying
  `EarlyStopping` over saved `val_loss_curve`s, never by a run. Restoring on the
  sens argmax was ~+0.006 frozen (inside noise); the divergence it exploits is a
  backbone-fine-tuning effect (`trunk-ft-restore-sens`: curves diverged on all 11
  folds, 8 shipping a later epoch), so retest it there and only if trunk FT is
  ever unparked. Note it is **not** `--monitor val_sens`.
- **[throughput only] Split framing from embedding in `--workers`.** `--workers`
  covers both framing (CPU) and embedding (VRAM-bound), so `--workers 1`
  needlessly serialises the framing. No effect on any metric.

# Ruled out — do not re-propose

Short entries so nobody spends a run rediscovering them. Full reasoning is in
`log.jsonl` and each branch's `notes.md`.

- **Per-site score calibration / label-free threshold selection** — inert under
  this metric by construction. See the standing fact.
- **`1_150` as an annotation-quality problem** — Luke listened 2026-09-09: "Most
  of them are very quiet, but still legitimate targets." It is a genuine low-SNR
  detection problem. Its early stopping was broken, not its model, and
  `--fixed-epochs` already recovered +0.160 to +0.172 there.
- **Leave-one-concept-out as the cause of the hard folds** — ruled out
  2026-09-05, re-confirmed 2026-09-08. r = 0.013 between sublabel coverage and
  per-fold sensitivity; every fold is at 0% thin against the real training pool.
  A rerun that appeared to overturn this had filtered to `role=='rotate'` —
  check roles before reviving it.
- **`--monitor val_sens` in any form** — see the standing fact.
- **Buzz-only epoch selectors (`auc_buzz`, `ce_buzz`)** — closed negative by
  `buzz-selector-curves`: under a non-leaking rule all four selectors return
  *identical* per-fold sensitivities, because the pooled sens curve is flat over
  e129-164. The budget is where the sensitivity is, not the selector.

**Caveat that belongs to every hard-fold claim here.** Seven `ins_buzz`
annotations spanning one 300 s file supply 63% of mustard's and 52% of
Fit+Fast's buzz seconds. Their `buzz_frames` counts overstate their independent
sample size, and part of what the metric rewards is detecting a continuous drone.
