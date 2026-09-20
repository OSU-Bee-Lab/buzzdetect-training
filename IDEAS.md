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

Ranked best-first by expected value on the headline and the hard folds, cost second. **Item 18 is a free diagnostic: run it while item 19's extraction runs**, not before it — it only changes how 19 is read. Item 23 is high-ceiling but blocked on Luke, which is why it sits low.

## 2. Perch as a further block — the only representation measured to separate the jet

*Evidence: **E3** for the separation numbers
(`diagnostics/2026-09-09_jet_1_95/README.md`); Perch 2.0 is the strongest
linear-probe encoder on BEANS (arXiv:2508.01277) and trains on insects.
Retargeted 2026-09-14 onto the lead.*

Held-out jet probe, P(wave-1 jet frame outranks a `1_95` buzz frame), lower is
better, with a same-fold control: yamnet 0.860 (2.6x control), aves 0.558
(1.5x), **perch 0.239 (1.2x)**. `1_95` is still the lowest fold on the lead
(0.170), and its threshold is still aircraft-set.

**`perch-probe`'s E3 verdict is not evidence against this.** It died of a
frame-length confound: `overlap_event_s` scales with frame length, so Perch's
5 s frame demanded 1.0 s of overlap and per-fold label dilution predicted the
delta. This design keeps the lead's frame grid and labels.

**2a — broadcast, no Perch re-extraction.** Both caches are framed over the
same snips, so 1.0 s frame *j* lies inside 5 s Perch frame *j // 5*. Join at
train time: `[lead(j), perch(j//5)]`. Verify the index map first
(`extract.py`'s per-chunk padding, clamp the last `j//5`). Frame count and
labels are identical to the lead's, so the lead is the control.

**2b — centred Perch, only if 2a pays.** Perch on a 5 s window centred on each
1.0 s frame, extracted in `.local/venv-perch-extract` (TF 2.21).

*Falsifier:* **`1_95` is the test, not the headline.** A headline gain with
`1_95` inside its delta SD is a rich-fold gain. Broadcasting also adds ~2.5 s
of real context each side, so compare the per-fold signature against the
honest `yamnet_context` pattern (rich folds up, hard folds flat) before
crediting Perch.

## 20. All 12 AVES layers, then a learned layer mix

*Evidence: **E4** `aves-mid` (layers 6/9/12 beat layer 12 alone, +0.026 to
+0.032, two draws). Literature: multi-layer probing gains consistently over
last-layer across every transformer tested (arXiv:2605.10494, 2026, BEANS
+0.08 accuracy, BirdSet +0.03 mAP; code at github.com/earthspecies/avex);
SUPERB-style softmax layer weighting is the standard form. Untagged here.*

Layers 6/9/12 were picked by argument. Nobody has looked at which AVES layers
carry buzz, and the fold pattern says it matters: `aves-mid` gained on `1_150`
but costs `1_37` and (inside the lead) `1_114`. Two legs.

**20a — layer profile, offline.** Extract a cache with all 12 transformer
layers mean-pooled (`12×768 + 2048` YAMNet blocks ≈ 11.3k-d, ~3.6 GB) under a
new embedder name. Then, offline, for each layer: YAMNet+shifted blocks plus
**that one AVES layer**, linear readout, real rotation (train on the 43-fold
pool, score each rotating fold with `03_train/metrics.py`). Plot headline and
`1_150`/`1_114`/`1_37`/`1_95` against layer depth. This costs one extraction
and no Keras CVs. It is a map, not a verdict — sklearn will not match the Keras
probe's numbers, only its ranking.

**20b — learned mix, in-pipeline.** Add a head option (e.g. `--layer-mix 12`)
that splits the AVES part of the input into 12×768 blocks, takes a softmax-
weighted sum (12 scalars), concatenates with the YAMNet blocks, then
`Dense(15)`. That is 2048+768 inputs to the dense layer instead of 4352, with
the depth choice learned. Smoke-test with `tools/smoke_model.py` and check the
ONNX export path, since this changes the graph. Log the learned weights per
fold — it is the finding.

*Control:* `pitchshift-aves-mid` (and its repeat). *Cost:* one extraction; 20b
is one CV at a narrower dense layer than the lead.

*Falsifier:* if 20a's best layers are 6/9/12 and 20b's weights put ≥70% mass on
them with a headline inside MDE, `aves-mid` already found the depth answer —
close the line. If 20a finds a different peak (e.g. layers 3-5), run the
3-layer concat of those instead of 20b.

## 21. Attentive pooling over AVES tokens — the transient is being averaged away

*Evidence: **E4** `aves-p3` (mean/max/std pooling, +0.014 ± 0.008, broad, the
right direction but small). Literature: attentive probes lift SSL
transformers far more than linear ones (AudioMAE 84→97, BEATs 94→98 AUROC on
BEANS) but **not CNNs** (EfficientNet: mean pooling won) — arXiv:2605.10494;
class-prototype pooling over patch tokens beats both on localized multi-label
events — arXiv:2509.24901 ("Unmute the Patch Tokens"). Untagged here.*

AVES emits 49 tokens per 1.0 s frame (20 ms stride). A ~1 s quiet buzz that
occupies part of a frame, or a jet that occupies all of it, look alike after a
mean. `aves-p3`'s fixed max/std was the handcrafted version and moved a
little; a learned attention query is the version the literature says pays on
transformers. `1_150` (short, faint, genuine buzz) is the fold this is for.

**Build.** An embedder that emits the **token sequence** of one AVES layer
(default layer 9; use 20a's winner if it has landed), shape 49×768, fp16.
Budget roughly 75 KB/frame, i.e. single-digit GB for `medium`. Check `df`
first; if tight, pool tokens 4x in time (12×768). Head: one learned query,
multi-head attention pooling (4 heads) with dropout inside the attention (the
paper's probe uses it), output concatenated with the 2048-d YAMNet blocks →
`Dense(15)`. This needs `03_train` to accept a 2-D per-frame input; keep it
behind a flag so every other embedder takes the byte-identical path.

**Do not apply this to YAMNet.** The same literature finds attention pooling
does nothing for CNN encoders, which is also why item 14 was parked.

*Control:* the lead's AVES part is 3 layers; the one-variable control is a
run with the **same single layer mean-pooled** (`[yamnet, shifted, aves_l9
mean]`, 2816-d). Run both; the pair isolates pooling from layer choice.

*Cost:* one extraction + two CVs. The attention head's per-epoch cost is
unknown — this is exactly the run the first-fold gate exists for.

*Falsifier:* `1_150` must move above its delta SD **and** `untagged` must
rise. If attention lands within ~0.014 of mean pooling (`aves-p3`'s size),
pooling is not the bottleneck on this data; close it.

## 15. PCEN as a parallel block — speculative, high ceiling, near known failures

*Evidence: **untagged proposal**; the nearest E1 relatives (bandpass, mel
masking) are catastrophic negatives.*

Per-Channel Energy Normalization (Lostanlen et al., arXiv:1911.00417) is an
adaptive per-channel gain whose time constant defines "background". Its
published effect is the FP-limited folds' failure: 50x fewer false alarms
near-field, 5x far-field. A ~90 s jet, a stationary trill chorus (`1_114`) and
long ambient spans (`1_37`) are all background at a 1-2 s time constant; a
~1 s buzz is not. **Only defensible as a parallel block** (a PCEN-frontend
YAMNet concatenated with the unmodified block), because replacement frontends
corrupt the input distribution YAMNet expects. `librosa.pcen` is in the
pinned env.

*Cost:* one YAMNet-class extraction + one CV at 2048-d against plain
`yamnet`. *Falsifier:* measure the threshold-setting census (standing facts)
on `1_95`, `1_114` and `1_37`. If the aircraft/trill/background share of
threshold-setting negatives doesn't fall, PCEN isn't doing its job, whatever
the headline says.

## 23. A 2025-26 bioacoustic encoder in place of AVES-v1 — **ask Luke first**

*Evidence: literature only. "What Matters for Bioacoustic Encoding"
(arXiv:2508.11845, ICLR 2026): SSL pretraining followed by **supervised
post-training on a mixed bioacoustic + general-audio corpus** gives the best
in- and out-of-distribution results, across 26 datasets including detection.
Its checkpoints (`EarthSpeciesProject/esp-aves2-*` on Hugging Face: BEATs,
EAT and EfficientNet-B0 variants, `-bio` and `-all`) are the direct
successors of the 2023 `aves-base-bio` this project uses. The 2026
comparative review (arXiv:2508.01277) ranks BEATs-NLM second only to Perch 2.0
on BEANS linear probing.*

`aves` is a 2023 HuBERT model, self-supervised only. The AVES block has been
the only representation to move `1_150`, which makes its generation the
obvious thing to upgrade.

**Blocking questions for Luke** (surface them, don't decide them):
1. **Licence.** The esp-aves2 cards are **CC-BY-NC-SA-4.0**. Check that against
   what buzzdetect ships and against the licence `aves-base-bio` already
   carries. If they match, nothing new is imposed.
2. **Dependency.** Loading is via `pip install avex`. **Never into
   `buzzdetect-train`** (TF pin hazard). A separate venv, like
   `.local/venv-perch-extract`, used only for extraction.
3. **Shipping.** The deliverable is one ONNX graph; BEATs' fbank front end
   has not been exported here. Verify export before any CV result is worth
   having.

**Design, once cleared.** Start with `esp-aves2-sl-beats-all` (the recipe the
paper names as strongest). One variable against `aves-mid`: `[yamnet(t),
beats layers 6/9/12 mean-pooled]` = 1024 + 2304 on the 1.0 s grid, control
`aves-mid-repeat`. BEATs patches a 10 ms-hop fbank into 16-frame patches, so a
1.0 s input yields only ~6 time tokens. Check a `lite` extraction produces
sane shapes before `medium`. If it pays, the swap into the lead is the
follow-up, not part of this run.

*Falsifier:* `1_150` is where AVES earned its place. A headline gain with
`1_150` flat is a different lever from the one being replaced; say so.

## 7. Nearest-neighbour search into genuinely unannotated audio

*Evidence: **untagged proposal**; the annotated-frame half is done
(`tools/annotation_triage.py`, `diagnostics/2026-09-13_annotation-triage/`).*

`02_set` only embeds annotated audio, so no cached embedding exists for a
single unheard frame. The open piece: one extraction pass over full-day
recordings, then vector search from confirmed hard cases (`1_95` jet frames,
`1_114` trill frames, `1_150` quiet buzz) into that audio, producing a reading
list for Luke. Scope it when full-day raw audio access is worth an extraction.

---

# Needs Luke

## 8. night-negatives — blocked on a data decision

*Evidence: **E3** disk survey, 2026-09-10.*

Every nighttime detection is false by construction, so in-domain hard negatives
generate at any volume from the deployments' own recorders. Night FPs sit in
the same ten minutes of `1_95` for two models two eras apart. **An experiment
may not add or edit an annotation effort unilaterally.** On disk:
`01_annotate/2025-06-07 night detections/` has 278 hand-labelled rows but no
`folds.csv`; `2026-05-26 Automatic Annotations/` has a complete `combine.R`
but unsynced inputs, and labels all 1500 as `ambient_background` — which the
2025 hand labels say is right 0.7% of the time (46% tree frog, 22%
`ins_trill`). **Open question:** is
`data/raw/Luke - Various Opportunistic Recordings/` syncable? Volume should be
a dose ladder (0x / 1x / 4x a fold's negatives), not one level.

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
