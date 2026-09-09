# perch-probe

A linear probe on frozen **Perch v2** embeddings (Google's bird-vocalization
taxonomy model), same head and same training code as the YAMNet baseline.
The experiment *is* the embedder, which LOOP.md's "YAMNet only" constraint
explicitly allows.

## Hypothesis

YAMNet is trained on AudioSet — general web video, where insect sound is a
handful of coarse classes. Perch v2 is trained on a very large bioacoustic
corpus at 32 kHz, and its representation is shaped by the problem of telling
similar-sounding animals apart in field recordings with wind, rain and
distance. Buzz detection is that problem. If the embedder's training domain is
what limits the probe, Perch should be visible above the noise floor; if the
limit is the annotations or the head, it should land on top of YAMNet.

## The caveat this run carries

**Perch's frame is 5.0 s where YAMNet's is 0.96 s, so its headline is not
directly comparable to any YAMNet number**, for the same reason
`framehop-overlap` was not: changing frame density changes the negative
population the 0.5% FPR threshold rests on. This shows in extraction — a 5 s
window overlaps many annotations, so most frames come out multi-label where
YAMNet's came out clean (the log is full of `rescued ... as
ambient_background+ins_trill`). The label is diluted, the frame count drops
~5x, and both cut against Perch.

That is a real limit on what one run can conclude, and it is stated up front
rather than discovered afterwards. The honest readings are:

- **Large positive** — worth taking seriously; it cleared a handicap.
- **Flat or small** — genuinely inconclusive. It does not separate "Perch is
  no better" from "5 s frames cost what Perch gained", and the follow-up is a
  matched-frame comparison, not another verdict.

A properly matched control would need YAMNet re-extracted at a 5 s effective
frame, which is a second extraction and a separate experiment.

## Setup

- **Embedder:** `embedders/perch/`, Perch v2 SavedModel, 1536-d, 32 kHz, 5 s.
- **Set:** `medium`, `--translation general`, frozen probe, all defaults
  otherwise. One change.
- **Comparator:** `models/cv_baseline` (0.218), with the caveat above.

### Two interpreters, on purpose

Perch v2's SavedModel is a JAX export whose StableHLO artifact **TF 2.16.2
cannot deserialize** (`Cannot deserialize computation` from `XlaCallModule`).
It needs TF >= 2.21. The pipeline is pinned to `tensorflow>=2.16,<2.17` and
every model in `log.jsonl` was trained under it, so upgrading the shared env
would silently change the training runtime under an open era.

`run_perch.sh` splits the stages instead:

| stage | interpreter | TF |
|---|---|---|
| 2, extraction | `.local/venv-perch-extract` | 2.21.0 |
| 3, training | the pinned conda env | 2.16.2 |

Safe because **stage 3 never runs the embedder** — `train.py` loads it with
`initialize=False` purely to read `n_embeddings`, and reads pickled embeddings
off disk. Training stays byte-for-byte comparable with the rest of the log.
The venv is `--system-site-packages` over the conda env with only TF (and
numpy held `<2`) shadowed, so librosa/soundfile/pandas are the same builds
that cut every other set's frames.

### The embedder does not use perch_hoplite

`embedders/perch/embedder.py` calls the SavedModel's `serving_default`
directly. Two things in the released v2 make the `perch_hoplite` wrapper
unusable rather than merely optional:

- `TaxonomyModelTF.batch_embed` calls `model.infer_tf`, which the v2
  SavedModel does not export (it has only `serving_default`).
- `from_config` parses the model's class-list CSVs, and the released
  `perch_v2_ebird_classes.csv` contains a genuine duplicate (`no_ebird_code`),
  which `ClassList` rejects outright.

Everything that wrapper added on top of the raw model was framing and peak
normalization. `extract.py` already frames; the normalization (DC-remove, scale
each frame's peak to 0.25) is reproduced verbatim in `embed()`.

### The cross-env handoff is tested, not assumed

The one real risk in the two-interpreter split is that embeddings written under
TF 2.21 might not read back correctly under TF 2.16.2. Checked directly on the
files the live extraction had already written (2026-09-08, mid-run):
`load_embedder('perch', initialize=False)` gives `n_embeddings` 1536 in the
pinned env; stage 3's own `read_pickle_exhaustive` reads them back at
(885, 1536), all finite; and they fit through stage 3's exact head
(Input(1536) -> Dropout(0.2) -> Dense) for one step. Pickled float arrays carry
no TF state, which is why this holds.

Extraction runs at `BUZZDETECT_CHUNK_FRAMES=48` — Perch allocates ~1.2 GB of
intermediates at that width, and the default 300 would want ~8 GB.

## Results

Both stages completed 2026-09-09 00:08 EDT. Extraction wrote 21,201 frames
(YAMNet's `medium`: 82,064); all 5 rotations trained.

**Headline: 0.244 vs `cv_baseline` 0.218, +0.026, 4 folds up / 1 down.**
Per fold, ordered by how many buzz frames the fold actually has:

| fold | buzz frames (yam) | base | perch | delta |
|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2144 | 0.426 | 0.231 | **-0.195** |
| Lily - Fit+Fast/2023_R3_Marysville/53 | 1031 | 0.425 | 0.457 | +0.032 |
| Luke - Diel Drivers/2026-05-06/1_95 | 433 | 0.037 | 0.068 | +0.031 |
| Lily Adam - One Hive/.../1_11 | 305 | 0.180 | 0.311 | +0.131 |
| Luke - Diel Drivers/2026-04-08/1_150 | 146 | 0.021 | 0.152 | +0.131 |

The whole +0.026 comes from the two thinnest folds. The richest fold lost
0.195; the second-richest moved +0.032, inside the 0.014-0.016 noise floor.
That is the `standardize-blocks` shape.

### The dilution is not uniform across folds, and it predicts the delta

`overlap_event_s = framelength_s * overlap_event_prop`, so the labelling rule
scaled with the frame: an event must overlap a frame by **1.0 s** to label it
under Perch, against **0.192 s** under YAMNet. Median buzz annotation is
1.00 s and 31% of buzz events (1028/3310) are shorter than 1.0 s, so a third
of the buzz corpus can no longer label a grid frame at all and enters only
through the centred-rescue path -- 488 rescues here against 187 for YAMNet.

Measuring dilution as *seconds of audio carrying a buzz label*
(buzz frames x frame length), per rotation fold:

| fold | buzz-sec yam | buzz-sec perch | dilution | buzz frame rate yam -> perch | delta |
|---|---|---|---|---|---|
| JamesU - MustardBumbler/1_29 | 2058 | 2510 | **1.22x** | 28.1% -> 32.1% | **-0.195** |
| Lily - Fit+Fast/.../53 | 990 | 1195 | 1.21x | 13.7% -> 16.1% | +0.032 |
| Luke - Diel Drivers/.../1_95 | 416 | 730 | 1.76x | 5.7% -> 9.3% | +0.031 |
| Lily Adam - One Hive/.../1_11 | 293 | 860 | **2.94x** | 4.4% -> 12.3% | +0.131 |
| Luke - Diel Drivers/.../1_150 | 140 | 395 | **2.82x** | 3.0% -> 7.0% | +0.131 |

The ordering is exact: the two folds that gained +0.131 are the two folds
where the 5 s frame inflated buzz-labelled audio ~2.9x, and the fold that lost
0.195 is the fold where the label barely changed (1.22x). Set-wide the same
holds -- Perch labels 17,970 frame-seconds as buzz against YAMNet's 9,618, so
**roughly half of what Perch calls buzz is audio YAMNet calls background**,
and buzz-bearing frames rose to 17.0% of the set from 12.2% despite 4x fewer
frames. 52% of Perch's buzz frames are multi-label (YAMNet 47.5%), and the set
carries *more* distinct collapsed labels (485 vs 436) from a quarter as many
frames.

So the gain lands exactly where the positive class was made fattest and the
detection target biggest. It is a measurement of the relabelling, not of the
embedder.

### The thresholds are 5x noisier than the baseline's

Fewer frames also means fewer negatives under the 0.5% FPR point: ~5-7
negative frames set each Perch threshold, against ~24-35 for `cv_baseline`.
Per-fold sensitivities here are correspondingly less stable, which is a second
reason not to read the two thin folds' +0.131 as signal.

## Conclusion

**Inconclusive, leaning negative -- exactly the "flat or small" reading this
experiment pre-registered**, and the follow-up it named (a matched-frame
comparison) is now the only thing that would settle it.

The one piece of evidence that is *not* confounded by frame length points the
wrong way: on `1_29`, the only rotation fold where the 5 s frame left the label
close to intact (1.22x) and the only one with enough buzz frames to resolve an
effect, Perch fell from 0.426 to 0.231. The bioacoustic-domain hypothesis is
not supported by anything measured here.

Nothing to promote. The `standardize-blocks` lesson repeats: the headline moved
because the thin folds moved, and the thin folds moved because the task changed
under them.

If this is picked up again, the next step is **not** another Perch run. It is
YAMNet re-extracted at a 5 s effective frame, which isolates frame length from
embedder and would also tell us whether the coarser frame is itself worth
having. Note that a fair matched control must also decide what to do with
`overlap_event_prop`: holding the proportion fixed changes the labelling rule
along with the frame, so the control should probably hold `overlap_event_s`
absolute instead.
