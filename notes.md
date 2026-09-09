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

## Conclusion
