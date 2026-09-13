# Annotation triage from existing surprisal.csv — 2026-09-13

IDEAS.md item 7, reduced scope. `tools/annotation_triage.py <model dir>
[--top N]` reads `<model>/surprisal/**/*_surprisal.csv` (written free during
every training run) and produces two ranked lists, no training, no
extraction, no new inference:

- **List A** — frames whose annotation does not assert `ins_buzz`, ranked by
  `activation_ins_buzz` descending: candidate hard negatives or missed
  annotations.
- **List B** — frames whose annotation does assert `ins_buzz`, ranked
  ascending: candidate faint/hard positives or over-wide annotation spans.

Full top-50 lists for `cv_baseline_v3`: `cv_baseline_v3_top50.txt`.

## Scope cut from the original idea — read before reusing this

The idea as written in IDEAS.md asked for nearest-neighbour search over
**unannotated** audio, to build Luke a reading list of what to label next. That
premise does not hold: `02_set` only ever extracts embeddings for audio that is
already annotated — the pickle filenames are the label combination present in
that snip (see `02_set/sets/medium/embeddings/yamnet/raw/**/*.pickle`) — so
there is no cached embedding for a single frame of genuinely unlabeled audio
anywhere on disk. Building that list for real means extracting a full day's
audio, which is extraction, not the free op the idea assumed.

What's implemented instead uses only *already-annotated* held-out frames
(`surprisal.csv`, from the cross-fold model that never trained on them) — an
annotation-quality audit, not a "what's next" list. It still answers the
correctness check the idea specified (see below), and List B's spread across
folds is exactly "what's already labeled that the model can't confirm," which
is useful on its own.

**IDEAS.md item 7 is trimmed, not deleted** — the untested remainder (real
nearest-neighbour search into actually-unannotated audio, which needs one
extraction pass over full-day recordings) stays in the queue.

## Result

**List A (hard negatives) — the correctness check passes cleanly.** 27 of the
top 30, and the entire top 23, are `Luke - Diel Drivers/2026-05-06/1_95` frames
labeled `mech_plane` / `mech_auto` / `ambient_background` — the known jet
flyover, exactly as item 7 predicted ("`1_95`'s jet minutes should top this
list"). The script is finding the right thing.

**List B (hard positives) is spread across every fold**, dominated by
`JamesU - MustardBumbler/1_29` (12/30) — the fold whose `_background` buzz
annotations run ~254 s median (LOOP.md's standing note on `1_29`/`53`): a
long continuous-drone span very plausibly has real quiet moments inside it that
a frame-level score correctly reads as low, which is a labeling-granularity
artifact rather than a model failure. `1_95` and `1_150` both appear too
(6 and 3 of 30) — consistent with `1_150` being a genuine low-SNR problem, and
with `1_95` having quiet buzz sitting near/under the same jet noise that
List A surfaces.

## Practical output

The un-cut half of List A beyond the jet (`1_114`, `53`) is worth Luke's eye as
candidate missing annotations. The `1_29`/`background` concentration in List B
is more likely a span-labeling granularity question than new audio to check —
worth mentioning to Luke, not necessarily worth his listening time.
