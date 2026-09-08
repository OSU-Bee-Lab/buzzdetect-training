# Era: fixed train/validate split, retired corpus

**29 runs, 2026-06-02 → 2026-06-09.** The pre-CV era.
Closed by the CV rework in 2026-08.

## What made these comparable

| | |
|---|---|
| Eval | one fixed train/validate split, scored against a hand-curated test corpus by the since-retired stage 4 |
| Metric | `sensitivity_at_95pct_precision` (headline), plus `sensitivity_at_90pct_precision` and `precision_at_80pct_sensitivity` |
| Set | mixed provenance — a scattershot of collection methods, not day-long deployments |
| Repeats | later entries carry `n_runs`, `ci_95` and `range` from a 5-run median protocol |

## Do not trust these numbers

Stronger than the usual caveat, and the reason is on the record.

**`temporal-context` — concatenating [prev, curr, next] frames — was logged here
as a clear negative (-2pp). The identical change, rerun as `context-stack` under
CV, was the largest gain of the next era (+0.050).** A verdict inverted purely by
changing how it was measured.

Three things were wrong at once here: the metric mixes in the test corpus's base
rate, the eval was a single fixed split with no deployment structure, and the
training set pooled collection methods that a real deployment never mixes. Any
verdict below could invert the same way.

Treat these as **leads worth rerunning**, never as settled answers.

## What the era concluded

A scannable digest of the 29 entries, so proposing something already tried costs
one glance rather than a read of the log. Every verdict carries the caveats
above: the numbers are gone and only the direction is worth anything.

| Area | What was tried | Old verdict |
|---|---|---|
| Regularization | Dropout(0.2) + label smoothing 0.2 | The whole gain over an unregularized probe (3.6pp). Current default. |
| | Label smoothing 0.3 | Collapses. 0.2 was the peak of a monotone trend. |
| | L2(1e-4), alone or added | Indistinguishable from no regularization. |
| | BatchNorm on input embeddings | Clear negative (-5.8pp) — training-set running stats didn't transfer. |
| Head shape | Dense(128, relu) before output | Worse than a linear probe, replicated twice, no overfitting signature. |
| Input surgery | Bandpass 100-3000 Hz before YAMNet | Clear negative (-4.7pp); corrupts YAMNet's expected input. |
| | Zeroing mel bins above 3000 Hz | Catastrophic (-14.3pp). |
| | Handcrafted frequency features | Neutral twice; YAMNet already encodes it. |
| | White-noise samples as 'static' | Neutral; the false positives are structured, not broadband. |
| Backbone | Fine-tune YAMNet layers 13-14 at 1e-5 | Clear negative (-8.3pp); overfit, train 78% vs val 59%. |
| Class weighting | 2x buzz upweight over balanced | Negative-to-neutral; balanced weights already fine. |
| Loss | Focal loss, alpha 0.25 and 0.75 | Shifts the operating point, doesn't lift the curve. |
| Translation | Binary (all non-buzz collapsed) | Hurt; multi-class auxiliary supervision helped. Retested under CV — see `binary-translation-cv`, neutral on the endpoint. |
| Training procedure | min_delta=0.002 early stopping | 6.6x variance reduction, no mean change. Adopted as default. |
| | Forcing buzz out of validation | Negative; early stopping needs buzz in the monitor fold. |
| Temporal | [prev, curr, next] frame concatenation | Negative (-2pp) — **and now known to be wrong**, see above. |

Two have since been retested under CV, and the pair is the whole argument for
rerunning rather than deferring: `[prev, curr, next]` concatenation inverted
from -2pp to the largest gain of the following era, while the `binary`
translation's negative held up (neutral, `binary-translation-cv`).

## No `set/` snapshot exists

This era's corpus and training set were **not snapshotted before they were
retired, and are gone.** `.local/archive/04_test-corpus/` is cited in old docs;
it does not exist. Neither did `.local/archive/log_precv.jsonl`, the intended
archive of this very log — it was written to a gitignored directory and
evaporated silently, while every doc went on pointing at it.

The log survived only by accident: it was tracked in git before `3ea37de`
truncated it, and was recovered here from `01b5c75:log.jsonl`.

That failure is why the archive is a tracked directory with the set state
copied into it. See `../README.md`.

## Notes and code

`notes/` holds 19, recovered from **dangling commits** — every `exp/*` branch
from this era had already been deleted, and only `git fsck --lost-found` still
reached them. Code is at `refs/archive/<name>` (local only).

## The era splits in two, cleanly

The ten entries with `"main_commit": null` are the ten with no surviving notes.
Same ten, both ways:

> `deeper-head`, `deeper-std`, `combined-embedder`, `focal-loss`,
> `focal-loss-alpha`, `label-smooth`, `label-smooth-02`, `label-smooth-03`,
> `binary-translation`, `dropout-probe`

They were **retracted in-era**. `ls02-repro`'s conclusion reads: *"All
null-commit results invalid; typical range on current set ~0.16–0.25;
single-run estimates unreliable."* They are single runs on an unrecorded commit,
against a spread wider than most of the effects claimed. `hyperparam-sweep`
makes the same point from the other side — its headline 0.295 was *"a single
lucky run"*, with dropout=0.2 giving 0.203 in the same sweep.

Treat these ten as discarded. They are kept because the retraction is itself
information — this is where the 5-run median protocol came from.

The other 19 have a real `main_commit`, and 16 carry `n_runs` / `ci_95` /
`range` from that protocol. Those are the ones with any weight — still only as
leads, for the reasons above.

Read every entry's `conclusion` before quoting its `metrics`. Two were amended
after the fact (`dropout-probe`, `temporal-context`) and the amendment is in the
conclusion text, not a separate field.
