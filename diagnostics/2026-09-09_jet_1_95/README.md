# 1_95 jet false positives — diagnostic scripts, 2026-09-09

Scratch scripts from the session that diagnosed `Luke - Diel Drivers/2026-05-06/1_95`.
**Not committed, not maintained, no CLI.** Paths are relative to the repo root;
run from there with the `buzzdetect-train` env. Kept because the analysis was
expensive to assemble and every script is a few minutes to re-run but an hour to
re-derive.

`sep_test.py` is deliberately **not** here: it used a single `pickle.load()` per
file and so read one frame out of each. The cached pickles hold many appended
objects — use `dataset.read_pickle_exhaustive`, as the scripts below do.

## Run order

| script | what it does |
|---|---|
| `fp_census.py <model dir>...` | raw labels of negatives above each fold's own fpr0.005 threshold, plus the buzz/non-buzz mean-logit gap per fold. Works on any model with `surprisal/` on disk. Also on `exp/mech-margin` as `tools/fp_census.py`. |
| `fp_1_95.py` | the same FPs rebased onto Even Sample's snip files (`<ident>_s<start_filetime>`, offset within snip, wall-clock time). Writes `fp_1_95_cv_baseline.csv`, which is here. |
| `sep2.py` | loads every cached YAMNet embedding with buzz / wave-1 / wave-2 tags into `/tmp/X.npy`, `/tmp/buzz.npy`, `/tmp/tag.npy`. **Run this before `sep3`, `sep4`, `learnable`.** |
| `sep3.py` | in-sample separability: linear vs MLP-256 vs MLP-1024, all fitted *including* the wave-1 frames as supervised negatives. |
| `sep4.py` | the honest version: 1_95 held out entirely. |
| `learnable.py` | adds 1_95's own non-wave-1 negatives (incl. wave 2, the same aircraft minutes later) to the pool and re-tests wave-1 rejection. |
| `emb_probe.py <embedder>` | self-contained: the held-out wave-1 probe on any cached embedder, with a same-fold non-jet negative control. Run for `yamnet`, `aves`, `perch`. |

## What they measured

Statistic is `P(a wave-1 jet frame outranks a random 1_95 buzz frame)`; lower is
better.

    linear,  in-sample (jets supervised as negatives)   0.813
    MLP-256, in-sample                                  0.238   <- memorises
    linear,  1_95 held out                              0.855
    MLP-256, 1_95 held out                              0.809   <- does not generalise
    + 1_95's own non-wave-1 negatives, incl. wave 2     0.873   <- no transfer

Across embedders, held out, with the control that makes it interpretable:

    embedder   P(wave1>buzz)   control P(other neg>buzz)   ratio
    yamnet          0.860              0.332               2.6x
    aves            0.558              0.377               1.5x
    perch           0.239              0.206               1.2x

Perch resolves only 18 wave-1 frames (frames ~4x longer), so that row is thin.

**The ratio is a mechanism diagnostic, not a target.** Luke, 2026-09-09: "I'd
rather the target be the real world endpoint. sens@fpr captures all of this and
more." `1_95`'s sens@fpr0.005 already *is* this pathology's measurement.

Narrative, caveats and the corrections that reshaped it (the jet is **not** loud
or remarkable; level modulates within an event but not between them) are in
`IDEAS.md` under `near-chance-deployments`.
