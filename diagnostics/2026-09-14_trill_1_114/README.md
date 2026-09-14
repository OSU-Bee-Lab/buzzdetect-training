# Which block carries the trill confusion at `1_114` — 2026-09-14

IDEAS.md item 18. `pitchshift-aves-mid` trades `1_114` away (-0.204 vs plain
`yamnet_pitchshift`), and the census in IDEAS.md found its threshold is set
entirely by `ins_trill` negatives (21/21). Before spending a CV on a fix
(item 19), find out which representation block confuses trill for buzz.

## Method

`probe.py`. On the `yamnet_pitchshift_aves_mid` cache, for each 4352-d vector's
block (`yamnet` [0:1024], `shifted` [1024:2048], `aves6/9/12`
[2048:2816]/[2816:3584]/[3584:4352], and `full`), fit a linear readout on every
fold except `1_114` (label-smoothing-free logistic regression via
`weighted_cross_entropy_with_logits`, same recipe as the jet probe
`diagnostics/2026-09-09_jet_1_95/emb_probe.py`), score `1_114`, and report
`P(a trill frame outranks a buzz frame)` — lower is better separation.
Repeated for `yamnet_pitchshift_decimate`'s own shifted block (0.96 s grid).

**Correction to the item's planned control**: item 18 asked for a control
`P(any other 1_114 negative outranks a buzz frame)`. `1_114` has **zero**
non-trill negatives — every one of its 4817 frames is either `ins_buzz` (521)
or `ins_trill` (4817, i.e. *all* non-buzz frames, confirming the census's
21/21 wasn't a small-sample artifact: the whole recording is buzz-over-trill).
The control is undefined for this fold; there is nothing to compare against
within it. Read the raw `P(trill>buzz)` instead — no ratio.

## Results

```
                dim   P(trill>buzz)   train_trill_frames
yamnet          1024  0.163           16834
shifted         1024  0.049           16834
aves6            768  0.204           16834
aves9            768  0.228           16834
aves12           768  0.239           16834
full (all 5)    4352  0.114           16834
decimate_shifted 1024 0.044           17535
```

(`n_buzz`=521, `n_trill_neg`=4296 for the full-vector cache; trill coverage in
the training pool is not a data problem — 16834 frames is ample.)

## What it decides

**The AVES blocks are the worst trill/buzz separators (0.20-0.24), the
YAMNet-shifted blocks are the best (0.044-0.049), and concatenating them
together (`full`, 0.114) lands worse than the shifted block alone.** This is
readable directly: a linear readout over the concatenation doesn't just fail
to inherit the shifted block's separation, it partly overwrites it — consistent
with `pitchshift-aves-mid`'s measured -0.204 on `1_114` against plain
pitch-shift, and with `hidden-aves-verify`-style dilution once AVES dimensions
dominate a linear head.

**For item 19** (decimated shift + AVES): the decimated shift block alone is
marginally the single best separator here (0.044 vs the tiled shift's 0.049),
but that gap is tiny next to the AVES-blocks' cost once concatenated. Item 19's
falsifier ("both `1_114` and `1_95` must rise") should not be read as "the
decimated shift alone fixes `1_114`" — this probe predicts a full 4352-d
concat still risks losing ground there to AVES, same as the tiled lead did.
If item 19 lands flat or negative on `1_114`, this is the mechanism, not a
surprise, and the fix is a block weight/attention scheme (item 20b) rather
than another shift variant.

**Not a coverage problem**: 16834-17535 training-pool trill frames is not
thin. The confusion is representational, on AVES's side specifically.
