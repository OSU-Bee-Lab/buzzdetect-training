# hidden-context-verify

## Hypothesis

IDEAS.md item 1c, second leg: having found `--hidden 1024` null on plain
YAMNet (`exp/hidden-head-verify`, +0.012, inside MDE), test it on top of
`yamnet_context` — the era's best confirmed representation (+0.085 over the
anchor) — since E3's `yamnet-aves-head-fixed` lead (+0.030) was measured on a
concat (`yamnet_aves`), not on plain YAMNet, and the queue's own framing is
that the hidden head's case rests on whether it helps a richer representation
express what a linear readout can't.

## Changes

Same `--hidden` mechanism as `exp/hidden-head-verify` (ported by patch, not
rebase, since `main` doesn't carry the flag yet either — see that branch's
commit for the diff). No other code changes; `--embedder yamnet_context`
reuses the cache `context-verify` already extracted.

## Run

```
03_train/main.py --name hidden_context_verify --set medium --embedder yamnet_context \
  --translation general --hidden 1024 --dropout 0 --fixed-epochs 400 --verbose -y
```

Matched to `context_verify_v1` exactly except `--hidden 1024`. First fold took
**78 min** (3072-d input makes each epoch ~10x plain YAMNet's), over LOOP.md's
~50-min handoff line — wrote `HANDOFF.md` as a safety net but stayed with the
run directly rather than actually handing off, re-arming a Monitor each time it
timed out. ~8 h total.

## Results

Comparator is `context_verify_v1` (yamnet_context, h0), **not**
`cv_baseline_v3` — this leg is about whether the hidden head adds anything on
top of the concat, not about the concat itself.

| fold | context (h0) | +hidden(1024) | delta |
|---|---|---|---|
| 1_37 | 0.562 | 0.480 | -0.082 |
| 1_150 (hard) | 0.343 | 0.296 | -0.047 |
| 1_29 | 0.601 | 0.567 | -0.034 |
| 53 | 0.525 | 0.511 | -0.014 |
| 1_114 | 0.167 | 0.157 | -0.010 |
| 1_95 (hard) | 0.017 | 0.058 | +0.041 |
| 1_143 | 0.640 | 0.703 | +0.063 |
| willard | 0.466 | 0.554 | +0.088 |

5 folds down, 3 up. **Headline: 0.415 -> 0.416 (+0.001), dead flat.** Inclusive
`sensitivity`: 0.341 -> 0.339 (-0.002).

Pooled tiers: `background` 0.455 -> 0.414 (-0.041, driven by 1_29/53 both
down), `untagged` 0.440 -> 0.450 (+0.010, flat), `loud` 0.870 -> 0.820 (-0.050,
thin — 122 frames total, no real read), `quiet` 0.119 -> 0.117 (flat),
`faint` 0.0 -> 0.091 (thin, 12 frames, noise).

**The two named hard folds moved in opposite directions**, same as the
plain-YAMNet leg's split but with `1_95` now the one that *gained*: `1_150`
down -0.047 (same fold, same direction as the plain-YAMNet leg's -0.053 — two
independent runs now agree the hidden head costs this fold specifically, and
both times entirely on its `untagged` column, i.e. genuine audible apple-bloom
buzz, not quiet/faint noise — see the fold's raw labels: no
`ins_buzz_pollination` tag anywhere in it, apple isn't buzz-pollinated, so this
is plain flight/foraging buzz). `1_95` (the jet fold) gained +0.041, on its
`untagged` column too, `quiet` unchanged — a real if still-small move on the
fold that's resisted almost every other intervention.

## Conclusion

**No headline gain (+0.001, dead flat) and no consistent hard-fold story** —
the hidden head trades one hard fold for the other rather than helping both.
Combined with the plain-YAMNet leg (+0.012, also inside MDE, also negative at
`1_150`), the pattern across *two* representations is now: `--hidden 1024`
consistently costs `1_150` specifically (same tier, same direction, both runs)
while doing nothing reliable to the headline. That's a real, repeatable
mechanism worth flagging even though the topline is null — whatever the extra
capacity buys elsewhere, it's coming partly at the expense of the fold with the
thinnest, quietest positives, which is exactly the failure mode a wider
unregularized head predicts on a small-N hard fold.

Not running the `yamnet_aves` leg (the third representation IDEAS item 1c
named) this session — closing out here per instruction. `--hidden`'s case is
weak on the two representations tested; IDEAS.md is being updated to reflect
that rather than left implying the idea is still open-and-promising.

No stopping-rule confound: both arms `--fixed-epochs 400`.
