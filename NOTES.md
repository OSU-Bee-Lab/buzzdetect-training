# doublerate-std

## Hypothesis
Running YAMNet at 32 kHz (double the standard 16 kHz) with 0.48 s frames (half the standard 0.96 s) should improve detection of brief or onset-heavy buzz sounds. Finer temporal resolution means more embedding frames per second, which could capture transient buzz events that would be smeared or missed at the coarser 0.96 s frame rate.

## Evidence
- No prior standard-set results for this embedder; a lite run (`lite_doublerate`) was started but never evaluated.
- The yamnet_bandpass experiment showed that pre-processing the signal before YAMNet embedding can yield real gains on lite (13.06% vs ~0% vanilla). This suggests the embedding step is a meaningful lever.
- Temporal resolution is a well-known factor in acoustic event detection, particularly for short-duration sounds.

## Baseline
Best model on standard: `exp_dropout_v1` at 31.75% sensitivity @ 95% precision.
