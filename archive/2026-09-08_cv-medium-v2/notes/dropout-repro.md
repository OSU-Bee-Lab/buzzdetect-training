# dropout-repro
## Hypothesis
Replicate dropout-probe (Dropout(0.2) + ls=0.2 + lr=0.002) on current medium set to validate 0.3175 claim. Expected to demonstrate dataset drift invalidated the original result.

## Changes
No code changes. Same config as current main (which already incorporates dropout-probe settings).

## Results
- Claimed baseline (dropout-probe, stale set): 0.3175
- This run (exp_dropout_repro_v1, current set): 0.172

## Conclusion
Original dropout-probe result was invalid — trained on an earlier, different medium set. 0.172 on current set. Dropout-probe entry in log.jsonl should be treated as unreliable; true best on current set is likely closer to the label-smooth-02 range (~0.30).
