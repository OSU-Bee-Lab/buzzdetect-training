
## Progress at park (2026-10-06 11:15)
fast32 a0.50 trunk at ~45k/47.6k (3.6 step/s); then its 56k decay, then twofast32 trunk 23.8k->47.6k + decay,
then experiments 3 and 4 from scratch. Exp 2 ETA ~15:00; all ETA ~20:00+. When the chain is past exp 1 the
first `while kill -0 2589974` line is a no-op (pid gone), so a relaunch needs no edit.
