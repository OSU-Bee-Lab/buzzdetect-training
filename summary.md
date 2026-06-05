# Experiment Summary

> Entries before 2026-06-05 lack a `main_commit` and predate several training/dataset changes. Treat their metric values as directional only.

| Experiment | Date | Commit | sens@95prec | Conclusion |
|------------|------|--------|-------------|------------|
| deeper-head | 2026-06-02 | — | 0.2225 | MLP head worse than linear probe (dataset confound — see deeper-std) |
| deeper-std | 2026-06-02 | — | 0.1935 | MLP head worse than linear probe on standard set |
| combined-embedder | 2026-06-02 | — | 0.2545 | All-class validation marginal gain; not a major lever |
| focal-loss | 2026-06-02 | — | 0.2499 | Shifts operating point, doesn't lift curve |
| focal-loss-alpha | 2026-06-02 | — | 0.2406 | Dead end: focal loss shifts not lifts at 95% precision |
| label-smooth | 2026-06-02 | — | 0.2763 | eps=0.1 significantly improves calibration |
| label-smooth-02 | 2026-06-02 | — | 0.2993 | eps=0.2 best; exceeds 0.28 target |
| label-smooth-03 | 2026-06-02 | — | 0.1415 | eps=0.3 catastrophic collapse |
| binary-translation | 2026-06-02 | — | 0.2844 | Binary translation hurts; multi-class supervision helps |
| dropout-probe | 2026-06-02 | — | 0.3175 | Dropout(0.2) + label_smooth=0.2 best known at time |
