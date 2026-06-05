# Experiment Summary

11 experiments run as of 2026-06-05. Best known result is ~30% sensitivity @ 95% precision using a linear probe with label_smooth=0.2, dropout≈0.2–0.4, lr=0.002; note that training variance is high and single-run comparisons are unreliable. Confirmed dead ends: MLP heads, focal loss, label_smooth≥0.3, lr=0.001, binary translation; read individual experiment `notes.md` on each `exp/<slug>` branch for details.
