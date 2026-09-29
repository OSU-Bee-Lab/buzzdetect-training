# Distillation ladder: protocol (written before the first run)

Fixed for every run: student alpha 0.5 initialised from YAMNet (channel selection +
layer-wise refit), batch 512, lr 1e-3 cosine, lambda 0.1, Huber delta 1, **one
step budget for all rungs** (see `.local/distill/ladder.jsonl` / the chain script
for the number), so only the data varies. Runs: rung A seed 1, rung A seed 2, rung B seed 1.

Metric: V pool (100 h, 31 held-out deployments, never trained on). Detections at
logit > 0 vs the teacher's cached logits. Primary: `ins_buzz` **lost** detections
as % of the teacher's V-pool positives (`lost_pct`); gained % and other-class
flips and mean |logit error| are reported beside it. The ladder is judged on
`sensitivity_exclquiet` at fpr 0.005 over the 5 rotating folds (eval_folds.py),
compared with the baseline, the teacher's honest rotation number and the
teacher ONNX through the harness (inflated).

## Stopping rule

- Spread = |lost_pct(A seed 1) - lost_pct(A seed 2)|; A level = their mean.
- Advance from rung k to k+1 only if lost_pct improves by MORE than the spread
  (A level - lost_pct(k+1) > spread; for later rungs, k level - k+1 level).
- Stop at the first rung that does not. No other kill rule (Luke: "just try and
  see"); the floors (>= ~1.5x normal YAMNet GPU speed; >= 50% of the baseline's
  sensitivity) are reported, not enforced.
- One seed per rung above A: a rung's improvement must clear the A spread, which
  is itself two runs, so read a marginal pass as marginal.
