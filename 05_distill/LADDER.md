# Distillation ladder: protocol (written before the first run; closed at rung B, see FRONTENDS.md)

Step budget: 7000 (measured 2.4 step/s, ~49 min per run; 12000 would be ~83 min).
Fixed for every run: student alpha 0.5 initialised from YAMNet (channel selection +
layer-wise refit), batch 512, lr 1e-3 cosine, lambda 0.1, Huber delta 1, **one
step budget for all rungs** (see `05_distill/data/<teacher>/ladder.jsonl` / the chain script
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

## Later rungs and the final choice (chain 2)

- Same advance rule at every step: advance k -> k+1 only if the level of `lost_pct` improves by more than the A repeat spread (`ladder_record.py table` prints each verdict; `gate` enforces C vs B before the expensive rung-D cache, `FORCE_D=1` overrides).
- Rungs A-C train from a local in-memory pack; rung D streams from `<cache>/_shards/` (shard-level shuffle buffer, `shards.py`), which the packed rung (~160 GB) cannot avoid. Before D depends on it, rung B is trained once through the streaming loader and must match the in-memory B within the A spread (`streamcheck`); the ladder rows use in-memory A-C and streamed D.
- Architecture variants (a0.50_d12: layers 13-14 removed; a0.375) are trained at rung B, seed 1, and timed at 20 s and 200 s. A variant is also trained at rung D only if its B `lost_pct` is within the A spread of a0.50's (not worse by more than it) AND its B headline is within 0.03 of a0.50's B headline; if both qualify, the faster at 200 s GPU.
- **After D the final choice is made by the floors, not the ladder:** GPU speed >= 1.5x YAMNet at 200 s (Luke's real chunk length), headline (`sensitivity_exclquiet` @0.005, 5 rotating folds) >= 0.207 (50% of the moderate baseline's 0.414); among candidates clearing both, the highest headline.
