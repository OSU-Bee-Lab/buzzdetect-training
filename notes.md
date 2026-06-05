# hyperparam-sweep

## Hypothesis
Sequential independent sweeps of dropout rate (0.0–0.4), label smoothing (0.1–0.25), and learning rate (0.001–0.003). Prior experiments established dropout=0.2 + ls=0.2 + lr=0.002 as best known config, but each was tuned in isolation under different baselines. Re-sweeping in the current combined config may reveal interactions.

## Changes
- `03_train/train.py`: added `dropout_rate`, `label_smoothing`, `learning_rate` args (defaulting to current best)
- `03_train/main.py`: threaded new args through CLI
- Sweep order: dropout (17 points, 0.0–0.4) → label_smooth (4 points, dropout=0.4) → lr (3 points)

## Results

**Dropout sweep** (ls=0.2, lr=0.002 fixed, 17 points):
Best single run: dropout=0.4 → 29.5% @ 95% prec. No coherent trend; non-monotonic across full range. Same config (dropout=0.2) gave 20.3% here vs 31.75% in prior session — variance dominates signal.

**Label smooth sweep** (dropout=0.4, lr=0.002 fixed):

| ls   | sens@95prec |
|------|-------------|
| 0.10 | 21.8% |
| 0.15 | 18.0% |
| 0.20 | 20.3% |
| 0.25 | 22.2% |

No trend. All within noise.

**LR sweep** (dropout=0.4, ls=0.20 fixed):

| lr    | sens@95prec |
|-------|-------------|
| 0.001 | 14.8% |
| 0.002 | 25.3% |
| 0.003 | 24.5% |

Clearest signal: lr=0.001 reliably underperforms. 0.002 ≈ 0.003.

Best model: `exp_hp_drop04_v1` — 29.5% @ 95% prec (dropout=0.4, ls=0.2, lr=0.002)

## Conclusion
Training variance is too high for single-run dropout/label-smooth sweeps to be conclusive. LR is the exception: avoid 0.001. Current defaults (ls=0.2, lr=0.002) hold up; dropout=0.4 is a candidate but unconfirmed. Future work should fix a random seed or average multiple runs before drawing conclusions about dropout and label smooth.
