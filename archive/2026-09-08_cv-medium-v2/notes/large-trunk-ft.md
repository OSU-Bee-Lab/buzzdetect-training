# large-trunk-ft

**Not a LOOP experiment — a one-off shipped-model build.** This trains the
current best approach (`trunk-ft-1e5`, +0.046 vs frozen on `medium`) on the
`large` set. `large` is **not comparable** to any `log.jsonl` entry: same
annotations and folds as `medium`, but extracted at `framehop_prop 0.2` (5x
frame density) instead of `1`. Per `event-level-metric` / `framehop-overlap` in
IDEAS, changing frame density moves the negative population the FPR threshold
rests on, so its `folds_sx.csv` numbers cannot be read against the CV log.

Future LOOP runs must train on `medium` only. This entry exists to record the
history of the shipped model, not to advance the loop.

## Approach

- Embedder: `yamnet_trunk` (caches `layer12_pointwise_conv_relu`, 12288-d
  float16), merged from `exp/trunk-ft` @ 38ee546.
- Head: `build_head(lr_backbone=1e-5, lr_head=2e-4)` — unfreeze YAMNet layers
  13-14 at a differential rate, BN in 13-14 frozen. Dropout(0.2) -> Dense.
- `--batch 1024`, `clipnorm 1.0`, patience 50, epochs cap 400, `general`
  translation (left as-is per decision 2026-09-01 — editing `translations/`
  would shift the `medium` baseline too).
- **Whole run on CPU** (`CUDA_VISIBLE_DEVICES=""`): the 4 GB GTX 1650 OOMs on
  `large`'s 5x frames at 12288-d, and CPU ≈ GPU speed on this box for stage 3.
  Extraction also CPU-forced.

## Runs

| stage | cmd | status |
|---|---|---|
| 2. extract | `02_set/main.py --set large --embedder yamnet_trunk` | started 2026-09-01 |
| 3. train | `03_train/main.py --name large_trunk_ft_1e5 --set large --embedder yamnet_trunk --translation general --lr-backbone 1e-5 --lr-head 2e-4 --batch 1024` | pending |

## Results

_pending_

## Conclusion

_pending_

## Harness note (2026-09-01)

Three attempts to run stage-2 extraction via Claude Code's `run_in_background`
were killed after 16-60s each (decreasing each time only because incremental
resume got further into the run before dying). No OOM (21 GB free, RSS <1 GB),
no traceback, no systemd-oomd log entry — clean external SIGKILL. The earlier
`noise-floor-cv` *training* run survived 12 min via the same mechanism, so it is
not universal, but stage-2 extraction reliably dies young under it.

Workaround: run detached via `nohup ... & disown`, log to a file in the worktree:

```
nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES="" BUZZDETECT_CHUNK_FRAMES=48 \
  MALLOC_ARENA_MAX=2 \
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python -u \
  02_set/main.py --set large --embedder yamnet_trunk --workers 0 --verbose \
  > extract_large.log 2>&1 &
```

This survives (56s+ and climbing where the harness runs died at 25s). No
completion notification — progress is polled from `extract_large.log`. Training
stage will be run the same way.
