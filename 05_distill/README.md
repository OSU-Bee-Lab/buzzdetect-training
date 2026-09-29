# 05_distill

Distilling the teacher (`v4-ft-ps-e60-moderate`, two YAMNet trunk passes) into a
single-pass MobileNetV1-style student: YAMNet's architecture with channel widths
scaled by alpha. This directory holds step 0, a weight-agnostic speed check to
decide whether a lite model is worth training (floor: >= 1.5x regular YAMNet on
GPU).

- `student.py`: Keras builder. `widths_for(alpha, n_layers)` gives per-layer
  filter counts; `build_student(filters)` is waveform -> logits with YAMNet's
  layer names (so weights can later be loaded by channel selection).
- `bench_arch.py`: exports each candidate (random weights) through the deploy
  passes (`04_deploy/onnx_passes.optimize`, rename io) and times it with the
  buzzdetect engine's `make_session`, alongside the reference engine models.

## Run

```bash
tools/launch_job.sh --cpu <log> -- conda run -n buzzdetect-train python 05_distill/bench_arch.py export
tools/launch_job.sh <log> -- /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/bench_arch.py time
```

Outputs go to `.local/distill/arch/` (`<name>/model.onnx`, `results.json`).
Method matches `buzzdetect/benchmarks/model-speed`: 20 s audio, 2 warmup + 15
timed runs. CPU rates are informational only (i7-2600, no AVX2).

## Results (2026-09-28, GTX 1650, audio s per wall s)

| model | GPU | x YAMNet | x teacher | CPU | params | MMACs/frame |
|---|---|---|---|---|---|---|
| yamnet_large_general | 2970 | 1.00 | 2.44 | 293 | | |
| teacher v4-ft-ps-e60-moderate | 1216 | 0.41 | 1.00 | 152 | | |
| a1.00 | 2723 | 0.92 | 2.24 | 365 | 3,232,719 | 68.6 |
| a0.75 | 3342 | 1.13 | 2.75 | 422 | 1,835,871 | 39.1 |
| a0.50 | 4363 | 1.47 | 3.59 | 681 | 831,471 | 17.8 |
| a0.375 | 4903 | 1.65 | 4.03 | 847 | 476,439 | 10.3 |
| a0.25 | 5457 | 1.84 | 4.49 | 1006 | 219,519 | 4.8 |
| a0.50, layers 13-14 removed | 4712 | 1.59 | 3.88 | 638 | 422,127 | 15.4 |
| front end only (STFT/mel) | 6280 | 2.11 | 5.17 | 1168 | | |

The front end alone runs at 6280 s/s, so it caps any student at about 2.1x
YAMNet on GPU. Speedup saturates well before the MAC count does: 68.6 to 4.8
MMACs (14x) buys only 2.0x.
