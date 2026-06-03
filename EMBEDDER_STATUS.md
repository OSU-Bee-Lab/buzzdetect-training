# Embedder Status

Tested against the `lite` set. `n_workers=0` (in-process) used for all TF-based embedders on macOS.

| Embedder | Framework | Status | Notes |
|----------|-----------|--------|-------|
| yamnet | TF | OK | Fully extracted |
| yamnet_bandpass | TF | OK | Fully extracted |
| aves | PyTorch | OK | Fully extracted |
| ast | PyTorch | OK (slow) | 10.24 s frames at batch=8 on CPU; ~36 files in 12 min. Will complete but takes a while. |
| panns_cnn14 | PyTorch | Untested | Model file present |
| laion_clap | PyTorch | Untested | Model file present |
| msclap | PyTorch | Untested | Model file present |
| birdnet | TFLite | Untested | Model file present; same TF stack as nighthawk, may share its deadlock issue |
| perch | TF | Untested | Has same librosa-warmup fix as nighthawk; deadlock risk unknown |
| nighthawk | TF | **Broken** | Hangs at `embed()` on first real audio. Warm-up call in `initialize()` and `TF_DISABLE_METAL=1` both tried — neither fixes it. Root cause unclear. |
