# Experiment Ideas

**Untried candidate experiments, nothing else.** When you run an idea, delete its
section entirely; the verdict lives in the run's log entry and `notes.md`. No
strikethrough, "DONE" or summary left behind. If only part is tested, trim to the
untested remainder. Facts, history and provenance go elsewhere:
`docs/training-proposer-facts.md`, `docs/distill-proposer-facts.md`, or the run's
`notes.md`.

Results: `03_train/log.jsonl`, `05_distill/log.jsonl`. Protocol: `LOOP.md`.

---

# Queue — training arm

Ranked best-first by expected value on the headline and the hard folds, cost second.

## 29. BEATs / AVES2 as an extra block on the fine-tuned trunk

*Evidence: untagged proposal. The foundation-model review (arXiv 2508.01277) finds AudioSet-pretrained SSL models beat many bird-specific ones; AVES middle layers were a confirmed gain in the frozen era (+0.026 to +0.032).*

Write a `Recipe` embedder (not a hand-written ONNX) for `EarthSpeciesProject/esp-aves2-sl-beats-all` (PyTorch, so GPU extraction like AVES, `BUZZDETECT_NO_GPU=1` per CLAUDE.md). Probe layers 6/9/12 mean-pooled, concatenated to the `v4-ft-ps` features. Check licence and model size before building, and its inference rate against LOOP.md's AVES injunction. Falsifier: no gain on `untagged`/`loud` tiers means it is another rich-fold lever (see the context prior in `docs/training-proposer-facts.md`).

---

# Queue — distillation arm

Ranked best-first by expected movement of the speed / sensitivity frontier,
cost second.

## D2. 64 bands at a 16 ms window (fft 256), hop 16 ms

*Evidence: random-weight timing in batch 34 (2026-10-15, mmt, 200 s, GPU, one session): 64 bands, 125-7500 Hz, 16 ms window, hop 16 (60 frames) times a0.25_d8 6.36x and a0.50_d8 5.14x, against `yam64h16` (25 ms window, fft 512) 4.27x / 3.79x in the same session; the front end alone 8.18x vs 5.45x. At fft 256 one of the 64 mel bands is empty (constant input) and most low bands hold one bin. A 25 ms window at hop 20 timed no faster than `yam64h16` (3.78x); window = hop = 20 ms timed 4.54x / 5.97x.*

`yam64h16` (0.711 / 0.727) and `fast32h16` (16 ms, 32 bands: 0.61-0.65) differ in both band count and window; this front end separates them, and if the 64 bands carry the gain it puts ~0.71 at 5-6x, above `twofast32:a0.50_d8` (0.70 at 5.33x). Students `a0.25_d8`, `a0.50_d8` at rung C, WSD 56000. Needs ~34 GB free on the root disk for its rung-C pack; unblocked 2026-10-15 when finished packs moved to the server disk (228 GB free on 2026-10-17). Falsifier: within 0.02 of the `fast32h16` d8 twins.
