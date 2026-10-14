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

## D1. The ensemble teacher through a wide student

*Evidence: `d1-ens-teacher` (2026-10-14): `twofast32:a0.50_d8` from the 3-draw ensemble teacher `v4-ft-ps-e60-moderate-ens3` is 0.698 against the single teacher's 0.701 / 0.694 (seeds 1 / 2), null; the teacher's own +0.028 does not reach that student.*

The ensemble teacher, its `<distill_cache>` targets and its packed `C__twofast32` shards exist. Untested is whether a student with less distillation error (`yamnet:a0.50_d8`, 0.735 from the single teacher, or `yam64h16:a0.50_d8`, 0.727) keeps some of the ensemble's gain. Needs its front end's shards packed under the ensemble teacher (~34 GB). Falsifier: within 0.02 of the single-teacher student.
