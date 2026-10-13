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

## D1. An ensemble teacher

*Evidence: `ps-d8-ensemble` (training arm, 2026-10-11): averaging 3 draws of the fine-tuned recipe is +0.049 ± 0.011 on `medium`'s CV, 5 members +0.066. Untested on `moderate` and untested through a student.*

A student's speed does not depend on its teacher's cost, so the ensemble gain is free at the lite tier if it survives distillation. Train N (3) draws of the teacher's recipe on `moderate`, deploy one graph that averages their logits (and pick one member's code, or the mean code, for the code-regression loss: say which), and distil the frontier's anchor students from it against their existing rows. It is a new teacher: its own `<distill_cache>/<teacher>/` targets (the `_mel` cache is shared), its own log `key`, so the matched controls must be re-judged under it or compared by headline only. Costly (N teacher trainings plus a target re-cache); a training-arm confirmation on `moderate` comes first. **Luke approved this, including its `moderate` runs** (an exception to the big-set rule). Falsifier: the student of the ensemble lands within seed noise (0.02) of the same student of one member.
