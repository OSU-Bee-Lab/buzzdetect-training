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

## D2. A 64-band, hop-16 front end between `yamnet` and `fast32h16`

*Evidence: `matmul-stft` (2026-10-11): with the Pad gone, the frontier has a hole between yamnet a0.25_d8 (0.713 at 3.24x) and twofast32 a0.50 (0.691 at 4.81x), and frame rate is the largest speed term (front end alone: yamnet 3.6x, fast32 6.7x, fast32h16 9.2x). Hop 10 -> 16 ms cost ~0.035 on fast32 (0.687 -> 0.650 at a0.50); 64 -> 32 bands and the 25 -> 16 ms window are not separated.*

Register `yam64h16` (25 ms window, 64 bands, 125-7500 Hz, hop 256; 60 frames) and distil `a0.25_d8` and `a0.50_d8` on it at rung C (WSD to 56000, buzz+rain+human), exported with `mmt`. Bench it with random weights first (`bench_arch.py --seconds 200`, `+mmt`): it should land near 4-5x. Costs a `cache_fe` pass and ~25-45 GB of rung-C shards on the root disk; check free space first. Gain: a student above the line from 0.713 at 3.24x to 0.691 at 4.81x by more than 0.02. Falsifier: it lands on that line (frame rate, not bands or window, is what the fast front ends lose to).

## D3. Seed-2 repeats of the two `twofast32` d8 students

*Evidence: `twofast32-d8` (2026-10-11): `twofast32:a0.50_d8` is 0.701 at 5.33x, +0.033 over everything at its speed or faster (fast32h16 a0.75_d8 0.656 at 5.57x) and level with its full-depth twin (0.691 at 4.81x); `a0.25_d8` is 0.668 at 6.14x, +0.022 over fast32h16 a0.50_d8 (0.646 / 0.642 at 6.7x). Both single seeds, both over the 0.02 repeat threshold.*

Owed by the repeat rule before either counts: `main.py --rung C --seed 2 --wsd 7000,14000,28000,56000 --runs "twofast32:a0.50_d8:classes=ins_buzz+ambient_rain+human twofast32:a0.25_d8:classes=ins_buzz+ambient_rain+human"`, then `reexport.py --fe-impl mmt --suffix _mmt` on the new branches (`exp/twofast32-d8`'s `chain.sh` does both; ~9 h for the pair). `a0.50_d8` was still climbing at 56000 (0.687 -> 0.701), so read the seed-2 curve's last doubling too. Falsifier: seed 2 of `a0.50_d8` lands at or under 0.676, or `a0.25_d8` under 0.666 (its seed-1 margin over the bar is 0.002, so expect that one to be a coin flip).
