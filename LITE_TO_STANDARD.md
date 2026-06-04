# Lite → Standard Re-runs

Experiments run only on the `lite` set that have not been evaluated on `standard`.
Because the standard set is ~2.8× larger (1706 vs 604 embeddings), lite results
cannot be compared to the standard-set baseline (25.06% sensitivity @ 95% precision).
These experiments may be promising or definitively ruled out only after a standard run.

## yamnet_bandpass

**What it is:** Hamming-windowed sinc bandpass FIR filter (100–3000 Hz, 201 taps)
applied to the waveform before YAMNet embedding. Targets the buzz-relevant frequency
range and attenuates low-frequency rumble and high-frequency noise outside it.

**Lite result:** `lite_bandpass` reached 13.06% @ 95% precision; `yamnet_bandpass_lite`
reached 7.93%. Vanilla yamnet on lite is ~0% (interpolated). So the bandpass yields a
large relative gain within lite, but the absolute numbers are far below the standard-set
baseline, making the result uninterpretable relative to production.

**Why it might help on standard:** The gain on lite is real and consistent across two
model runs. If it's correcting a genuine signal-quality problem (background hum,
high-frequency noise), that problem exists in both sets. Could compound with dropout +
label smoothing.

**Pre-work required:** Embeddings do not exist on standard — extraction (stage 2) must
run before training.

```bash
# Stage 2
conda run -n buzzdetect-train python 02_set/main.py \
  --set standard --embedder yamnet_bandpass

# Stage 3
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_bandpass_std_v1 --set standard --embedder yamnet_bandpass \
  --translation general --epochs 300
```

---

## yamnet_combined

**What it is:** Extended 1545-dim embedding (vs. standard 1024-dim YAMNet). Combines
YAMNet's internal representations to produce a richer feature vector.

**Lite result:** `lite_combined` model exists but has no test metrics — it was never
fully evaluated even on lite. Cannot assess relative to anything.

**Why it might help on standard:** More embedding dimensions could provide additional
discriminative features. Has not been properly tested anywhere.

**Pre-work required:** Embeddings do not exist on lite or standard. Extraction must run
first. Note: the embedder class declares `embeddername = "yamnet"` (not
`"yamnet_combined"`), which may cause path conflicts with standard yamnet embeddings —
verify or fix before extracting.

```bash
# Stage 2
conda run -n buzzdetect-train python 02_set/main.py \
  --set standard --embedder yamnet_combined

# Stage 3
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_combined_std_v1 --set standard --embedder yamnet_combined \
  --translation general --epochs 300
```

---

## yamnet_doublerate

**What it is:** YAMNet run at 32 kHz (double the standard 16 kHz) with 0.48 s frames
(half the standard 0.96 s). Doubles the temporal resolution of the embedding stream.

**Lite result:** `lite_doublerate` model exists but has no test metrics — never fully
evaluated even on lite.

**Why it might help on standard:** Finer temporal resolution could improve detection of
brief or onset-heavy buzz sounds. Has not been properly tested anywhere.

**Pre-work required:** Embeddings do not exist on lite or standard. Extraction must run
first.

```bash
# Stage 2
conda run -n buzzdetect-train python 02_set/main.py \
  --set standard --embedder yamnet_doublerate

# Stage 3
conda run -n buzzdetect-train python 03_train/main.py \
  --model exp_doublerate_std_v1 --set standard --embedder yamnet_doublerate \
  --translation general --epochs 300
```

---

## Priority

| Experiment | Has lite metrics | Embeddings ready | Priority |
|---|---|---|---|
| yamnet_bandpass | Yes (13.06%) | No — needs extraction | **High** |
| yamnet_combined | No | No — needs extraction + code check | Medium |
| yamnet_doublerate | No | No — needs extraction | Medium |

Run yamnet_bandpass first: it has demonstrated within-lite benefit and is the most
interpretable gap in the current experiment history.
