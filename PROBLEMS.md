# Loop Friction Log

## 2026-06-02 — find command output too large ✓ fixed

**What happened:** LOOP.md instructed agents to run `find 02_set/sets -name '*.pickle' | sed 's|/[^/]*$||' | sort -u` to identify available embeddings. This produces ~300KB of output (one line per snip-level directory), truncated in context.

**Impact:** Agent used a modified `find -maxdepth 4 -type d` instead, which surfaced empty embedding dirs as valid options.

**Fix:** Replaced with a count-per-embedder command in LOOP.md that immediately shows which embedders have zero pickles.

---

## 2026-06-02 — Empty embedding directories ✓ fixed

**What happened:** `yamnet_combined/raw/` and `yamnet_doublerate/raw/` existed as directory trees with no `.pickle` files. No marker distinguished them from populated embedders.

**Impact:** Agent attempted training on `yamnet_combined`, failed with `ValueError: no samples found`.

**Fix:** Removed the empty directories.

---

## 2026-06-02 — `models/models.py` missing from experiment worktrees ✓ fixed

**What happened:** `models/.gitignore` had `/*` which gitignored `models/models.py` (a code file, not an artifact). Worktrees don't get the file, so `04_test/inference.py` fails with `ModuleNotFoundError: No module named 'models.models'` when the test is run from a worktree.

**Impact:** Stage 4 (test) fails in the experiment worktree.

**Fix:** Added `!models.py` to `models/.gitignore` and tracked the file. Also clarified in LOOP.md that stage 4 must be run from the main worktree (where `models/models.py` always exists), and moved the "copy model to main" step to before stage 4.

---

## 2026-06-02 — `write_model_py.py` generates model.py that fails to load with custom loss classes ✓ fixed

**What happened:** `write_model_py.py` generated `keras.saving.load_model(path)` without `compile=False`. When a model is trained with a custom loss class (e.g. `FocalLoss`), Keras tries to reconstruct the loss at load time and raises `TypeError: Could not locate class 'FocalLoss'`.

**Impact:** Stage 4 (eval/inference) fails for any model trained with a custom loss. Required manual patch to the generated `model.py`.

**Fix:** Added `compile=False` to `write_model_py.py` in main. Inference never needs the loss object, so this is safe universally.

---
