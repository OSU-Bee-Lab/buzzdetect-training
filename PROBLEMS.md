# Loop Friction Log

Problems and confusions encountered during agent runs. Use these to improve LOOP.md and the infrastructure.

---

## 1. Empty embedding directories are misleading

**What happened**: `standard/embeddings/` contains subdirs `yamnet_combined/` and `yamnet_doublerate/` but they are empty (no pickle files). An agent will see these dirs listed by `ls` and assume the embeddings are available, form a hypothesis around one of them, create a worktree, and only discover the problem when training fails to find data.

**Suggested fix**: Either populate these or remove the empty dirs. Alternatively, add a step to LOOP.md's "Think" section: "verify which embeddings are actually populated before proposing a hypothesis (`find 02_set/sets/<setname>/embeddings -name '*.pickle' | wc -l`)".

---

## 2. `models/models.py` is gitignored — missing in worktrees, stage 4 fails

**What happened**: Stage 4 (inference/eval) imports `from models.models import load_model, BaseModel`. In a worktree, `models/models.py` doesn't exist because `models/.gitignore` has a `/*` rule that gitignores it. Running stage 4 from a worktree produces:

```
ModuleNotFoundError: No module named 'models.models'
```

Workaround used: manually symlinked `models/models.py` from the main worktree.

**Suggested fixes** (pick one):
- Add `!models.py` to `models/.gitignore` so git tracks the file and worktrees get it
- Add `ln -s "$ROOT/models/models.py" "$WT/models/models.py"` to `setup_worktree.sh`
- Move `models/models.py` to the project root or another non-gitignored location

---

## 3. Path inconsistency in LOOP.md for log.jsonl

**What happened**: 
- Step 0 says: "read `.local/experiments/log.jsonl`"  
- Step 5B says: "append one JSON line to `experiments/log.jsonl` in the **main** worktree"
- The "Reading prior experiments" section also says "`.local/experiments/log.jsonl`"

These are different paths. An agent writing to one won't find their entries when reading from the other.

**Suggested fix**: Pick one canonical path and use it consistently. `experiments/log.jsonl` (git-tracked in main) seems intentional per step 5B. Update step 0 and the "Reading prior experiments" section to match.

---

## 4. LOOP.md step 7 has a truncated sentence

**What happened**: Step 7 reads: "For example, timeouts because the" — the sentence is cut off. An agent reading this gets no useful guidance on what kinds of friction to log.

**Suggested fix**: Complete the sentence. Something like: "For example, timeouts because the training loop ran too long, missing dependencies, or unclear path conventions."

---

## 5. No initial log.jsonl — cold-start confusion

**What happened**: LOOP.md step 0 says to read the experiment log and see what's been tried. The file didn't exist. The agent has to infer history from model configs and eval.py output instead.

**Suggested fix**: This is inevitable for the first run. Just add a note in LOOP.md: "If the log doesn't exist yet, this is the first experiment — proceed to step 1."

---

## 6. Incomplete prior experiment (deeper-head) creates confusion

**What happened**: The `exp/deeper-head` worktree has a NOTES.md with hypothesis but no results or conclusion. The model `exp_deeper_head_lite` exists and appears in eval.py output, but there's no log entry and no completed notes. An agent reading the notes can't tell if the experiment is in-progress, crashed, or was abandoned.

**Suggested fix**: Standardize NOTES.md to include a `## Status: [in-progress | complete | abandoned]` field. Alternatively, require log.jsonl entries even for abandoned experiments.

---

## 7. Worktree eval.py can't compare against all existing models

**What happened**: Models trained in a worktree go to the worktree's `models/` dir. Running `eval.py` from the worktree only shows models in that dir (plus `model_general_v3`). To compare against `test_standard` or other main-branch models, you have to check `models/test_standard/tests/metrics.csv` manually or run eval.py from the main worktree (where the new model won't be visible).

**Suggested fix**: LOOP.md could note this explicitly. Or `eval.py` could accept a `--models-dir` argument, or scan multiple dirs. Alternatively, the worktree `models/` dir could be a symlink to the main `models/` dir (though this creates write conflicts).

---

## 8. Prior results in eval.py are uncontextualized

**What happened**: Running `eval.py` shows models (`test`, `test_standard`, `exp_deeper_head_lite`) but no context on what sets, architectures, or hypotheses they represent. An agent must read each model's `config_model.json` individually to understand what was tried.

**Suggested fix**: Once `log.jsonl` has entries, eval.py could cross-reference it to print hypothesis/set/embedder alongside metrics. For now, LOOP.md could note: "read each model's `config_model.json` to understand what was tried."

---

## 9. Abandoned yamnet-combined worktree created during this run

**What happened**: I created a `yamnet-combined` worktree expecting those embeddings to exist (see problem 1). After discovering they were empty, I created the correct `deeper-std` worktree instead. The `yamnet-combined` worktree is now abandoned with only a NOTES.md (no experiment ran).

**Impact**: Minor clutter. `git worktree list` shows it. Not harmful, just noisy.
