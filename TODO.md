# TODO

## Scripts / skills

### High

- **Step 0 briefing skill** — synthesizes log.jsonl, eval.py output, and available
  embeddings into a structured "what's been tried / what's available / what looks
  promising" briefing. Runs at the start of every loop iteration. Should also include
  augmented embedding inventory: for each set/embedder, what augment dirs exist and how
  many files are in each. (Agents proposing augment-based experiments currently have to
  inspect the embeddings dir manually to know what's already built and what the exact
  dirname strings are.)

- **`finish_experiment.sh <slug> <modelname>`** — mechanical cleanup after training:
  copies the model from the worktree to the main tree, then runs stage 4 from the main
  worktree. Afterward, reads `models/<modelname>/tests/metrics.csv` and
  `models/<modelname>/config_model.json` to emit a pre-filled log.jsonl template with
  date, branch, set, embedder, and quantitative metrics filled in — leaving `hypothesis`,
  `conclusion`, and `baseline.model` as `"FILL_IN"` placeholders for the agent to
  complete. This is the right division: the quantitative parts are mechanical, the
  qualitative parts are not and can't be generated from artifacts alone.

### Medium

- **Richer metrics view** — shows sensitivity at 80%, 90%, 95% precision for two models
  side-by-side. May just be a small addition to eval.py rather than a separate tool.

### Low

- **`setup_worktree.sh`: warn on uncommitted tracked-file changes.** If there are
  unstaged or staged-but-not-committed changes to tracked files, they won't appear in the
  new worktree. Currently silent. A check at the end of setup (`git status --short |
  grep -v '^??' | grep -v '^$'`) with a printed warning saves a confusing round of
  "why is my code not there." Doesn't need to block, just warn.
