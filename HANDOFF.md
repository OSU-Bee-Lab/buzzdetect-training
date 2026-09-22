# HANDOFF — exp/perch-centred

**Nothing is running.** Every process from this investigation was killed
before handoff (memory safety); there is no pid to watch. Read `notes.md`
first -- it has the full investigation (two real memory bugs found and
fixed, a third suspected but unconfirmed, and why a burst/restart workaround
doesn't work for this pipeline's ident-level atomicity).

## State

- `embedders/perch_centred/embedder.py` (committed, in the shared tree):
  the design is correct and verified to produce real embeddings. It carries
  two real fixes (bucketed batch shape, XLA JIT disabled) that are worth
  keeping regardless of what's found next.
- No extraction has completed even one medium ident. `02_set/sets/*/embeddings/perch_centred`
  does not exist yet -- nothing to clean up, nothing partially written.
- `run_perch_centred.sh` (committed): a burst-and-restart driver. It works
  as a mechanism (verified: killing a burst mid-run and relaunching resumes
  cleanly, memory resets), but is **not** the fix -- `extract_ident_both`
  only writes an ident's output after all its snips finish, so restarting
  inside an ident discards that ident's progress entirely, and bursts short
  enough to stay memory-safe (under ~5 min by the observed growth rate)
  don't reliably finish even one snip. Don't lean on it as-is; it's here in
  case a future fix makes bursts long enough to matter (e.g. if bursting by
  whole idents becomes viable once the leak is smaller).

## What to do next

1. **Isolate `read_range`/`_read_pad_frame`** (in `02_set/extract.py`), the
   same way the model call was isolated (a standalone script, real audio
   file, call them in a tight loop, watch RSS via `resource.getrusage` or
   `ps`). This is the leading untested hypothesis in `notes.md`'s Results
   section -- confirm or rule it out before looking elsewhere.
2. If that's clean, try running each chunk's `self.model(...)` call in a
   short-lived subprocess so OS teardown guarantees memory is reclaimed
   between calls, accepting the SavedModel-reload overhead per chunk.
3. Once extraction is stable (RSS flat or clearly bounded over an hour-plus
   run), the real medium extraction is long regardless (~8-15 h estimated at
   Perch's own established CPU rate x this design's ~5.2x higher frame
   density) -- launch it as an ordinary long job, Monitor-watched with
   re-arms per CLAUDE.md, not a burst driver.

## If picking this up cold

- Two-interpreter split, same as `perch-probe`: extraction runs under
  `.local/venv-perch-extract` (TF 2.21, no GPU), training under the pinned
  conda env (TF 2.16.2) -- see `embedders/perch_centred/embedder.py`'s
  module docstring and `perch-probe`'s own notes for why.
- The box has 23 GB RAM, no swap headroom to spare (8 GB configured, gets
  hit fast once RSS passes ~15-17 GB). Watch `free -h` directly when
  testing memory behaviour, not just `ps` RSS for the one process -- system
  `available` is what matters.
- `_BATCH_BUCKET = 16` and `set_jit(False)` in `embedder.py` are both real
  fixes worth keeping even if a further fix is found; don't revert either
  while debugging the remainder.
