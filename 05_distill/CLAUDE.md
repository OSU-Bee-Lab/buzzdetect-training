# 05_distill: stage-local gotchas

`README.md` is the operator's guide (one command, new teachers, what is cached where, resume semantics).
`DESIGN.md` is the contract, `FRONTENDS.md` the live experiment, `LADDER.md` the closed data-size ladder.

## Invariants

- **A teacher is a parameter, and only `dpaths.py` knows the default.** Class list, dead classes, buzz index,
  code width, eval set and every path come from `dpaths.py` (`D.spec()`, `D.CACHE`, `D.LOCAL`, ...). Never
  write a teacher name, a class index (`8`, `(0, 14)`), `15` or `2048` in a script; add it to the spec instead.
  Anything that needs the teacher reads `DISTILL_TEACHER` (main.py sets it for its subprocesses).
- **Share at the level a product depends on** (README's table): spectrograms in `<distill_cache>/_mel/<spec>`
  (audio + front end only), targets under `<distill_cache>/<teacher>/`, runs under `.local/distill/<teacher>/`.
  A new derived product gets a fingerprint (`store.stamp_or_refuse` for costly ones, a meta fingerprint that
  triggers a rebuild for cheap ones), never a bare "exists means current".
- **Presence means done.** Write npz through `store.write_npz` (temp name + rename); order writes so the last
  one is the done marker (`cache.write_slice` writes the mel, then the targets).
- **Edit a front end by renaming it.** `_mel/<spec>` is stamped with the spec's definition; editing
  `frontends.py` under an existing name stops the next cache_fe/cache run instead of mixing spectrograms
  (bump `store.MEL_VERSION` if `mel_patches`' math changes).
- **`import tensorflow` first** in entry points that import pandas (CLAUDE.md, root). `dpaths.py`, `store.py`
  and `main.py` import neither, so the engine venv and the onnx venv can use them.
- **Never assume 15 outputs or buzz at column 8 in code that reads a student.** A `--classes` student has
  its own class list (`curve.json` `keep_classes`, the exported `config_model.json` `classes`); take the buzz
  column from that list (`eval_folds.infer`, `export_student.parity`, `distill_train.ACT_*` do).
- `frame_length`/`slice` geometry (62 frames, 15360 samples, 953600 with lookahead) is fixed in `dpaths.py`;
  `teacher_onnx.py` refuses a teacher framed differently.

## Gotchas

- **Never edit scripts here while a chain runs from this worktree.** A chain re-reads its python stage scripts
  at every stage and bash reads a running script incrementally. Work on another branch/worktree, merge after.
  (2026-09-29: the generalization was written on `distill-generic` for exactly this reason.)
- **`migrate_layout.py` before the first `main.py` on data made under the old layout** (done for the v4 teacher on
  2026-09-29; a checkout that predates that still needs it): until then
  `.local/distill/<teacher>/` is empty, so main.py sees nothing done and would redo everything. It refuses to
  run while a distillation job is alive. After migrating, `distill_cache` in paths.local.json must be the
  parent `.../distill-cache/`, not the teacher directory.
- **`cache.py` refuses a changed teacher ONNX.** A retrained model re-shipped under the same name has new
  targets; move `<distill_cache>/<teacher>` aside (do not delete a rung-D cache lightly: days of decode) or
  give the new model a new name.
- **A resumed training run is not bit-identical** to an uninterrupted one: weights and optimizer state are
  restored exactly, the batch order is reseeded (`seed + step`).
- `student_init.frames_from_folds` (used by `test_synth.py` and `student_init.py --check`) reads the
  `moderate` set's framed audio: it is a plumbing check, independent of the teacher.
- The 4 GB card fits one GPU job; tests hide it (`CUDA_VISIBLE_DEVICES=''`) and run niced.
