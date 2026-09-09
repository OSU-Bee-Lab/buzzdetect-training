# 02_set — building a set and extracting embeddings

Three layers: snips → framed-audio cache → embeddings (`extract.py`). `--workers`
parallelises framing+embedding only; `extract_snips` reads source audio off the
slow HDD and is always serial, so a large set's snip sync is a fixed up-front
cost no worker count changes.

`frametimes.csv` (written per ident in layers 2+3) maps frame → source
timestamp; embedding row *i* is frame *i* (`extract_ident_both`).

## Invariants

- **A set's `config_extract.json` always wins** over CLI extraction params —
  embeddings on disk were built under it. Changing them means deleting the file
  and re-extracting (`extract_set`).
- **Annotations are fingerprinted, and the fingerprint is what makes a rerun
  incremental.** Each ident's snip dir holds a `manifest.json` (fingerprint +
  the snips its annotations imply); each framed-audio and embedding dir holds an
  `annotations.fingerprint`. Matching means the product is current and the
  source file is never opened; differing means only that ident is deleted and
  rebuilt. Anything that changes what extraction produces from a given
  annotation set must be reflected in the fingerprint, or stale output is kept
  (`_fingerprint_annotations`, `AssignIdent`).
- **Idents and fold names are both path-like**, of no fixed depth. Never glob a
  fixed number of `*` components to find an ident's directory, and never read
  the top level of `audio/snips/` as a list of idents — walk (`_find_ident_dirs`,
  `_find_snip_dirs`). Getting this wrong deletes data.
- **Every fold is embedded regardless of role**, including `exclude`, so
  flipping a role never costs a re-extraction. Roles are training-time policy
  (`03_train/dataset.py::read_fold_roles`).
- **No augmentation may cross a fold boundary** — `augment.py` is bound by the
  rule in `03_train/CLAUDE.md`. Read it before touching a `CombineSpec`.

## Where sets come from

**`medium` is the only set built from `01_annotate`.** Its `build.R` drives the
four steps next to it — `combine.R` → `annotations.csv`, `folds.R` → `folds.csv`,
`summarize.R` → the `summary_*.csv`, `translate.R` → `translations/` (mapping
rules live in `translate.R`, not the CSVs).

Every other set derives its annotation-level files from the **already-built**
`medium` and never opens `01_annotate`:

- `large` copies medium's files verbatim — it is medium at a finer framehop,
  only `config_extract.json` differs.
- `tiny`/`lite` write a subset of `medium/annotations.csv`, then
  `source('../_derive.R')` regenerates `folds.csv` (medium's rows for the kept
  idents), the summaries, and `translations/` (medium's CSVs trimmed to labels
  the subset still carries — nothing is remapped).

Snips and embeddings are never derived; each set extracts its own under
`--set <name>`. Changing a mapping or fold rule means editing `medium/` and
rebuilding it, then rebuilding the derived sets.

## Translations

Per-set, at `sets/<set>/translations/<name>.csv`. Regeneration is **wholesale,
not additive** — a label the set no longer emits loses its row, even though
embeddings named after it may still be on disk. Such a label then has no row at
all, which `translate_labels` leaves unchanged and `survey_untranslated` reports
at train time. `config.py::path_translation` resolves a name against the set
first, with a project-wide `translations/` fallback. Hand-edits to the CSVs are
overwritten by `build.R`.

The one exception is `medium/translations/general_v1.csv`: a frozen byte-copy of
the pre-2026-09-03 `general` table, written by hand rather than by `translate.R`,
so a rebuild never touches it. It is **historical** since the 2026-09-08
cutover — the log it protected is in `archive/`, and current experiments pass
`--translation general`. Don't hand-edit or regenerate it.

## Do not modify

`build.R` for any set, or the project-wide `translations/`. Changing the data
under an experiment makes it incomparable to everything else in the log.
