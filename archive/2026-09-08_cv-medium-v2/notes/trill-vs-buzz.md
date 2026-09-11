# trill-vs-buzz

Runs the current best structural config (`trunk_ft_1e5`, `exp/trunk-ft`
8aafecb, +0.046 vs a matched frozen control — see `IDEAS.md#lora-adapter`)
merged with the `predictions.csv` provenance columns (`exp/prediction-
provenance` b6949b4). Neither branch alone answers this: trunk-ft has no
`path`/`labels_raw` on its predictions, and prediction-provenance was never
run through a real CV. This worktree merges both (`exp/trunk-ft` then
`exp/prediction-provenance` into `exp/trill-vs-buzz`) so the resulting model
is both a fresh CV confirmation of `trunk_ft_1e5` under the current
(still-growing) annotation set and the first model with provenance columns to
query for `IDEAS.md#trill-vs-buzz` step 1.

Full history for the fine-tune mechanism and the provenance columns
themselves is in `exp/trunk-ft`'s and `exp/prediction-provenance`'s own
`notes.md` (`git show exp/trunk-ft:notes.md`, `git show
exp/prediction-provenance:notes.md`) — not duplicated here.

## Hypothesis (trill-vs-buzz diagnostic, IDEAS.md)

The false positives that set each fold's 0.5% FPR threshold are
disproportionately `ins_trill` — buzz's nearest acoustic neighbour and the
largest non-ambient class in the rotating folds (13,422 frames vs 5,691
buzz). If true, the next lever is a pairwise buzz-vs-trill margin rather than
another representation change.

## Merge notes

- `git merge exp/trunk-ft` into a fresh `exp/trill-vs-buzz` (branched from
  main @ c57468f): two conflicts, both additive CLI-flag/config-dict merges
  in `03_train/main.py` and `03_train/train.py` (trunk-ft's `--batch`,
  `--lr-backbone`, `--lr-head`, `--min-delta`, `--only-folds` alongside
  main's `--no-surprisal`/`--skip-cv`/`--stop-tol`) — resolved by keeping
  both sides' flags and threading all of them through `train_set`/`_train_one`.
- `git merge exp/prediction-provenance` on top: one conflict in
  `_score_fold` (trunk-ft's batched `_predict_buzz` — needed so scoring
  12288-d embeddings doesn't blow VRAM in one forward pass — vs
  prediction-provenance's single-call scoring) — resolved by keeping the
  batched call and prediction-provenance's `path`/`frame_index`/`labels_raw`
  columns together. `03_train/sx.py` (prediction-provenance's
  `read_fold_predictions_events`) merged clean, untouched by trunk-ft.
- Embedder dirs were unsymlinked (`cp -rL`) before merging so git could write
  through them; `yamnet_trunk` (trunk-ft's only new embedder) isn't in main,
  so no symlink/tracked-file collision there.
- Verified: `python -c "import ast; ast.parse(...)"` on both changed files,
  no leftover conflict markers (`grep -n '^<<<<<<<\|^=======\|^>>>>>>>'`).

## Run

Same config as `trunk_ft_1e5`: `yamnet_trunk` embedder, `medium` set,
`general_v1` translation, `--batch 1024 --lr-backbone 1e-5 --lr-head 2e-4
--min-delta 0.002 --patience 50`. New model name `trill_vs_buzz_ft1e5` to
keep it distinct from `trunk_ft_1e5` in `exp/trunk-ft`'s own `models/`.

## Results

_pending — see HANDOFF.md for progress check_

## Conclusion

_pending_
