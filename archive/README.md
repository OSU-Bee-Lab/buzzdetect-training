# Archived experiment logs

One directory per **era** — a stretch during which the logged numbers were
comparable to each other. An era ends when something changes that makes old
numbers stop meaning what they said: the metric, the eval design, the fold
roster, or the training data.

| Era | Runs | Dates | Metric | Ended by |
|---|---|---|---|---|
| [`2026-06_fixed-test`](2026-06_fixed-test/) | 29 | 2026-06-02 → 06-09 | sens@precision, fixed train/validate split, retired corpus | The CV rework |
| [`2026-08_cv-medium-v1`](2026-08_cv-medium-v1/) | 30 | 2026-08-19 → 09-08 | mean per-fold sens@fpr0.005, 11 rotating folds on `medium` | Training-data revision, 2026-09-08 |

Directory names are `<first-entry-date>_<slug>` so they sort in the order the
eras ran. Each holds:

- `log.jsonl` — the entries, verbatim as they were written
- `README.md` — what made these numbers comparable, and what ended that
- `notes/` — the per-experiment working notes
- `set/` — a snapshot of the training set as of the cutover *(where it survives)*

## Why the set snapshot is here

`annotations.csv`, `folds.csv`, `summary_*.csv` and the translation tables are
gitignored generated products. They are also what decides what a number means.
Without them a logged `0.262` is a bare float — you cannot tell which
deployments it averaged over or how much buzz they held.

So they are copied here at cutover, uncompressed (git deltas text well, and a
reader should be able to `cat` them). ~2 MB per era, once.

**The `2026-06_fixed-test` era has no `set/`** — its corpus was retired before
anyone thought to snapshot it, and it is gone. That is the reason this
directory exists in the form it does; see that era's README.

## Adding an era

At cutover, before truncating `log.jsonl`:

1. `mkdir archive/<first-date>_<slug>/{notes,set}`
2. Copy `log.jsonl`, the six set files from `02_set/sets/<set>/`, and the
   translations that runs actually resolved against
3. Copy `notes.md` from every branch the log names — **including branches you
   are about to delete.** A worktree can be pruned freely; a deleted branch
   takes its notes with it
4. Write the README from the template the existing ones follow
5. Tag the pre-truncation commit
6. *Then* truncate

## Recovering deleted branches

Notes for 34 experiments were recovered here from **dangling commits** — their
branches had already been deleted, and only `git fsck --lost-found` still
reached them. Each is now pinned at `refs/archive/<name>`, which keeps its full
tree (code diff included) alive against `git gc`:

```
git for-each-ref refs/archive/          # list them
git show refs/archive/<name>:notes.md   # read one
git diff main refs/archive/<name>       # what the experiment changed
```

These refs are **local only** — `git push` does not carry them by default.
Push them with `git push origin 'refs/archive/*:refs/archive/*'` if this
machine's disk is the single point of failure.
