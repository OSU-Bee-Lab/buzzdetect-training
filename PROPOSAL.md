# Proposal: Set-Level Evaluation Tooling

## Problem

The current tools (`summarize_metrics.py`, `compare_metrics.py`) operate on individual model runs. With 5-run sets as the standard protocol, they require manual extraction of values and mental arithmetic to assess whether a result is meaningful. There is no propagated uncertainty estimate, so it is easy to mistake a run-to-run fluctuation for a real signal.

## Proposed Tools

### `evaluate_set.py` — per-set summary with CI

Replaces the ad-hoc `summarize_metrics.py v1 v2 v3 v4 v5` pattern for multi-run experiments.

```bash
conda run -n buzzdetect-train python evaluate_set.py <set_base> [<set_base> ...]
```

**Discovery**: looks for `models/<set_base>_v1` through `_v5` (configurable with `--n`). Falls back to exact names if passed directly.

**Output** (example):

```
Set                  n    median   mean    std    95% CI
------------------------------------------------------------
buzz-upweight        5    0.208    0.206   0.015  [0.188, 0.224]
low-delta            8    0.226    0.225   0.021  [0.208, 0.243]
```

With `--detail`, also prints individual run values.

**CI method**: t-distribution (n-1 df). Appropriate for small n; no assumption about normality beyond what t-test requires.

### `compare_sets.py` — cross-experiment comparison with CI

Replaces the use of `compare_metrics.py` for experiment-level comparisons (individual model rankings in `compare_metrics.py` remain useful for inspecting production models).

```bash
conda run -n buzzdetect-train python compare_sets.py [<set_base> ...] [--top N]
```

Auto-discovers all experiment sets in `models/` by scanning for `_v1` suffix (or accepts explicit names). Ranks by mean sensitivity at 95% precision. Annotates rows where a set's CI overlaps the best set's CI (meaning the difference may not be real).

**Output** (example):

```
Set                  n    mean    95% CI             overlap with best?
----------------------------------------------------------------------
low-delta            8    0.225   [0.208, 0.243]     —
buzz-upweight        5    0.206   [0.188, 0.224]     yes (borderline)
ls02-repro           1    0.160   —                  n/a (single run)
```

## Workflow Changes

**In the LOOP.md `Run the pipeline` step**, after the 5 training+test runs:

```bash
# Report results
conda run -n buzzdetect-train python evaluate_set.py <modelname>
```

**In `notes.md`**, record the CI instead of just the range:

```
- This experiment (<modelname> v1–v5): <val>, <val>, <val>, <val>, <val>
  mean=<mean>  median=<median>  95% CI=[<lo>, <hi>]
```

**In `log.jsonl`**, add `ci_95` alongside the existing fields:

```json
"metrics": {
  "sensitivity_at_95pct_precision": 0.208,
  "n_runs": 5,
  "range": [0.182, 0.222],
  "ci_95": [0.188, 0.224]
}
```

**Interpretation guidance** (to add to LOOP.md):

- CIs don't overlap → strong evidence of a real difference; act on it
- CIs overlap by less than half their width → weak evidence; worth a follow-up
- CIs overlap substantially → no evidence; treat as noise

## What Does Not Change

- `summarize_metrics.py` and `compare_metrics.py` remain untouched (they are still useful for inspecting individual models and production model rankings)
- 5-run protocol stays as-is
- The test pipeline (`04_test/`) is unchanged

## Implementation Notes

- Both scripts share the same `sensitivity_at_95pct` helper already used in `compare_metrics.py` — factor into `metrics_utils.py` to avoid duplication
- CI computation: `scipy.stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)`
- For n=1, report `—` for CI; for n=2, report range only (t-CI on 1 df is uninformative)
- Both scripts should accept an optional `--precision` flag (default 0.95) so the same tools work if the operating point changes
