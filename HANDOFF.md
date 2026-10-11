# HANDOFF: conv-stft (batch 30, experiment 3 of 4)

Batch 30 status at park (2026-10-11 05:05): 1 of 4 logged (`fused-blocks`). `ps-d8-aves` is training
(its own handoff). This experiment has **no job running**: it needs the GPU, which the AVES CV holds until
~06:47. Run it after `ps-d8-aves` is logged (one GPU job at a time).

## State
notes.md has the hypothesis. Code is committed here: `frontends.make_features_layer(fe, impl='conv')`
(STFT as one strided Conv1D, patch cut as a Reshape), `export_student.py --fe-impl conv`,
`bench_arch.py` names with `+conv` (`fe@fast32h16+conv`, `a0.25@fast32h16+conv`, `a0.50_d8@yamnet+conv`),
`ladder_record.py record --run`, `reexport.py`. exp/fused-blocks' fused-block arch is patched in.
Checked on CPU already: conv vs STFT log-mel within 1e-5 on every front end and 3e-5 against YAMNet's own
feature layer, same patch counts at every length tried; `fe_C_fast32h16_a0.25_s1_c-buzz-rain-human_wsd56000`
re-exported as `..._cfe` (already in main's models dir): 0 of 313 fixture frames flip, max |logit diff|
2.6e-6, graph 130 -> 90 nodes, no Gather. **No GPU timing yet**: that is the experiment.

## To run (from this worktree)
1. Speed of the front end alone and of whole graphs, random weights (~3 min):
   `OUT=<scratch>; C="fe@fast32h16 fe@fast32h16+conv fe@fast32 fe@fast32+conv fe@yamnet+conv frontend_only a0.25@fast32h16 a0.25@fast32h16+conv a0.50_d8@fast32 a0.50_d8@fast32+conv a0.50_d8 a0.50_d8@yamnet+conv"`
   `tools/launch_job.sh bench.log -- bash -c "python 05_distill/bench_arch.py export --out $OUT --candidates $C && /home/luke/projects/buzzdetect/engine/.venv/bin/python3 05_distill/bench_arch.py time --seconds 200 --out $OUT --candidates $C"`
   If conv is not faster on the front end alone, the hypothesis is falsified: write it up with the bench
   table and log it with the one `_cfe` row (step 2 on that single run), no need for the rest.
2. If it is faster, re-export, score, time and record the frontier students (about 5 min each):
   `tools/launch_job.sh reexport.log -- 05_distill/reexport.py --fe-impl conv --suffix _cfe --runs <R...>` with
   R = the 56000 branches of `fe_C_fast32h16_a0.25_s1`, `fe_C_fast32h16_a0.25_d8_f2_s1`,
   `fe_C_fast32h16_a0.50_d8_s1`, `fe_C_fast32h16_a0.50_s1`, `fe_C_fast32_a0.50_d8_s1`, `fe_C_fast32_a0.50_d10_s1`,
   `fe_C_twofast32_a0.50_s1`, `fe_C_yamnet_a0.25_d8_s1_select`, `fe_C_yamnet_a0.50_d8_s1_select` (each
   `..._c-buzz-rain-human_wsd56000`; take exact names from `ladder_record.py frontier`).
3. Read: each `_cfe` row's headline must equal its source's to the third decimal (same weights); the result
   is the speed column. `tools/results.py <source> <source>_cfe` for one pair as the parity table.
4. notes.md, then `tools/finish_experiment.sh conv-stft --arm distill --summary "..." --runs "<the _cfe names>"`
   (delete this file first).

## Fourth experiment (suggestion, not started)
If conv-stft moves the ceiling, the trunk matters again at the fast end: the natural fourth is the repeat
rule's `fast32h16:a0.25` seed 2 (`ladder_record.py repeats`), or a wider trunk under the new ceiling. If
it does not, run the repeat-rule command.
