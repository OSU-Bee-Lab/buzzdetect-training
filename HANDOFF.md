# HANDOFF: bn-site-d8 and bn-shrunk-d8 (batch 42)

Job: pid 2314980, log `.local/worktrees/bn-site-d8/chain_b42b.log` (`chain_b42b.sh`; first waits for chain_b42, pid 2307151). Steps: smoke (1-epoch bntrain on 1_150 + soup.py --bn-site), bntrain-d8 CV (TRUNK_BN_TRAIN=1, TRUNK_SAVE_WEIGHTS=1), then `tools_exp/soup.py` rescoring into `models/bn-site-d8` (site stats) and `models/bn-shrunk-d8` (shrunk, N0 6000).

**If it's still running, report progress and stop.**

When `models/bn-shrunk-d8/folds_sx.csv` exists:
1. Buzz logit SD: bntrain-d8 per fold in chain_b42b.log; the soup lines print it per fold too.
2. Comparisons (all in this worktree's models/):
   - `tools/results.py .local/worktrees/bn-site-d8/models/bntrain-d8 .local/worktrees/bn-site-d8/models/bn-site-d8` (b, exactly matched weights)
   - `tools/results.py .local/worktrees/bn-site-d8/models/bntrain-d8 .local/worktrees/bn-site-d8/models/bn-shrunk-d8` (c)
   - side: `tools/results.py ps-depth8-r3 .local/worktrees/bn-site-d8/models/bntrain-d8` (trainable BN at depth 8)
3. Record: bn-site-d8 logged from this worktree (model bn-site-d8, baseline bntrain-d8). bn-shrunk-d8 needs its own worktree for finish_experiment: `bash tools/setup_worktree.sh bn-shrunk-d8`, copy notes (its own Hypothesis/Results), and pass `--model .local/worktrees/bn-site-d8/models/bn-shrunk-d8 --baseline-model .local/worktrees/bn-site-d8/models/bntrain-d8`. Delete IDEAS.md #37 (b)/(c) text in main (the whole #37 section: (a) is already logged as bn-pooled). Surface results to Luke in the done summary: (b)/(c) are his deployment call.
4. `git rm HANDOFF.md` before finish_experiment.

If it died: read chain_b42b.log, fix, relaunch from this worktree (CV resumes; soup.py skips scored folds; smoke skipped once .smoke_ok exists):
`cd .local/worktrees/bn-site-d8 && /home/luke/projects/buzzdetect-training/tools/launch_job.sh chain_b42b.log -- bash chain_b42b.sh [<pid to wait for>]`
