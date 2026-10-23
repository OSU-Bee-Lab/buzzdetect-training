# HANDOFF: noquiet-d8 (batch 42)

Runs as the second CV of `chain_b42.sh` (pid 2307151, log `.local/worktrees/pitch-aux-d8/chain_b42.log`). Batch status, adoption and relaunch: see `.local/worktrees/pitch-aux-d8/HANDOFF.md`.

**If it's still running, report progress and stop.**

When `models/noquiet-d8/folds_sx.csv` exists in this worktree:
1. Check buzz logit SD per fold in chain_b42.log.
2. `$(bash tools/python_path.sh) tools/results.py ps-depth8-r3 .local/worktrees/noquiet-d8/models/noquiet-d8` (control ps-depth8-r3, 0.497). Read the quiet tier beside the headline; untagged/loud are the prediction.
3. Note in Changes: dropping quiet samples also raises ins_buzz's inverse-frequency weight slightly (fewer buzz samples; `build_weights` not pinned). 76 of 1191 training samples dropped in the 1_150 rotation.
4. LOOP.md step 5: delete IDEAS.md #39 in main, `git rm HANDOFF.md`, `tools/finish_experiment.sh noquiet-d8 --model .local/worktrees/noquiet-d8/models/noquiet-d8 --baseline-model ps-depth8-r3 ...`.

If it died: relaunch chain_b42.sh as in pitch-aux-d8/HANDOFF.md (stage 3 resumes finished folds).
