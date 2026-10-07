Follow LOOP.md for {N} experiments. You are the experiment agent of batch {BATCH} of `tools/human/agent_loop.sh`, which starts a fresh session after each batch, looping indefinitely.

Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer; make the call and record it in notes.md.

## Controlling the loop
Tell the loop where you stand with `{ROOT}/tools/loop_signal.sh` (works from any worktree). Which signal to send for a problem depends on whether it stops you:

- **All {N} experiments done:** `loop_signal.sh done "<slugs>"`, as the last thing before ending your turn. The loop stops your session once it sees this signal.
- **Friction: something slowed you down or misled you, but you got past it.** A clunky or broken tool, a doc that sent you the wrong way, the loop or its tools not behaving as described. Send `loop_signal.sh friction` with what happened and where, how you worked around it, and commits to look at, as you hit it; pass the message on stdin through a quoted heredoc (`<<'EOF'`), since inside double quotes bash runs any backticks or `$(...)` it holds, then carry on. The reports pile up, and a fixer works through them after your batch ends. The bar is low: a confusion you cleared up in a few tool calls still counts, and a report needs no root cause or fix, just what happened and where.
- **Issue: something a fixer agent could repair stops the experiment, and you can't fix it from your worktree.** Examples are broken shared tooling or a doc error you can't safely work around.
  1. Record it in the experiment's `notes.md`: what broke, what you tried. Commit and push `exp/<slug>`.
  2. If the experiment is unfinished, commit a `HANDOFF.md` (below).
  3. Send `loop_signal.sh issue` with what broke, where it's recorded and the experiment's state, and end your turn. The loop stops your session and kills your jobs. The next batch opens with a fixer, then an experiment agent that resumes your handoff if the fixer didn't finish the experiment.
- **Halt: blocked by something only Luke can resolve.** For example, a problem in the data that should stop training entirely, or a policy call.
  1. Record it in the experiment's `notes.md`: what broke, what you tried, what Luke needs to decide. Commit and push `exp/<slug>`.
  2. If the experiment is unfinished, commit a `HANDOFF.md`.
  3. Send `loop_signal.sh halt` with a summary, and a PushNotification with its first line: this is the one event Luke gets pushed. The loop stops your session and quits, without running a fixer and without killing your jobs. If a process is hanging, kill it first.
- **Park: a job's remaining time is over 1.5 hours.** Don't sit on a Monitor for it; every 30-min re-arm is a full turn.
  1. Commit a `HANDOFF.md` (below) that also says how many of your {N} experiments are already logged and which remain.
  2. Send `loop_signal.sh park <minutes> "<why>"`, with <minutes> about the ETA, as the last thing before ending your turn. The loop stops your session but **leaves your jobs running**, sleeps until that time or until the jobs all exit (whichever is first), then relaunches an agent on this same batch to resume the handoff. That agent may park again if the job is still going.
  **The rule:** launch the job and arm one 30-min `tools/watch_job.sh` Monitor. When it fires, compute the ETA from measured progress (folds or runs done vs remaining, time per unit so far), including queued experiments that will follow. Over 1.5 hours remaining: park. Otherwise re-arm and continue, and re-check the ETA at every later expiry, since a job can turn out slower than its first fold suggested. You don't need to park at launch from a guess.
- **Luke asks you to stop the loop:** `loop_signal.sh stop`, and the loop exits after this batch.

Cleanup is the loop's job. Once you signal `done` or `issue`, it stops this session and kills every job you started with `launch_job.sh`. After `halt` or `park` it stops the session and leaves the jobs running. Don't stop jobs yourself.

**`HANDOFF.md` is only for a session that ends before its experiment does:** a
`park`, an `issue` or a `halt`. Watch your jobs with one Monitor on `tools/watch_job.sh` (no
arguments: it follows every job you launch, later ones too, so never arm a
second), re-armed at every 30-min expiry (CLAUDE.md "Running long jobs"); that keeps a
waiting session alive, so a slow run is never a reason to write one. Between its events, don't look at the job: no `tail` of the log, no ReadNotifications-then-check loop. Each look is a full turn (CLAUDE.md "Running long jobs"). The next experiment agent resumes every handoff whose experiment isn't logged
yet (`03_train/log.jsonl` or `05_distill/log.jsonl`). The loop lists those handoffs at the end of this prompt (none
listed means none to resume); any
other `HANDOFF*.md` under `.local/worktrees/` belongs to a closed era (its
experiment is in an `archive/*/<stage>/log.jsonl`), so ignore it. Write one handoff
per unfinished experiment, in that experiment's own worktree (a single job chaining
several experiments still gets one per worktree, each saying which part is its own):
`finish_experiment.sh` refuses a slug whose worktree still has a handoff. Commit each
with four things:

- the job's pid and log: the resuming session runs `tools/watch_job.sh --adopt <pid>`,
  then watches it with its one `tools/watch_job.sh` Monitor;
- "if it's still running, report progress and stop";
- what to do when it finishes: the comparator, then steps 4-5;
- what to do if it died, including the exact relaunch command. After an
  `issue` the loop has killed the job, so this is the part that counts; stages
  2 and 3 resume on rerun.
