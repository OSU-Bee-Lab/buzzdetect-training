Follow LOOP.md for {N} experiments. You are the experiment agent of batch {BATCH} of `tools/human/agent_loop.sh`, which starts a fresh session after each batch, looping indefinitely.

Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer; make the call and record it in notes.md.

## Controlling the loop
Tell the loop where you stand with `{ROOT}/tools/loop_signal.sh` (works from any worktree). Which signal to send for a problem depends on whether it stops you:

- **All {N} experiments done:** `loop_signal.sh done "<slugs>"`, as the last thing before ending your turn. The loop stops your session once it sees this signal.
- **Friction: something slowed you down or misled you, but you got past it.** A clunky or broken tool, a doc that sent you the wrong way, the loop or its tools not behaving as described. Send `loop_signal.sh friction "<what happened and where, how you worked around it, commits to look at>"` as you hit it, then carry on. The reports pile up, and a fixer works through them after your batch ends. The bar is low: a confusion you cleared up in a few tool calls still counts, and a report needs no root cause or fix, just what happened and where.
- **Issue: something a fixer agent could repair stops the experiment, and you can't fix it from your worktree.** Examples are broken shared tooling or a doc error you can't safely work around.
  1. Record it in the experiment's `notes.md`: what broke, what you tried. Commit and push `exp/<slug>`.
  2. If the experiment is unfinished, commit a `HANDOFF.md` (below).
  3. Send `loop_signal.sh issue` with what broke, where it's recorded and the experiment's state, and end your turn. The loop stops your session and kills your jobs. The next batch opens with a fixer, then an experiment agent that resumes your handoff if the fixer didn't finish the experiment.
- **Halt: blocked by something only Luke can resolve.** For example, a problem in the data that should stop training entirely, or a policy call.
  1. Record it in the experiment's `notes.md`: what broke, what you tried, what Luke needs to decide. Commit and push `exp/<slug>`.
  2. If the experiment is unfinished, commit a `HANDOFF.md`.
  3. Send `loop_signal.sh halt` with a summary, and a PushNotification with its first line: this is the one event Luke gets pushed. The loop stops your session and quits, without running a fixer and without killing your jobs. If a process is hanging, kill it first.
- **Luke asks you to stop the loop:** `loop_signal.sh stop`, and the loop exits after this batch.

Cleanup is the loop's job. Once you signal `done` or `issue`, it stops this session and kills every job you started with `launch_job.sh`. After `halt` it stops the session and leaves the jobs running. Don't stop jobs yourself.

**`HANDOFF.md` is only for a session that ends before its experiment does:** an
`issue` or a `halt`. Watch every job with a Monitor on `tools/watch_job.sh`,
re-armed at every 30-min expiry (CLAUDE.md "Running long jobs"); that keeps a
waiting session alive, so a slow run is never a reason to write one. The next experiment agent resumes every handoff whose experiment isn't in
`log.jsonl` yet. The loop lists those handoffs at the end of this prompt (none
listed means none to resume); any
other `HANDOFF*.md` under `.local/worktrees/` belongs to a closed era (its
experiment is in an `archive/*/log.jsonl`), so ignore it. Commit a handoff in
the worktree with four things:

- the job's pid and log, for `tools/watch_job.sh <pid> <log>` in a Monitor;
- "if it's still running, report progress and stop";
- what to do when it finishes: the comparator, then steps 4-5;
- what to do if it died, including the exact relaunch command. After an
  `issue` the loop has killed the job, so this is the part that counts; stages
  2 and 3 resume on rerun.
