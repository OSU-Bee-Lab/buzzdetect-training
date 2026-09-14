Follow LOOP.md for {N} experiments. You are batch {BATCH} of `tools/agent_loop.sh`, which starts a fresh session after each batch. What differs from an attended run:

- LOOP.md step 6 says to stop after one experiment. Here, start the next one, until {N} are logged and pushed.
- Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer; make the call and record it in notes.md.
- Tell the loop where you stand with `{ROOT}/tools/loop_signal.sh` (works from any worktree):
  - all {N} done: `loop_signal.sh done "<slugs>"`, as the last thing before ending your turn. The loop stops your session.
  - blocked by something you can't resolve inside the experiment (tooling you can't repair, a crash you can't diagnose, a data or policy call that is Luke's): record it in that experiment's notes.md, commit and push, then `loop_signal.sh issue` with what broke, where it's recorded, what you tried and the experiment's state, and end your turn. The next session will be told to fix it.
  - Luke asks you to stop the loop: `loop_signal.sh stop`, and it exits after this batch.
- Cleanup is the loop's: once you signal, it stops this session (ending its Monitors) and kills every job you started with `launch_job.sh`. Don't stop jobs or Monitors yourself. If a job is still running when you signal, write its exact relaunch command in the experiment's HANDOFF.md.
- **Temporary rule: report friction as you hit it.** Any clunky or broken tool, a doc that sent you the wrong way, the loop or its tools not behaving as described: `loop_signal.sh friction "<what happened, and where>"`, then work around it and carry on. Don't fix harness problems yourself unless they block you; a fixer agent works through the list after your batch.
