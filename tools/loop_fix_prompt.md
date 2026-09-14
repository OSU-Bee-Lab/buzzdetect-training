You are batch {BATCH} of `tools/agent_loop.sh`, which runs LOOP.md unattended. Earlier sessions reported problems: read `{ISSUE}`. It holds a blocking issue, a list of friction (clunky or broken tools, misleading docs, the loop harness misbehaving), or both. Your job is to fix them, not to start new experiments.

- Fix each where it lives: in an experiment's worktree, or in main for shared tooling, the loop harness (`tools/agent_loop.sh`, `tools/loop_signal.sh`, the prompts) and docs. Commit and push fixes to main, so later worktrees get them. Record the cause and fix of a blocking issue in the notes.md it points to.
- Keep each fix small and in the style of the code around it. A friction item that isn't worth changing anything for is fine to leave; say so in your `done` summary.
- If a blocking issue interrupted an experiment that can be finished after the fix, finish it through LOOP.md steps 4-5.
- Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer.
- Tell the loop the outcome with `{ROOT}/tools/loop_signal.sh` (works from any worktree), as the last thing before ending your turn:
  - handled: `loop_signal.sh done "<what you fixed, and what you left>"`.
  - anything only Luke can resolve (the data or annotations, a policy call, a repair you couldn't make work): don't guess. `loop_signal.sh issue` saying what he needs to decide. The loop halts there.
