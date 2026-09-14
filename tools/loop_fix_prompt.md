You are batch {BATCH} of `tools/agent_loop.sh`, which runs LOOP.md unattended. The previous session stopped on an issue: read `{ISSUE}` and the notes.md it points to. Your job is to fix that issue, not to start new experiments.

- Fix it where it lives: in the experiment's worktree, or in main if the fault is in main's own tooling. Record the cause and the fix in the notes.md it points to.
- If the interrupted experiment can be finished after the fix, finish it through LOOP.md steps 4-5.
- Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer.
- Fixed: write `.local/agent_loop/done` (one line: what you fixed), then end your turn.
- Not fixable without Luke (a data or policy call, or a repair you couldn't make work): don't guess. Write `.local/agent_loop/issue` saying what he needs to decide, then end your turn. The driver halts there.
