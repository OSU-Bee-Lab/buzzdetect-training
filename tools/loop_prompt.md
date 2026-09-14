Follow LOOP.md for {N} experiments. You are batch {BATCH} of `tools/agent_loop.sh`, which starts a fresh session after each batch. What differs from an attended run:

- LOOP.md step 6 says to stop after one experiment. Here, start the next one, until {N} are logged and pushed.
- Nobody is watching live, though Luke may message you over Remote Control. Don't end your turn waiting for an answer; make the call and record it in notes.md.
- A HANDOFF.md left by an earlier session is work to resume first; it counts toward {N}.
- If Luke asks you to stop the loop, `touch .local/agent_loop/stop`: the driver exits after this batch. To stop now, also write `.local/agent_loop/done` and end your turn; any detached job keeps running.
- When all {N} are done, write `.local/agent_loop/done` (one line: the slugs), then end your turn. The driver stops your session.
- If something blocks you that you can't resolve inside the experiment (tooling you can't repair, a crash you can't diagnose, a data or policy call that is Luke's), record it in that experiment's notes.md. Then write `.local/agent_loop/issue`: what broke, where it's recorded, what you tried, and the experiment's state. Commit and push what you have, then end your turn. The next session will be told to fix it.
