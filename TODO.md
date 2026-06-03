# TODO

- **`setup_worktree.sh`: warn on uncommitted tracked-file changes.** Unstaged or
  staged-but-uncommitted changes to tracked files don't appear in new worktrees —
  currently silent. A check + warning at the end of setup prevents a confusing round of
  "why isn't my code there."
