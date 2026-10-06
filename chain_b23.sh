#!/bin/bash
# batch 23: run experiments 2-4 sequentially after D1 (pid 2589974) exits
source /home/luke/projects/buzzdetect-training/tools/python_path.sh
W=/home/luke/projects/buzzdetect-training/.local/worktrees
C="classes=ins_buzz+ambient_rain+human"
while kill -0 2589974 2>/dev/null; do sleep 30; done
echo "[chain] D1 done; starting d2-fast-56k"
cd $W/d2-fast-56k && env MALLOC_ARENA_MAX=2 "$PY" -u 05_distill/main.py --rung C --runs "fast32:a0.50:$C twofast32:a0.50:$C" --wsd-max 56000; echo "[chain] d2-fast-56k exit $?"
cd $W/fast32-a0375-c && env MALLOC_ARENA_MAX=2 "$PY" -u 05_distill/main.py --rung C --runs "fast32:a0.375:$C" --wsd-max 56000 --wsd-stop 1.3; echo "[chain] fast32-a0375-c exit $?"
cd $W/fast32h16-a0375-c && env MALLOC_ARENA_MAX=2 "$PY" -u 05_distill/main.py --rung C --runs "fast32h16:a0.375:$C" --wsd-max 56000 --wsd-stop 1.3; echo "[chain] fast32h16-a0375-c exit $?"
