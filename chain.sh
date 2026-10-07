#!/bin/bash
# batch 25: three distillation experiments in sequence, each from its own worktree
# (archs live only on those branches). main.py resumes at stage/checkpoint level, so
# rerunning this whole script after a crash skips finished students.
W=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(/home/luke/projects/buzzdetect-training/tools/python_path.sh)
CL=classes=ins_buzz+ambient_rain+human
run() { cd "$W/$1" && echo "[chain25] === $1 ===" && "$PY" 05_distill/main.py --rung C --runs "$2" --wsd-max 56000 --wsd-stop 1.3; }
run fast32-d8d6 "fast32:a0.50_d8::$CL fast32:a0.50_d6::$CL" &&
run fast32h16-depth "fast32h16:a0.50_d8::$CL" &&
run shallow-wide "fast32h16:a0.75_d8::$CL"
