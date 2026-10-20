#!/usr/bin/env bash
# batch 39 chain 2: after chain 1 (mixstyle-xfold, pauc-l01; pid 271991) exits, train three
# weight-saving ps-depth8 draws here, then score ps-d8-soup's and bn-pooled's offline combinations.
# Every step resumes on rerun (stage 3 skips finished folds, soup.py skips scored folds).
set -uo pipefail
WT=/home/luke/projects/buzzdetect-training/.local/worktrees
PY=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
while kill -0 271991 2>/dev/null; do sleep 60; done
echo "[chain] chain 1 ended"
cd "$WT/ps-d8-soup" || exit 1
for m in m1 m2 m3; do
  echo "[chain] ps-d8-soup-$m: medium CV"
  env TRUNK_LR_BACKBONE=1e-5 TRUNK_FP16=1 TRUNK_BATCH=1024 TRUNK_SAVE_WEIGHTS=1 \
    "$PY" -u 03_train/main.py --name "ps-d8-soup-$m" --set medium --embedder yamnet_trunk_pitchshift_depth8 \
      --translation general --epochs 30 -y --verbose 2>&1 | tee "train_$m.log" | grep --line-buffered -vE "━|step - "
  rc=${PIPESTATUS[0]}; echo "[chain] ps-d8-soup-$m: exit $rc"; [ $rc -eq 0 ] || exit $rc
done
M=(models/ps-d8-soup-m1 models/ps-d8-soup-m2 models/ps-d8-soup-m3)
"$PY" -u tools_exp/ensemble_predictions.py ps-d8-soup-ens "${M[@]}" || exit 1
"$PY" -u tools_exp/soup.py ps-d8-soup --members "${M[@]}" || exit 1
for a in 25 50 75; do
  "$PY" -u tools_exp/soup.py "ps-d8-soup-w$a" --members "${M[@]}" --wise "0.$a" || exit 1
done
cd "$WT/bn-pooled" || exit 1
S=$WT/ps-d8-soup/models
"$PY" -u tools_exp/soup.py bn-pooled --members "$S/ps-d8-soup-m1" --bn-pool 20000 || exit 1
"$PY" -u tools_exp/soup.py bn-pooled-soup --members "$S/ps-d8-soup-m1" "$S/ps-d8-soup-m2" "$S/ps-d8-soup-m3" --bn-pool 20000 || exit 1
echo "[chain] all done"
