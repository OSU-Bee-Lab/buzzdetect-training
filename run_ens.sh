#!/bin/bash
# sf-ens-d8: build every pre-registered ensemble (notes.md Design) and score each vs ps-d8-ens3.
set -u
W=/home/luke/projects/buzzdetect-training/.local/worktrees
P=$(bash /home/luke/projects/buzzdetect-training/tools/python_path.sh)
a=$W/ps-depth8/models/ps-depth8; b=$W/ps-depth8-repeat/models/ps-depth8-repeat; c=$W/ps-depth8-redraw/models/ps-depth8-r3
sf=$W/ps2-seamfree-d8/models/ps2-seamfree-d8; r2=$W/ps2sf-d8-r2/models/ps2sf-d8-r2
ens3=$W/ps-d8-ensemble/models/ps-d8-ens3
declare -A E=(
  [sfe-ab]="$a $b" [sfe-ac]="$a $c" [sfe-bc]="$b $c"
  [sfe-sf-ab]="$sf $a $b" [sfe-sf-ac]="$sf $a $c" [sfe-sf-bc]="$sf $b $c"
  [sfe-r2-ab]="$r2 $a $b" [sfe-r2-ac]="$r2 $a $c" [sfe-r2-bc]="$r2 $b $c"
  [sfe-sfr2-a]="$sf $r2 $a" [sfe-sfr2-b]="$sf $r2 $b" [sfe-sfr2-c]="$sf $r2 $c"
  [sfe-sfr2-ab]="$a $b $sf $r2" [sfe-sfr2]="$sf $r2"
)
mkdir -p ens_results
for n in sfe-ab sfe-ac sfe-bc sfe-sf-ab sfe-sf-ac sfe-sf-bc sfe-r2-ab sfe-r2-ac sfe-r2-bc sfe-sfr2-a sfe-sfr2-b sfe-sfr2-c sfe-sfr2-ab sfe-sfr2; do
  echo "== $n: ${E[$n]}"
  $P tools_exp/ensemble_predictions.py $n ${E[$n]} > ens_results/$n.build.txt 2>&1 || { echo "build failed $n"; tail -3 ens_results/$n.build.txt; continue; }
  $P /home/luke/projects/buzzdetect-training/tools/results.py $ens3 models/$n 2>/dev/null | grep -v '^20..-' > ens_results/$n.txt
  grep mean ens_results/$n.txt
done
