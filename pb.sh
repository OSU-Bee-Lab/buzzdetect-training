set -e
cd 05_distill
for fe in yam64h20 yam64h24; do
  m=/home/luke/projects/buzzdetect-training/05_distill/data/v4-ft-ps-e60-moderate/models/fe_C_${fe}_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt/model.onnx
  /home/luke/anaconda3/envs/buzzdetect-train/bin/python prefix_bench.py cut $m --out /home/luke/.claude/jobs/18f3b0d6/tmp/prefix_$fe
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 prefix_bench.py time $m --out /home/luke/.claude/jobs/18f3b0d6/tmp/prefix_$fe --seconds 200
done
