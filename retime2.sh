set -e
cd "$(dirname "$(realpath "$0")")/05_distill"
for r in 1 2; do
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 bench_arch.py time --seconds 200 --repeats 15 --out /home/luke/.claude/jobs/18f3b0d6/tmp/bench2 --candidates fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_twofast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_twofast32_a0.50_d8_s2_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h24_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yamnet_a0.25_d8_s1_select_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h32_a0.25_d8_l5x0.50_s1_c-buzz-rain-human_wsd56000_mmt fe_C_twofast32_a0.25_d8_l5x0.50_s1_c-buzz-rain-human_wsd56000_mmt
  cp /home/luke/.claude/jobs/18f3b0d6/tmp/bench2/results.json /home/luke/.claude/jobs/18f3b0d6/tmp/bench2/results_$r.json
done
m=/home/luke/projects/buzzdetect-training/05_distill/data/v4-ft-ps-e60-moderate/models/fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz/model.onnx
/home/luke/anaconda3/envs/buzzdetect-train/bin/python prefix_bench.py cut $m --out /home/luke/.claude/jobs/18f3b0d6/tmp/bench2/prefix_h20z
/home/luke/projects/buzzdetect/engine/.venv/bin/python3 prefix_bench.py time $m --out /home/luke/.claude/jobs/18f3b0d6/tmp/bench2/prefix_h20z --seconds 200
