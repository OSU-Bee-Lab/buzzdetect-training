set -e
cd "$(dirname "$(realpath "$0")")/05_distill"
for r in 1 2 3; do
  echo "[retime] round $r"
  /home/luke/projects/buzzdetect/engine/.venv/bin/python3 bench_arch.py time --seconds 200 --repeats 15 --out /home/luke/.claude/jobs/18f3b0d6/tmp/bench --candidates fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h24_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h24_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmtz fe_C_yamnet_a0.25_d8_s1_select_c-buzz-rain-human_wsd56000_mmt fe_C_yamnet_a0.25_d8_s1_select_c-buzz-rain-human_wsd56000_mmtz fe_C_yamnet_a0.50_d8_s1_select_c-buzz-rain-human_wsd56000_mmt fe_C_yamnet_a0.50_d8_s1_select_c-buzz-rain-human_wsd56000_mmtz fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmt fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000_mmtz
  cp /home/luke/.claude/jobs/18f3b0d6/tmp/bench/results.json /home/luke/.claude/jobs/18f3b0d6/tmp/bench/results_$r.json
done
