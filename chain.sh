set -e
cd "$(dirname "$(realpath "$0")")/05_distill"
PY=$(bash ../tools/python_path.sh)
$PY reexport.py --fe-impl mmtz --suffix _mmtz --runs fe_C_yam64h20_a0.25_d8_s1_c-buzz-rain-human_wsd56000 fe_C_yam64h24_a0.25_d8_s1_c-buzz-rain-human_wsd56000 fe_C_yam64h16_a0.25_d8_s1_c-buzz-rain-human_wsd56000 fe_C_yam64h20_a0.50_d8_s1_c-buzz-rain-human_wsd56000 fe_C_yam64h16_a0.50_d8_s1_c-buzz-rain-human_wsd56000 fe_C_yamnet_a0.25_d8_s1_select_c-buzz-rain-human_wsd56000 fe_C_yamnet_a0.50_d8_s1_select_c-buzz-rain-human_wsd56000 fe_C_fast32_a0.50_d8_s1_c-buzz-rain-human_wsd56000
