# d1-yamnet-a050-clean
## Hypothesis
IDEAS D1. The top of the student frontier (the candidate *standard* tier) is the
YAMNet-front-end a0.50 student, whose only rung-C readings (0.723/0.722 at 7k/14k,
1.38x) trained on the contaminated `C` pack. Retrained on the clean rebuilt pack
(rung C, WSD to 56k, `--wsd-stop 1.3`, classes buzz+rain+human, seed 1), it should land
within ~0.02 of 0.722. Matched control: `fe_C_yamnet_a0.25_s1_select_c-buzz-rain-human_wsd*`
(same everything but width; 0.693/0.691/0.704/0.707). Falsifier: clean headline more than
~0.02 below 0.722 says the contamination was worth something.
## Changes
None to code. Run: `05_distill/main.py --rung C --runs "yamnet:a0.50:select:classes=ins_buzz+ambient_rain+human" --wsd-max 56000 --wsd-stop 1.3`.
