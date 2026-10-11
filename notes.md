# conv-stft
## Hypothesis
Distillation arm. `fused-blocks` (2026-10-11) found a speed ceiling on fast32h16: fifteen trunks from 0.8 to
16 MMACs, separable or fused, all time 2.46-2.84x YAMNet at 200 s (random weights, `exp/fused-blocks`
bench2.log), so past ~2.7x the trunk is not what the time is spent on. What every one of those graphs
shares is the exported front end: `tf.signal.stft` becomes an index Gather over the whole waveform (one
int64 index per sample of every frame), a window Mul, two Transposes and a MatMul DFT, then a second
Gather to cut patches.

The same spectrogram can be computed by one strided 1-D convolution whose kernels are the Hann window
times the DFT's cos / sin rows (only the bins the mel matrix reads), and the patch cut is a Reshape when
patches do not overlap. Same numbers up to float rounding, so **no retraining**: an existing student's
weights go into a graph with the conv front end.

Prediction: re-exported frontier students keep their detections (parity as in the int8 work: no ins_buzz
detection flips at logit > 0 on the fixture, headline equal to the third decimal) and time faster at
200 s on the GPU, most at the fast end where the front end is the largest share. A frontier-wide shift
right at zero headline cost. Falsified if the front end alone does not time faster with the conv form
(then the ceiling is per-call overhead, not the front end's ops), or if parity fails.

Matched control: each student's own existing export (same weights), from `05_distill/log.jsonl`.
