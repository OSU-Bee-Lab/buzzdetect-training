# label-smooth-02

## Hypothesis
Label smoothing ε=0.2 pushes calibration further than ε=0.1, potentially closing the remaining
0.4pp gap to the production target (0.2800 sensitivity @ 95% precision).

The label-smooth experiment (ε=0.1) achieved 0.2763 vs. target 0.2800 — the mechanism clearly
works. ε=0.2 applies more aggressive smoothing (targets 0.9/0.1 instead of 0.95/0.05), which
may further improve calibration of the buzz score distribution.

Risk: too much smoothing could degrade performance if it prevents the model from learning
strong buzz-specific features. But given we haven't reached a peak, ε=0.2 is worth testing.

## Config
- Same as label-smooth experiment, except label_smoothing=0.2
- Embedder: yamnet, standard set, general translation, linear probe
- Validation: all classes
- Loss: BinaryCrossentropy(from_logits=True, label_smoothing=0.2)
