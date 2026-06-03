# binary-translation

## Hypothesis

The 11-class `general` translation trains the linear probe to distinguish many non-buzz subclasses
(mech_auto, mech_hum, mech_plane, ambient_*, animal, etc.) that are all equivalent to "not buzz"
at inference time. This wastes model capacity and may blur the buzz/background decision boundary.

Collapsing all non-buzz classes into a single `background` class forces the entire linear probe
to optimize for the buzz/not-buzz boundary directly. Combined with:
- Label smoothing eps=0.2 (confirmed optimal in label-smooth-02)
- Input dropout 0.3 (regularization; absent from all prior experiments)

## Evidence

- label-smooth-02: 0.2993 sensitivity@95% precision — best prior result
- deeper-std: MLP head hurts → representations are good, regularization is the lever
- Binary formulation is standard practice for one-vs-rest classifiers; 11-class softmax may not
  produce calibrated probabilities for a specific positive class

## Changes

- `translations/binary.csv`: new translation — ins_buzz → ins_buzz, everything else → background
- `03_train/train.py`: added Dropout(0.3) before Dense output; label_smoothing=0.2
