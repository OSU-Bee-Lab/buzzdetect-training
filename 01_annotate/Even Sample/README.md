# Even Sample Annotation Set
This annotation set will attempt to supercede the old training set, expand the diversity of bees in our existing training set, and create a new paradigm of annotating even subsets across the day (rather than ad-hoc annotation efforts).

## Motivation
### Nighttime not represented in tests
![alt text](data/raw/images/detections.png)
In June, 2026 I (Luke Hearon) was trying to train new models to improve performance, especially on spikes of nighttime false positives we've observed (e.g., Luke - Diel Drivers/2026-05-06/1_95/260506_1924.mp3 at 10:00 PM, a peak detection rate ~5%). I was unable to improve the sensitivity-precision frontier. I realized that our current test set does not cover nighttime audio, so improvements in this area would not be registered on a test. I had trained a model (yamnet_bandpass_medium) on the medium test set using a modified version of yamnet that added a bandpass filter (100, 3000)Hz. Performance on the aforementioend diel drivers file appeared to be vastly improved.


### Nighttime not represented in training
![alt text](data/raw/images/activations.png)

In plotting activation_ins_buzz of these models across the day, I also found that activations were very high at night, then dropped in the morning. This is apparently not a big concern, as the threshold still rejects most of the high activations at night, but it seems to indicate an issue in the training set.

Namely, activation_ins_buzz is likely dropping because there are other events that the model does recognize during the day; these events draw activation away from ins_buzz. At night, the model doesn't recognize any events and is forced to (allowed to) let ins_buzz activations increase. Note: I don't think there's a regularizing function after the dense layer, so it isn't mathematically that low activations in other neurons causes ins_buzz to increase, I think it's just the model's "uncertain" state. Under this hypothesis, this should be resolved by increasing the model's certainty at night.

Notably, there still is a distinct spike of activation at 10pm. This corresponds with passing planes. This is somewhat confusing, as mech_plane is in the training set for these models, but it may indicate that we have accidentally trained the model only on *daytime* planes and the model has learned some feature of daytime audio rather than the sound of the plane itself.

## Summary
This set will spread annotations across a number of targets (flowers, hives). For each target, we will select two days of audio to annotate. To maximize diversity, no two recordings in the set may be from the same site and date, even if the targets differ. For example, we will not allow these two deployments in the set: milkweed recorded at Rothenbuhler on 2025-06-05; yellow jacket nest recorded at Rothenbuhler on 2025-06-05.

While reproducible, programmatic selections would be nice for reproducibility's sake, managing the data for this is a nightmare and it's not like you can just click a button and re-annotate all the files anyways sooooooo...


## Folds
I would also like to facilitate a transition from a fixed test/train/validate approach to an N-fold approach where each fold is one site-date, enabling cross fold validation. This will require a change to the set extration and training logic.