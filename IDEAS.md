# Overview
These are ideas for future loops put in by the human in the loop.

# Embedders
## Samplerate
YAMNet works at 16Khz, but this throws out potentially valuable information. Why not feed it 32KHz audio and just tell it that it's 16KHz? Maybe we can train it to recognize buzzes at half-speed.

### Note
I tried this and it seemed to tank performance; but maybe it was my own bad implementation?

## Mel bands
YAMNet puts mel bands across a wide range, why not target something closer to the bee buzz range?

### Note
I tried this too and it also seemed to tank performance. Any perturbance from YAMNet's expected audio seems to tank performance in a way that can't be recovered by training.


# Extra information
## Audio extraction
We could take the average amplitude in a few targeted bands and add them to the embedding array. Issue is this could greatly slow down processessing, especially if not CUDA-compatible.