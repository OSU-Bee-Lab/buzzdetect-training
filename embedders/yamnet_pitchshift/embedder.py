import os

import numpy as np

from embedders.recipe import Branch, Keras, Recipe, RecipeEmbedder, UP_OCTAVE

"""YAMNet, plus a pitch-shifted copy of the same frame, concatenated.

IDEAS.md item 5. Honey-bee flight is ~230 Hz with harmonics; YAMNet's mel
filterbank starts at 125 Hz and is near-linear (coarse) below 1 kHz, where
AudioSet's training mass is thin. This shifts each frame up an octave before
a second embedding pass, landing the fundamental and first harmonics at
460/920/1380 Hz where the filterbank is denser and AudioSet has more mass --
then concatenates that block with the unshifted one, so the probe can use
whichever (or both) it finds useful. A straight replacement was tried and
sank (`yamnet-bandpass` -0.047, `yamnet-mask` -0.143, both E1) -- this is a
concat specifically to avoid that failure mode.

The shift is by resampling, not a formant-preserving pitch shifter: each 0.96 s
frame (15360 samples at 16 kHz) is downsampled to 8 kHz (halves the sample
count, low-pass filtering content above 4 kHz of the ORIGINAL signal in the
process) and the result is relabelled as 16 kHz -- so playing it back takes
half the real time, and every frequency in it is doubled. That resampled clip
is then tiled 2x and cropped back to 15360 samples, so YAMNet still sees a
full 0.96 s patch (its patch_hop == patch_window == 0.96 s; feeding it a
shorter buffer changes the frame grid, which is exactly the confound that sank
`perch-probe` -- see framehop-overlap in the archive).

`embedders/yamnet_doublerate/` looks like a shortcut for this but is NOT one:
it changes framelength_s to 0.48s, which changes the frame grid, frame count
and overlap_event_s system-wide. This embedder holds framelength_s at YAMNet's
native 0.96 s and does the resampling entirely inside embed(), so the frame
grid and every label stay byte-identical to plain `yamnet`.
"""

class EmbedderYamnetPitchshift(RecipeEmbedder):
    embeddername = "yamnet_pitchshift"
    framelength_s = 0.96  # seconds -- YAMNet's native grid, untouched
    digits_time = 2
    samplerate = 16000  # Hz
    n_embeddings = 2048  # 1024 unshifted + 1024 pitch-shifted
    dtype_in = 'float32'
    # [plain, octave-up], each through its own call of the whole of YAMNet
    recipe = Recipe(branches=(Branch(Keras()),
                              Branch(Keras(), UP_OCTAVE)))

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        curdir = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(curdir, '..', 'yamnet', 'yamnet.keras')
        model = tf.keras.models.load_model(yamnet_keras, compile=False)
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model

        # Force TF thread pool init before librosa/Accelerate claims threads --
        # see embedders/yamnet/embedder.py for why.
        self.embed(np.zeros(int(round(self.framelength_s * self.samplerate)), dtype=np.float32))
