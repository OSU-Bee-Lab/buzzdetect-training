import os

import numpy as np

from embedders.embedding import BaseEmbedder

"""YAMNet, plus a downward-pitch-shifted copy of the same frame, concatenated.

IDEAS.md item 5's sibling: `yamnet_pitchshift` (x2 rung) shifts each frame UP
an octave before a second YAMNet pass, on the hypothesis that honey-bee
harmonics (~230/460/690 Hz) sit at the bottom edge of YAMNet's mel filterbank,
where it is coarse and AudioSet's training mass is thin -- +0.069 headline,
8/8 folds, the era's largest clean gain. This is the mirror direction: shift
DOWN instead. Whether that helps or hurts isn't assumed either way going in --
the recordings carry real content on both sides of the arrangement (higher-
frequency environmental noise some of the time, clearer low-frequency content
other times), so this is a live candidate, not a control expected to fail.

**This is not just `yamnet_pitchshift`'s resample ratio in reverse, and that
distinction matters.** Downsampling a 16 kHz-native frame further only removes
information; there is nothing above 8 kHz in it to shift down. So this
embedder declares `samplerate = 32000` instead of 16000 -- `02_set/extract.py`
resamples directly from each recording's native rate (44.1 kHz on this set) to
whatever the embedder declares, once, so this reaches real content up to 16 kHz
that the plain `yamnet` and `yamnet_pitchshift` pipelines never see (they are
capped at 8 kHz by their own 16 kHz declaration). The down-shift is then free:
take the first HALF of the 32 kHz-rate frame's samples (15360 of 30720) and
feed them to YAMNet AS-IS -- no resample call. YAMNet reads sample values, not
a sample-rate tag, so those 15360 samples (which span the first 0.48 s of the
frame at their true 32 kHz rate) get reinterpreted as a full 0.96 s of 16 kHz
content: playback stretches 2x and every frequency in them halves, including
the 8-16 kHz band that a native-16kHz pipeline would have already discarded.

The unshifted YAMNet block still needs a true 16 kHz representation of the
FULL 0.96 s window, so it downsamples the 32 kHz frame with one librosa call.
That is one extra resample step relative to `yamnet_pitchshift`'s unshifted
block (which reads 16 kHz directly from native audio) -- expect it to be
byte-close but not byte-identical; immaterial next to the ~0.027 headline MDE,
but worth knowing if a future diff looks smaller than float noise should allow.
"""


class EmbedderYamnetPitchshiftDown(BaseEmbedder):
    embeddername = "yamnet_pitchshift_down"
    framelength_s = 0.96  # seconds -- YAMNet's native grid, untouched
    digits_time = 2
    samplerate = 32000  # Hz -- see module docstring: this is what buys real
                        # content above 8 kHz from the native (44.1 kHz) audio
    n_embeddings = 2048  # 1024 unshifted + 1024 pitch-shifted-down
    dtype_in = 'float32'

    def initialize(self):
        import tensorflow as tf
        from embedders.yamnet.yamnet import WaveformFeatures  # registers custom layer
        _ = WaveformFeatures.dtype

        curdir = os.path.dirname(os.path.realpath(__file__))
        yamnet_keras = os.path.join(curdir, '..', 'yamnet', 'yamnet.keras')
        model = tf.keras.models.load_model(yamnet_keras, compile=False)
        # YAMNet's own patch grid is defined in terms of its expected 16 kHz
        # input; framehop_s is still 0.96 s of real time, so this is unaffected
        # by our 32 kHz capture rate.
        model.layers[1].params.patch_hop_seconds = self.framehop_s
        self.model = model

        import librosa
        self._librosa = librosa
        self._frame_samples_native = int(round(self.framelength_s * self.samplerate))  # 30720 @ 32kHz
        self._yamnet_sr = 16000
        self._frame_samples_yamnet = int(round(self.framelength_s * self._yamnet_sr))  # 15360

        # Force TF thread pool init before librosa/Accelerate claims threads --
        # see embedders/yamnet/embedder.py for why.
        self.embed(np.zeros(self._frame_samples_native, dtype=np.float32))

    def embed(self, audio):
        """
        Generate embeddings for audio data

        Args:
            audio: numpy array of audio samples at self.samplerate (32 kHz), a
                whole number of self.framelength_s frames (extraction always
                calls with framehop_prop=1, i.e. non-overlapping frames -- see
                02_set/extract.py)

        Returns:
            numpy array of embeddings, (n_frames, 2048)
        """
        audio = np.asarray(audio, dtype=np.float32)
        n = len(audio) // self._frame_samples_native
        if n == 0:
            return np.empty((0, self.n_embeddings), dtype=np.float32)
        usable = audio[:n * self._frame_samples_native]
        frames_native = usable.reshape(n, self._frame_samples_native)

        # Unshifted block: true 16 kHz representation of the FULL 0.96 s frame.
        unshifted_16k = self._librosa.resample(
            usable, orig_sr=self.samplerate, target_sr=self._yamnet_sr,
        ).astype(np.float32)
        unshifted = np.asarray(self.model(unshifted_16k), dtype=np.float32)
        if len(unshifted) != n:
            raise ValueError(
                f'yamnet_pitchshift_down: YAMNet returned {len(unshifted)} '
                f'frames for {n} input frames (unshifted block)'
            )

        # Down-shifted block: first half of each native 32 kHz frame, fed to
        # YAMNet as-is (no resample) -- see module docstring for the mechanism.
        down_frames = frames_native[:, :self._frame_samples_yamnet]
        down = np.asarray(self.model(down_frames.reshape(-1)), dtype=np.float32)
        if len(down) != n:
            raise ValueError(
                f'yamnet_pitchshift_down: YAMNet returned {len(down)} frames '
                f'for {n} input frames (down-shifted block)'
            )

        return np.concatenate([unshifted, down], axis=1)
