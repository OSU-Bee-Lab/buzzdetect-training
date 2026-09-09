import os

import numpy as np

from embedders.embedding import BaseEmbedder

_CPU_SUBDIR = 'perch_v2_cpu'
_GPU_SUBDIR = 'perch_v2_gpu'

# Perch's own preprocessing, from perch_hoplite's
# zoo_interface.EmbeddingModel.normalize_audio with the released target_peak.
_TARGET_PEAK = 0.25


class EmbedderPerch(BaseEmbedder):
    """Perch v2 (Google bird-vocalization taxonomy model), embeddings only.

    The SavedModel is called directly rather than through
    `perch_hoplite.zoo.taxonomy_model_tf.TaxonomyModelTF`, for two reasons the
    v2 release makes unavoidable:

    - `TaxonomyModelTF.batch_embed` calls `model.infer_tf`, which the v2
      SavedModel does not export — it exposes only `serving_default`
      (inputs (batch, 160000) -> embedding (batch, 1536), plus a classifier
      head and spectrogram we don't use).
    - `from_config` parses the SavedModel's class-list CSVs, and the released
      `perch_v2_ebird_classes.csv` has a genuine duplicate ('no_ebird_code'),
      which perch_hoplite's `ClassList` rejects outright. Those lists only name
      the classifier head's logits.

    All that wrapper contributed on top of the raw model was framing and peak
    normalization. Framing is already done upstream — `extract.py` cuts frames
    on its own grid and hands `embed()` a flat buffer of whole frames (which is
    why the embedder is loaded with framehop_prop=1) — so only the
    normalization is reproduced here, verbatim.
    """

    embeddername = "perch"
    framelength_s = 5.0
    digits_time = 1
    samplerate = 32000
    n_embeddings = 1536
    dtype_in = 'float32'

    def initialize(self):
        # Deferred import: TF must not be imported in the parent process before fork
        import tensorflow as tf

        curdir = os.path.dirname(os.path.realpath(__file__))
        has_gpu = bool(tf.test.gpu_device_name())
        subdir = _GPU_SUBDIR if has_gpu else _CPU_SUBDIR
        model_path = os.path.join(curdir, subdir)

        self.framelength_samples = int(self.framelength_s * self.samplerate)
        loaded = tf.saved_model.load(model_path)
        self.model = loaded.signatures['serving_default']
        # Keep a reference: the signature alone does not own the loaded object,
        # and letting it be collected invalidates the captured resources.
        self._loaded = loaded

        # Force TF thread pool init before librosa/Accelerate claims threads; without
        # this the first embed() call after librosa.resample deadlocks on macOS.
        self.embed(np.zeros(self.framelength_samples, dtype=np.float32))

    def embed(self, audio):
        """Embed a flat buffer of whole, non-overlapping frames.

        Args:
            audio: 1-D float32 samples at self.samplerate, a whole number of
                framelength_s frames long (extract.py's chunking guarantees
                this; a ragged tail is dropped rather than zero-padded, since a
                padded frame is not audio the model should score).

        Returns:
            (n_frames, 1536) float32 embeddings, row i for frame i.
        """
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        n_frames = len(audio) // self.framelength_samples
        if n_frames == 0:
            return np.zeros((0, self.n_embeddings), dtype=np.float32)

        framed = audio[:n_frames * self.framelength_samples].reshape(
            n_frames, self.framelength_samples
        )

        # Perch's normalize_audio: DC-remove, then scale each frame's peak to
        # _TARGET_PEAK. `where` leaves an all-zero frame at zero rather than
        # dividing by it.
        framed = framed - framed.mean(axis=-1, keepdims=True)
        peak = np.max(np.abs(framed), axis=-1, keepdims=True)
        framed = np.divide(framed, peak, where=(peak > 0.0)) * _TARGET_PEAK

        return self.model(inputs=framed)['embedding'].numpy()
