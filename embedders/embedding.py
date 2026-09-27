import importlib.util
import os
from abc import ABC, abstractmethod
from pathlib import Path
import config as cfg


class BaseEmbedder(ABC):
    """Abstract base class for all audio embedders"""

    # Class attributes that each embedder should define
    embeddername: str = None
    samplerate: int = None
    framelength_s: float = None  # in seconds
    n_embeddings: int = None
    digits_time: int = None  # how many digits should timestamps be rounded to? Should be equal to framelength digits
    dtype_in: str = None

    def __init__(self, framehop_prop):
        """Initialize embedder with framehop defined as a proportion

        Args:
            framehop_prop: Duration in seconds between successive frames
        """
        self.framehop_prop = framehop_prop
        self.framehop_s = self.framelength_s * framehop_prop
        self.model = None

    def audio_cache_key(self) -> str:
        return (
            f"sr{self.samplerate}"
            f"_fl{self.framelength_s:g}"
        )

    @abstractmethod
    def initialize(self):
        """Load embedding model; set load_model=False if you just need attributes"""
        pass

    @abstractmethod
    def embed(self, samples):
        """Generate embeddings for audio data"""
        pass

    def to_onnx(self, opset=17):
        """Export this embedder's trunk to ONNX: waveform in, embeddings out.

        Used by 04_deploy/export_onnx.py to fuse the trunk with a trained
        classifier head into buzzdetect's shipped graph. Contract for the
        returned onnx.ModelProto: exactly one graph input, a 1-D float32
        tensor of arbitrary length (raw samples at self.samplerate), and
        exactly one graph output, (n_frames, self.n_embeddings) -- using
        whatever framing/padding rule this embedder's own embed() uses.
        export_onnx.py probes that rule empirically (04_deploy/export_onnx.py
        probe_framing()) rather than assuming one, so any rule is fine as
        long as embed() and this graph agree; that agreement is what
        export_onnx.py's parity check (verify()) is checking.

        Two ways to satisfy it:

        - A recipe (embedders/recipe.py): an embedder assembled from framing,
          per-frame ops and models declares them as data, and RecipeEmbedder
          derives both embed() and this method from that one description.
          This is the way for anything beyond a single model.
        - This default: embed() is nothing but `self.model(audio)` on one Keras
          model, exported with Keras's own ONNX exporter. The class that
          defines embed() vouches for that with `embed_is_model = True` in its
          own body; a subclass that overrides embed() does not inherit the
          vouch, so it cannot silently export its parent's graph.
        """
        owner = next(c for c in type(self).__mro__ if 'embed' in c.__dict__)
        if not owner.__dict__.get('embed_is_model', False):
            raise NotImplementedError(
                f'{self.embeddername}: embed() (defined in {owner.__name__}) does '
                f'more than call self.model, and there is no recipe to export it '
                f'from. Express it as a recipe (embedders/recipe.py) to export it.')

        import keras

        if not isinstance(self.model, keras.Model):
            raise NotImplementedError(
                f'{type(self).__name__} has no Keras trunk (self.model is '
                f'{type(self.model).__name__}) and has not overridden '
                f'to_onnx(); ONNX export needs one or the other.')

        import tempfile

        import onnx

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'trunk.onnx')
            # self.model must already have been called once (initialize()
            # does this for every embedder that owns a Keras trunk) -- Keras
            # refuses to export a model it has never seen called.
            self.model.export(path, format='onnx', verbose=False,
                              opset_version=opset)
            return onnx.load(path)


def load_embedder(embeddername: str, framehop_prop: float, initialize: bool):
    """
    Generic function to load any embedder by name.

    Each embedder directory should contain:
    - embedder.py: Implementation of BaseEmbedder with class attributes
    """
    embedder_path = Path(cfg.DIR_EMBEDDERS) / embeddername

    if not embedder_path.exists():
        raise ValueError(f"Embedder '{embeddername}' not found in {cfg.DIR_EMBEDDERS}")

    # Import the embedder module
    spec = importlib.util.spec_from_file_location(
        f"{embeddername}_embedder",
        embedder_path / "embedder.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the embedder class (should inherit from BaseEmbedder). Only classes
    # defined in this file count: an embedder that imports another embedder's
    # class to subclass it would otherwise load whichever sorts first in dir().
    embedder_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (isinstance(item, type) and
                issubclass(item, BaseEmbedder) and
                item is not BaseEmbedder and
                item.__module__ == module.__name__):
            embedder_class = item
            break

    if embedder_class is None:
        raise ValueError(f"No BaseEmbedder subclass found in {embeddername}/embedder.py")

    # Instantiate and load
    embedder = embedder_class(framehop_prop=framehop_prop)

    if initialize:
        embedder.initialize()

    return embedder
