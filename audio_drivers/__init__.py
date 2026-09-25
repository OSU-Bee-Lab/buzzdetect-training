"""Open source audio the way buzzdetect does: a vendored driver where one exists
for the extension, soundfile otherwise. See README.md."""
import importlib
import os

import soundfile as sf

_DIR = os.path.dirname(os.path.abspath(__file__))
DRIVER_EXTENSIONS = {os.path.splitext(f)[0].lower() for f in os.listdir(_DIR)
                     if f.endswith('.py') and not f.startswith('_')}


def open_audio(path):
    ext = os.path.splitext(path)[1][1:].lower()
    if ext in DRIVER_EXTENSIONS:
        return importlib.import_module(f'{__name__}.{ext}').Driver(path)
    return sf.SoundFile(path)
