# Audio drivers

Copied verbatim from buzzdetect (`engine/src/stream/drivers/`, commit 552d1f3) for
formats libsndfile can't open (wma, mp4, mts). Same contract: a `Driver` class that duck-types
`soundfile.SoundFile`. One file per extension; `open_audio()` in
`__init__.py` picks the driver by extension and falls back to soundfile.

Re-copy from buzzdetect rather than editing here, so the two stay identical.
