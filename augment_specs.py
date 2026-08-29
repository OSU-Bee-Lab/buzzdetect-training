from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoiseSpec:
    prop: float = 0.0
    labels: Optional[list] = None  # None = apply to all labels
    # If set, noise is scaled per-frame to this signal-to-noise ratio (dB)
    # relative to that frame's RMS, and `prop` is ignored. Fixed-`prop` noise
    # is an absolute amplitude, which on quiet field recordings (median
    # buzz-frame RMS ~0.006 in the medium set) lands 13-25 dB *above* the
    # signal at the historical default props — i.e. the augmented positive is
    # broadband noise wearing a buzz label. SNR mode keeps the noise a fixed
    # margin below the signal instead.
    snr_db: Optional[float] = None

@dataclass
class VolumeSpec:
    prop: float
    labels: Optional[list] = None

@dataclass
class CombineSpec:
    class_source: str
    class_augment: str
    prop: float  # proportion of source in the mix (0–1)
    limit: int = 4  # max augmented frames per source frame


def spec_dirname(spec):
    if isinstance(spec, NoiseSpec):
        if spec.snr_db is not None:
            return f'augment_noise_snr{spec.snr_db:g}'
        return f'augment_noise_{spec.prop}'
    elif isinstance(spec, VolumeSpec):
        return f'augment_volume_{spec.prop}'
    elif isinstance(spec, CombineSpec):
        return f'augment_combine_{spec.class_source}+{spec.class_augment}'
    raise ValueError(f'unknown spec type {type(spec)}')
