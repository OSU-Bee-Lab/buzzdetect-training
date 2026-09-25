from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoiseSpec:
    prop: float = 0.0
    snr_db: Optional[float] = None  # if set, prop is ignored: noise is scaled
    # per-frame to this SNR (dB) relative to the frame's own RMS, instead of a
    # fixed absolute amplitude. See exp/aug-snr-noise (archive/2026-08_cv-medium-v1):
    # fixed-amplitude noise at the historical prop defaults (0.05-0.2) sits
    # 13-25 dB *above* the median buzz frame's RMS, making every augmented
    # "buzz" frame broadband noise wearing a buzz label.
    labels: Optional[list] = None  # None = apply to all labels

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
            return f'augment_noise_snr{spec.snr_db}'
        return f'augment_noise_{spec.prop}'
    elif isinstance(spec, VolumeSpec):
        return f'augment_volume_{spec.prop}'
    elif isinstance(spec, CombineSpec):
        return f'augment_combine_{spec.class_source}+{spec.class_augment}'
    raise ValueError(f'unknown spec type {type(spec)}')
