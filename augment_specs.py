from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NoiseSpec:
    prop: float
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
        return f'augment_noise_{spec.prop}'
    elif isinstance(spec, VolumeSpec):
        return f'augment_volume_{spec.prop}'
    elif isinstance(spec, CombineSpec):
        return f'augment_combine_{spec.class_source}+{spec.class_augment}'
    raise ValueError(f'unknown spec type {type(spec)}')
