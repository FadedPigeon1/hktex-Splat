"""Store sparse triangle coefficient arrays for fixed-layout preparation.

FaceMap is shared geometry data, not a standalone kernel-centric renderer.
Main API: FaceMap."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FaceMap:
    """Store ordered per-triangle source IDs, three heat values and six distance coefficients."""
    offsets: np.ndarray
    kernels: np.ndarray
    heat: np.ndarray
    metric: np.ndarray
    upper: np.ndarray

    def validate(self, source_count):
        n = len(self.kernels)
        if (self.offsets.ndim != 1 or self.offsets.dtype.kind not in "iu" or self.offsets[0] != 0 or
            self.offsets[-1] != n or (np.diff(self.offsets) < 0).any() or
            self.heat.shape != (n,3) or self.metric.shape != (n,6) or self.upper.shape != (n,) or
            self.kernels.dtype.kind not in "iu" or (n and (self.kernels.min()<0 or self.kernels.max()>=source_count))):
            raise ValueError("Invalid sparse face map")
        if not all(np.isfinite(a).all() for a in (self.heat,self.metric,self.upper)) or (self.upper<0).any():
            raise ValueError("Nonfinite/negative primitive bounds")
        return self

    @property
    def bytes(self):
        return sum(a.nbytes for a in (self.offsets,self.kernels,self.heat,self.metric,self.upper))
