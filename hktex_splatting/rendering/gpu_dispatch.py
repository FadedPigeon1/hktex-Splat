"""Own fixed device triangle/tile geometry and benchmark control task records.

DeviceDispatchGeometry feeds HKS scheduling; DeviceDispatch feeds its correctness control.
Main API: DeviceDispatchGeometry, DeviceDispatch."""
from dataclasses import dataclass
import numpy as np
import torch


class DeviceDispatchGeometry:
    """Only fixed camera layout; rebuild alongside PreparedGeometry."""
    def __init__(self, geometry):
        if geometry.dispatch_backend != 'stable' or not geometry.buffers_valid():
            raise ValueError('GPU dispatch requires valid stable camera geometry')
        self.geometry = geometry
        self._policy = (geometry.face_count, geometry.tile_size, geometry.requested_device,
                        geometry.dispatch_backend, id(geometry._layout))
        device = geometry.bary.device
        def tensor(a): return torch.tensor(np.array(a, copy=True), device=device)
        self.face_counts = tensor(geometry._face_counts)
        self.face_starts = tensor(geometry._face_starts)
        self.lengths = tensor(geometry._lengths)
        self.foreground = tensor(geometry._foreground)
        self.foreground_faces = tensor(geometry._foreground_faces.astype(np.int64))
        self.faces = torch.arange(geometry.face_count, device=device, dtype=torch.int64)
        self.pixel_counts = np.bincount(geometry._foreground_faces, minlength=geometry.face_count)
        self.pixel_counts.flags.writeable = False
        self._buffers = (self.face_counts, self.face_starts, self.lengths,
                         self.foreground, self.foreground_faces, self.faces)
        self._versions = tuple(t._version for t in self._buffers)

    def valid(self):
        buffers = (self.face_counts, self.face_starts, self.lengths,
                   self.foreground, self.foreground_faces, self.faces)
        g = self.geometry
        return (self._policy == (g.face_count, g.tile_size, g.requested_device,
                                g.dispatch_backend, id(g._layout)) and g.buffers_valid() and
                all(a is b for a, b in zip(buffers, self._buffers)) and
                tuple(t._version for t in buffers) == self._versions)

    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._buffers)


@dataclass
class DeviceDispatch:
    entries: torch.Tensor
    patches: torch.Tensor
    record_offsets: torch.Tensor
    geometry: object
    fragments: int

    @property
    def shape(self): return self.geometry._layout.shape

    @property
    def tile_size(self): return self.geometry.tile_size

    def record_origin(self, start):
        if start != 0: raise ValueError('Device dispatch currently supports one batch')
        return 0

    def batches(self, max_records):
        if max_records < self.tile_size**2 or self.fragments > max_records:
            raise ValueError('Device dispatch requires one bounded record batch')
        if len(self.entries): yield 0, len(self.entries)

    def records(self, start, end):
        """Debug/reference adapter only; training never downloads these arrays."""
        if start == end:
            return np.empty(0, np.int32), np.empty(0, np.int32)
        from hktex_splatting.rendering.triangle_tile_layout import Dispatch
        g = self.geometry._layout
        plan = Dispatch(g.pixels, g.patch_offsets, g.patch_faces, g.patch_tiles,
                        self.entries.cpu().numpy(), self.patches.cpu().numpy(),
                        self.record_offsets.cpu().numpy(), g.shape, g.tile_size)
        return plan.records(start, end)
