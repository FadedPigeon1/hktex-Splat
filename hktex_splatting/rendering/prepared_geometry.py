"""Prepare immutable camera/pixel geometry while refreshing learned support.

Shared visibility and tile layout for rendering and training.
Main API: PreparedGeometry."""
import numpy as np
import torch
from hktex_splatting.rendering.triangle_tile_layout import dispatch
from hktex_splatting.rendering.triangle_heat_map import FaceMap


class PreparedGeometry:
    def __init__(self, face_count, face_ids, bary, device='cpu', tile_size=8,
                 dispatch_backend='original'):
        if dispatch_backend not in ('original', 'stable'):
            raise ValueError('Unknown dynamic dispatch backend')
        self._dispatch_backend = dispatch_backend
        ids, bary = np.asarray(face_ids), np.asarray(bary)
        if face_count < 0 or ids.ndim != 2 or bary.shape != (*ids.shape, 3):
            raise ValueError('Invalid camera geometry shape/face count')
        if not np.isfinite(bary[ids >= 0]).all():
            raise ValueError('Nonfinite foreground barycentrics')
        self._ids, self._bary = ids.copy(), bary.copy()
        self.face_count, self.tile_size = face_count, tile_size
        self.requested_device = torch.device(device)
        empty = FaceMap(np.zeros(face_count+1, np.int64), np.empty(0, np.int32),
                        np.empty((0, 3), np.float32), np.empty((0, 6), np.float32),
                        np.empty(0, np.float32))
        self._layout = dispatch(empty, self._ids, tile_size)
        self._lengths = np.diff(self._layout.patch_offsets)
        self._face_counts = np.bincount(self._layout.patch_faces, minlength=face_count)
        self._face_starts = np.r_[0, np.cumsum(self._face_counts)]
        self._foreground = np.flatnonzero(self._ids.ravel() >= 0).astype(np.int32)
        self._foreground_faces = self._ids.ravel()[self._foreground].copy()
        # Own device storage even on CPU: caller writes cannot alias a snapshot.
        def tensor(array):
            return torch.tensor(np.ascontiguousarray(array).copy(), device=device)
        self.bary = tensor(np.nan_to_num(bary).astype(np.float32).reshape(-1, 3))
        self.face_ids = tensor(ids.astype(np.int32).reshape(-1))
        self.patch_offsets = tensor(self._layout.patch_offsets)
        self.pixels = tensor(self._layout.pixels)
        self._buffers = (self.bary, self.face_ids, self.patch_offsets, self.pixels)
        self._versions = tuple(t._version for t in self._buffers)
        for array in self._cpu_arrays():
            array.flags.writeable = False

    def _cpu_arrays(self):
        return (self._ids, self._bary, self._layout.pixels, self._layout.patch_offsets,
                self._layout.patch_faces, self._layout.patch_tiles, self._lengths,
                self._face_counts, self._face_starts, self._foreground, self._foreground_faces)


    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._buffers)

    @property
    def dispatch_backend(self):
        return self._dispatch_backend

    def buffers_valid(self):
        buffers = (self.bary, self.face_ids, self.patch_offsets, self.pixels)
        return (all(a is b for a, b in zip(buffers, self._buffers)) and
                tuple(t._version for t in buffers) == self._versions)
