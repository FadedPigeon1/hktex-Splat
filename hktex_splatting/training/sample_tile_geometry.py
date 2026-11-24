"""Build flat GPU sample tiles for HKS training without camera-grid packing.

SampleTileSurface groups original samples by triangle, constructs GPU dispatch
metadata and restores original order through inverse. Existing support, corner
coefficients, splatting and backward operators consume this per-batch snapshot.
Host face IDs are retained only for the audited triangle support selector.
"""
from types import SimpleNamespace
import numpy as np
import torch


class SampleTileSurface:
    """Immutable geometry for one nonempty surface batch; never learned output."""
    @torch.no_grad()
    def __init__(self, faces, embedding, face_ids, bary, *, host_ids=None,
                 max_bytes=512*1024**2):
        count = len(face_ids)
        if (count < 1 or face_ids.ndim != 1 or bary.shape != (count, 3) or
                face_ids.dtype not in (torch.int32, torch.int64) or
                faces.device != embedding.device or faces.requires_grad or embedding.requires_grad or
                bary.requires_grad or face_ids.requires_grad):
            raise ValueError('Expected fixed geometry and nonempty face/barycentric samples')
        if max(count, len(faces)) >= 2**31:
            raise ValueError('Sample geometry requires int32-indexable faces and samples')
        # Conservative bound for geometry/sort temporaries; support has its own budget.
        if count*(512+embedding.shape[1]*embedding.element_size())+len(faces)*32 > max_bytes:
            raise MemoryError('Direct sample geometry exceeds configured byte budget')
        host = (face_ids.detach().cpu().numpy() if host_ids is None else np.asarray(host_ids))
        if host.shape != (count,) or host.dtype.kind not in 'iu' or np.any(host < 0) or np.any(host >= len(faces)):
            raise ValueError('Invalid host face IDs')
        self._ids_cpu = np.sort(host, kind='stable').reshape(count, 1).copy()
        self._ids_cpu.flags.writeable = False
        device = embedding.device
        ids = face_ids.to(device=device, dtype=torch.long)
        bary = bary.to(device=device, dtype=embedding.dtype)
        torch._assert_async(((ids >= 0) & (ids < len(faces))).all(), 'Invalid sample face IDs')
        torch._assert_async((torch.isfinite(bary).all() & (bary >= -2e-6).all() &
                             ((bary.sum(1)-1).abs() <= 2e-6+1e-5).all()), 'Invalid sample barycentrics')
        order = torch.argsort(ids, stable=True)
        self.inverse = torch.empty_like(order)
        self.inverse[order] = torch.arange(count, device=device)
        self.face_ids, self.bary = ids[order], bary[order]
        torch._assert_async((self.face_ids == torch.tensor(self._ids_cpu[:, 0].copy(), device=device)).all(),
                            'Host/device sample IDs differ')
        self.faces, self.embedding, self.camera = faces, embedding, None
        self.shape, self.tile_size = (count, 1), 8  # Output view only, not an image layout.
        self.tile_count = (count+63)//64
        self.active_tile_count = self.tile_count
        self.pixels = torch.arange(count, device=device)
        self.active_ordinals = self.pixels.to(torch.int32)
        self.vertices = faces[self.face_ids].long()
        self.queries = (embedding[self.vertices]*self.bary[:, :, None]).sum(1)
        self.tiles = self.pixels//64
        slots = torch.arange(self.tile_count*64, device=device).reshape(-1, 64)
        self.tile_pixels = torch.where(slots < count, slots, -1).int()
        self.tile_lengths = (count-torch.arange(self.tile_count, device=device)*64).clamp(0, 64).int()
        self.active = torch.ones(self.tile_count, device=device, dtype=torch.bool)
        self.positions = self.depth = None
        self._names = ('faces', 'embedding', 'inverse', 'face_ids', 'bary', 'pixels',
                       'active_ordinals', 'vertices', 'queries', 'tiles', 'tile_pixels',
                       'tile_lengths', 'active')
        self._buffers = tuple(getattr(self, name) for name in self._names)
        self._versions = tuple(t._version for t in self._buffers)
        self._policy = (self.shape, self.tile_size, self.tile_count, self.active_tile_count, id(self._ids_cpu))
        self.dispatch_geometry = SampleDispatchGeometry(self)

    def valid(self):
        return (self._policy == (self.shape, self.tile_size, self.tile_count, self.active_tile_count, id(self._ids_cpu)) and
                not self._ids_cpu.flags.writeable and
                all(getattr(self, name) is t for name, t in zip(self._names, self._buffers)) and
                tuple(t._version for t in self._buffers) == self._versions)


class SampleDispatchGeometry:
    """GPU face/tile incidences implementing the shared HKS dispatch contract."""
    @torch.no_grad()
    def __init__(self, surface):
        self.surface = surface
        count, faces = len(surface.pixels), len(surface.faces)
        if faces*surface.tile_count >= 2**63:
            raise ValueError('Face/tile key exceeds int64')
        keys, lengths = torch.unique_consecutive(surface.face_ids*surface.tile_count+surface.tiles,
                                                 return_counts=True)
        # Samples are face-sorted and tile ordinals increase, so keys are sorted.
        self.patch_tiles = keys % surface.tile_count
        patch_faces = keys//surface.tile_count
        self.face_counts = torch.bincount(patch_faces, minlength=faces)
        self.face_starts = torch.cat((self.face_counts.new_zeros(1), self.face_counts.cumsum(0)))
        self.pixel_counts_device = torch.bincount(surface.face_ids, minlength=faces)
        self.lengths = lengths
        self.geometry = SimpleNamespace(face_count=faces, tile_size=surface.tile_size,
            bary=surface.bary, face_ids=surface.face_ids, _ids=surface._ids_cpu,
            _layout=SimpleNamespace(patch_tiles=self.patch_tiles))
        self._names = ('patch_tiles', 'face_counts', 'face_starts', 'pixel_counts_device', 'lengths')
        self._buffers = tuple(getattr(self, name) for name in self._names)
        self._versions = tuple(t._version for t in self._buffers)

    def valid(self):
        return (self.surface.valid() and self.geometry.bary is self.surface.bary and
                self.geometry.face_count == len(self.surface.faces) and
                self.geometry.tile_size == self.surface.tile_size and
                self.geometry._layout.patch_tiles is self.patch_tiles and
                self.geometry.face_ids is self.surface.face_ids and self.geometry._ids is self.surface._ids_cpu and
                all(getattr(self, name) is t for name, t in zip(self._names, self._buffers)) and
                tuple(t._version for t in self._buffers) == self._versions)

    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._buffers)
