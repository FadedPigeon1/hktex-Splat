"""Fresh GPU visibility snapshots and triangle/tile dispatch geometry for HKS.

Pixel/face arrays remain device tensors. Dynamic allocation still reads the
existing four scalar support counts; this is not a zero-synchronization path.
"""
from types import SimpleNamespace
import torch


class DeviceCameraSurface:
    @torch.no_grad()
    def __init__(self, faces, embedding, ids, bary, max_bytes=512*1024**2, *, pixels=None):
        if (ids.ndim != 2 or min(ids.shape)<1 or ids.dtype not in (torch.int32,torch.int64) or
                faces.ndim!=2 or faces.shape[1]!=3 or embedding.ndim!=2 or bary.shape != (*ids.shape,3) or
                ids.device != embedding.device or bary.device != embedding.device or faces.device != embedding.device or
                faces.requires_grad or embedding.requires_grad or bary.requires_grad):
            raise ValueError('Invalid fixed device camera inputs')
        if ids.numel()*(embedding.shape[1]*embedding.element_size()+128) > max_bytes:
            raise MemoryError('Device camera geometry exceeds budget')
        fast = (ids.is_cuda and ids.is_contiguous() and bary.is_contiguous() and
                embedding.dtype == bary.dtype == torch.float32 and
                pixels is not None and pixels.ndim == 1 and pixels.dtype == torch.int64 and
                pixels.device == ids.device and pixels.is_contiguous() and
                max(len(faces), ids.numel(), len(pixels)) < 2**31)
        if fast:
            self._prepare_fused(faces, embedding, ids, bary, pixels)
            return
        torch._assert_async(((ids >= -1) & (ids < len(faces))).all(),'Invalid raster triangle IDs')
        self.faces,self.embedding = faces,embedding
        self.ids = ids.clone(); self.shape = tuple(ids.shape); self.camera = None
        self.tile_size = 8
        if pixels is None:
            self.pixels = torch.where(self.ids.flatten() >= 0)[0]
        else:
            if pixels.ndim != 1 or pixels.dtype != torch.int64 or pixels.device != ids.device:
                raise ValueError('Invalid prepared foreground pixel indices')
            torch._assert_async(((pixels >= 0) & (pixels < ids.numel())).all() &
                (pixels[1:] > pixels[:-1]).all(), 'Prepared pixels must be ordered and in bounds')
            torch._assert_async((self.ids.flatten()[pixels] >= 0).all() &
                ((self.ids >= 0).sum() == len(pixels)), 'Prepared pixels differ from current visibility')
            self.pixels = pixels.clone()
        self.face_ids = self.ids.flatten()[self.pixels].long()
        self.bary = bary.reshape(-1,3)[self.pixels].to(embedding.dtype).clone()
        torch._assert_async((torch.isfinite(self.bary).all() & (self.bary >= -2e-6).all() &
                            ((self.bary.sum(1)-1).abs() <= 2e-6).all()),'Invalid device barycentrics')
        # Graph support defers pixel embeddings to diagnostics. Spectral support
        # uses this lazy gather for KNN; heat still comes from triangle corners.
        self._derived={}
        self.active_ordinals = torch.arange(len(self.pixels),device=ids.device,dtype=torch.int32)
        h,w = self.shape; self.tiles_x = (w+7)//8; self.tile_count = self.tiles_x*((h+7)//8)
        self.tiles = self.pixels//w//8*self.tiles_x+self.pixels%w//8
        self.tile_lengths = torch.zeros(self.tile_count, device=ids.device, dtype=torch.int64)
        self.tile_lengths.scatter_add_(0, self.tiles, torch.ones_like(self.tiles))
        self.tile_lengths = self.tile_lengths.int()
        self.active = self.tile_lengths > 0
        self.active_tile_count = min(self.tile_count,len(self.pixels))  # Safe queue-capacity upper bound.
        self.tile_pixels = torch.full((self.tile_count,64),-1,device=ids.device,dtype=torch.int32)
        # Physical 8x8 lanes need no pixel sort/scan. Every consumer checks -1,
        # so foreground holes stay empty instead of requiring compact tile rows.
        lanes = (self.pixels//w % 8)*8+self.pixels % w % 8
        self.tile_pixels[self.tiles, lanes] = self.active_ordinals
        self._names = ('faces','embedding','ids','pixels','face_ids','bary','active_ordinals','tiles','tile_lengths','active','tile_pixels')
        self._buffers = tuple(getattr(self,n) for n in self._names); self._versions = tuple(t._version for t in self._buffers)
        self.dispatch_geometry = DeviceCameraDispatch(self)
        self.device_visibility = True

    def _prepare_fused(self, faces, embedding, ids, bary, pixels):
        """Same immutable snapshot as the tensor path, packed in one device pass."""
        import triton
        from hktex_splatting.triton.heat_kernel_splatting import prepare_camera_surface
        self.faces, self.embedding = faces, embedding
        self.ids, self.pixels = ids.clone(), pixels.clone()
        self.shape = tuple(ids.shape); self.camera = None; self.tile_size = 8
        h, w = self.shape
        self.tiles_x = (w+7)//8; self.tile_count = self.tiles_x*((h+7)//8)
        if len(faces)*self.tile_count >= 2**63:
            raise ValueError('Device camera exceeds indexing limits')
        count, device = len(pixels), ids.device
        self.face_ids = torch.empty(count, device=device, dtype=torch.int64)
        self.bary = torch.empty((count,3), device=device, dtype=embedding.dtype)
        self.active_ordinals = torch.empty(count, device=device, dtype=torch.int32)
        self.tiles = torch.empty(count, device=device, dtype=torch.int64)
        self.tile_lengths = torch.zeros(self.tile_count, device=device, dtype=torch.int32)
        self.tile_pixels = torch.full((self.tile_count,64), -1, device=device, dtype=torch.int32)
        self._fused_pixel_counts = torch.zeros(len(faces), device=device, dtype=torch.int64)
        flags = torch.zeros(2, device=device, dtype=torch.int32)
        prepare_camera_surface[(triton.cdiv(max(ids.numel(),count),128),)](
            self.ids, bary, self.pixels, self.face_ids, self.bary, self.active_ordinals,
            self.tiles, self.tile_lengths, self.tile_pixels, self._fused_pixel_counts,
            flags, count, ids.numel(), len(faces), w, self.tiles_x, 128,
            num_warps=4, enable_fp_fusion=False)
        torch._assert_async((flags[0] == 0) & (flags[1] == count),
            'Invalid device camera IDs, foreground pixels or barycentrics')
        self.active = self.tile_lengths > 0
        self.active_tile_count = min(self.tile_count,count)
        self._derived = {}
        self._names = ('faces','embedding','ids','pixels','face_ids','bary','active_ordinals',
                       'tiles','tile_lengths','active','tile_pixels','_fused_pixel_counts')
        self._buffers = tuple(getattr(self,n) for n in self._names)
        self._versions = tuple(t._version for t in self._buffers)
        self.dispatch_geometry = DeviceCameraDispatch(self)
        self.device_visibility = True

    def valid(self):
        return (all(getattr(self,n) is t for n,t in zip(self._names,self._buffers)) and
                tuple(t._version for t in self._buffers) == self._versions and
                all(t._version==version for t,version in self._derived.values()))

    @property
    @torch.no_grad()
    def vertices(self):
        if 'vertices' not in self._derived:
            value=self.faces[self.face_ids].long()
            self._derived['vertices']=(value,value._version)
        return self._derived['vertices'][0]

    @property
    @torch.no_grad()
    def queries(self):
        if 'queries' not in self._derived:
            value=(self.embedding[self.vertices]*self.bary[:,:,None]).sum(1)
            self._derived['queries']=(value,value._version)
        return self._derived['queries'][0]


class DeviceCameraDispatch:
    @torch.no_grad()
    def __init__(self,surface):
        self.surface = surface
        count = len(surface.faces)
        if max(count,surface.ids.numel()) >= 2**31 or count*surface.tile_count >= 2**63:
            raise ValueError('Device camera exceeds indexing limits')
        keys,self.lengths = torch.unique(surface.face_ids*surface.tile_count+surface.tiles,sorted=True,return_counts=True)
        self.patch_tiles = keys%surface.tile_count
        self.face_counts = torch.zeros(count, device=surface.ids.device, dtype=torch.int64)
        self.face_counts.scatter_add_(0, keys//surface.tile_count, torch.ones_like(keys))
        self.face_starts = torch.cat((self.face_counts.new_zeros(1),self.face_counts.cumsum(0)))
        self.pixel_counts_device = getattr(surface, '_fused_pixel_counts', None)
        if self.pixel_counts_device is None:
            self.pixel_counts_device = torch.zeros_like(self.face_counts)
            self.pixel_counts_device.scatter_add_(0, surface.face_ids, torch.ones_like(surface.face_ids))
        self.geometry = SimpleNamespace(face_count=count,tile_size=8,bary=surface.bary,_ids=surface.ids,
            _layout=SimpleNamespace(patch_tiles=self.patch_tiles))
        self._names = ('lengths','patch_tiles','face_counts','face_starts','pixel_counts_device')
        self._buffers = tuple(getattr(self,n) for n in self._names);self._versions = tuple(t._version for t in self._buffers)
        self._policy=(self.geometry.face_count,self.geometry.tile_size,id(self.geometry._layout),id(self.geometry.bary),id(self.geometry._layout.patch_tiles))

    def valid(self):
        return (self.surface.valid() and self.geometry._ids is self.surface.ids and
                self._policy==(self.geometry.face_count,self.geometry.tile_size,id(self.geometry._layout),id(self.geometry.bary),id(self.geometry._layout.patch_tiles)) and
                all(getattr(self,n) is t for n,t in zip(self._names,self._buffers)) and
                tuple(t._version for t in self._buffers) == self._versions)

    @property
    def device_bytes(self):return sum(t.numel()*t.element_size() for t in self._buffers)

