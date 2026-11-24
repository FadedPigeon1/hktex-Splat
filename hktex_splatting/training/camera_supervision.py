"""Rasterize fresh fixed-mesh camera frames for the adaptive HKTex/HKS loop.

Targets are texture RGB or explicitly configured linear lit RGB. Camera frames replace the UV
importance pool; model, kernel motion, density callbacks and loss remain live.
"""
import time
import json
import hashlib
from collections import OrderedDict
import numpy as np
import torch


def cpu_array(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


class CameraSupervision(torch.utils.data.IterableDataset):
    def __init__(self, original, resolution, views=('top','side','three_quarter','logo_closeup'),
                 *, target_backend='cpu', target_sampler=None, max_texture_bytes=512*1024**2,
                 visibility='fresh', visibility_cache_bytes=64*1024**2,
                 raster_backend='cpu', raster_near=.01, lighting=None, camera_path=None):
        super().__init__()
        from hktex_splatting.rendering.mesh_visibility_raster import BVHRaster
        from hktex_splatting.rendering.cameras import cameras
        self.original, self.cfg, self.tex_img = original, original.cfg, original.tex_img
        self.mesh = original.mesh
        self.target_mesh = self.mesh.copy()
        self.target_mesh.original_uv = cpu_array(original.uv).copy()
        self.target_mesh.original_faces = cpu_array(original.tex_faces).copy()
        self.lighting = dict(lighting or {'mode': 'none'})
        self._lighting_normals = {}
        if self.lighting['mode'] != 'none':
            corners = np.asarray(self.mesh.vertices)[np.asarray(self.mesh.faces)]
            normals = np.cross(corners[:, 1]-corners[:, 0], corners[:, 2]-corners[:, 0])
            self._face_normals = (normals/np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)).astype(np.float32)
        if target_backend not in ('cpu','gpu'): raise ValueError('Unknown target sampling backend')
        self.target_backend = target_backend
        self.target_sampler = target_sampler
        if target_backend == 'gpu' and self.target_sampler is None:
            from hktex_splatting.training.target_texture import TargetTexture
            self.target_sampler = TargetTexture(self.target_mesh,device='cuda',max_bytes=max_texture_bytes)
        self.raster_backend,self.raster_near=raster_backend,raster_near
        if raster_backend == 'nvdiffrast':
            if visibility != 'fresh' or target_backend != 'gpu':
                raise ValueError('GPU raster requires fresh visibility and GPU targets')
            from hktex_splatting.rendering.cuda_mesh_raster import CudaMeshRaster
            self.raster=CudaMeshRaster(self.mesh.vertices,self.mesh.faces,near=raster_near)
            self.device_vertices=self.raster.vertices
            self.device_faces=self.raster.faces.long()
        elif raster_backend == 'cpu':
            self.raster = BVHRaster(np.asarray(self.mesh.vertices), np.asarray(self.mesh.faces))
        else:raise ValueError('Unknown camera raster backend')
        self.cameras = {name: dict(c, width=resolution, height=resolution) for name,c in cameras(path=camera_path).items()}
        self.views = views
        if visibility not in ('fresh','reuse') or visibility_cache_bytes < 1:
            raise ValueError('Invalid visibility policy/budget')
        self.visibility, self.visibility_cache_bytes = visibility, visibility_cache_bytes
        self._visibility = OrderedDict()
        self._fixed_vertices = np.asarray(self.mesh.vertices).copy()
        self._fixed_faces = np.asarray(self.mesh.faces).copy()
        self.raster_calls = self.visibility_hits = self.visibility_invalidations = 0
        self.visibility_bytes = 0
        self.frames = 0
        self.raster_seconds = self.target_seconds = 0.

    def visibility_buffers(self, name):
        """Rasterize fresh frames, or explicitly reuse immutable CPU view buffers."""
        if (not np.array_equal(self.mesh.vertices,self._fixed_vertices) or
                not np.array_equal(self.mesh.faces,self._fixed_faces)):
            raise ValueError('Fixed raster mesh changed; rebuild the camera dataset')
        signature = json.dumps(self.cameras[name],sort_keys=True,allow_nan=False)
        existing = self._visibility.get(name)
        if self.visibility == 'reuse' and existing is not None:
            if existing[0] == signature:
                if existing[1].flags.writeable or existing[2].flags.writeable:
                    raise ValueError('Retained visibility became writable; rebuild the dataset')
                self.visibility_hits += 1; self._visibility.move_to_end(name)
                return existing[1],existing[2],existing[3]
            old = self._visibility.pop(name)
            self.visibility_bytes -= old[1].nbytes+old[2].nbytes
            self.visibility_invalidations += 1
        start = time.perf_counter()
        if self.raster_backend == 'nvdiffrast':
            ids,bary,_=self.raster.buffers(self.cameras[name])
        else:_, _, _, ids, bary = self.raster.all_buffers(self.cameras[name])
        self.raster_seconds += time.perf_counter()-start
        self.raster_calls += 1
        key = None
        if self.visibility == 'reuse':
            size = ids.nbytes+bary.nbytes
            if size > self.visibility_cache_bytes:
                raise MemoryError('One visibility frame exceeds reuse budget; use fresh or increase budget')
            while self._visibility and (len(self._visibility) >= 4 or self.visibility_bytes+size > self.visibility_cache_bytes):
                _,old = self._visibility.popitem(last=False)
                self.visibility_bytes -= old[1].nbytes+old[2].nbytes
            ids, bary = ids.copy(),bary.copy()
            ids.flags.writeable = bary.flags.writeable = False
            key = name+':'+hashlib.sha256(signature.encode()).hexdigest()
            self._visibility[name] = (signature,ids,bary,key)
            self.visibility_bytes += size
        return ids,bary,key

    @torch.no_grad()
    def shade_target(self, batch, name):
        """Prepare fixed face normals/current view directions and shared lit targets."""
        if self.lighting['mode'] == 'none':
            return batch
        start = time.perf_counter()
        from hktex_splatting.rendering.reference_mesh import shade_batch
        colour = batch['colour']
        key = (colour.device, colour.dtype)
        if key not in self._lighting_normals:
            self._lighting_normals[key] = torch.as_tensor(self._face_normals, device=colour.device, dtype=colour.dtype)
        batch['lighting_normal'] = self._lighting_normals[key][batch['face_id'].to(colour.device).long()]
        batch['lighting_view'] = colour.new_tensor(self.cameras[name]['position'])-batch['pos'].to(colour)
        batch['colour'] = shade_batch(colour, batch, self.lighting)
        self.target_seconds += time.perf_counter()-start
        return batch

    def __iter__(self):
        from hktex_splatting.rendering.reference_mesh import texture_sample
        index = 0
        while True:
            name = self.views[index % len(self.views)]; index += 1
            ids,bary,key = self.visibility_buffers(name)
            start = time.perf_counter()
            if self.raster_backend == 'nvdiffrast':
                pixels=torch.where(ids.flatten()>=0)[0]
                if not len(pixels):raise ValueError('Camera frame has no visible mesh pixels')
                fi=ids.flatten()[pixels];bc=bary.reshape(-1,3)[pixels]
                colour=self.target_sampler.sample_device(fi,bc)
                pos=(self.device_vertices[self.device_faces[fi]]*bc[:,:,None]).sum(1)
                self.target_seconds+=time.perf_counter()-start;self.frames+=1
                yield self.shade_target(dict(pos=pos,face_id=fi,bary=bc,colour=colour,
                    _camera_ids=ids,_camera_bary=bary,_camera_pixels=pixels,_camera_device=True,
                    _camera_frame=dict(view=name,index=index,camera=self.cameras[name].copy())), name)
                continue
            rgb = None
            if self.target_backend == 'cpu': rgb, _ = texture_sample(self.target_mesh, ids, bary, 'nearest')
            mask = ids >= 0
            if not mask.any(): raise ValueError('Camera frame has no visible mesh pixels')
            fi = ids[mask].astype(np.int64)
            bc = bary[mask].astype(np.float32)
            colour = (torch.tensor(rgb[mask]) if self.target_backend == 'cpu' else
                      self.target_sampler.sample(fi,bary[mask]))
            pos = (np.asarray(self.mesh.vertices)[np.asarray(self.mesh.faces)[fi]]*bc[:,:,None]).sum(1)
            self.target_seconds += time.perf_counter()-start
            self.frames += 1
            batch = dict(pos=torch.tensor(pos,dtype=torch.float32), face_id=torch.tensor(fi),
                bary=torch.tensor(bc), colour=colour,
                _camera_ids=ids.copy(), _camera_bary=bary.copy())
            if key is not None: batch['_camera_key'] = key
            yield self.shade_target(batch, name)


def validation_subset(data, samples):
    """Keep the first visible pixels in their true screen positions for gates."""
    result = {key:value[:samples] for key,value in data.items() if not key.startswith('_camera_')}
    if '_camera_ids' in data:
        if data.get('_camera_device',False):
            ids=data['_camera_ids'].clone()
            foreground=torch.where(ids.flatten()>=0)[0]
            ids.view(-1)[foreground[samples:]]=-1
            result.update(_camera_ids=ids,_camera_bary=data['_camera_bary'].clone(),_camera_device=True)
            return result
        ids = cpu_array(data['_camera_ids']).copy()
        foreground = np.flatnonzero(ids >= 0)
        ids.flat[foreground[samples:]] = -1
        result['_camera_ids'] = ids
        result['_camera_bary'] = cpu_array(data['_camera_bary']).copy()
    return result
