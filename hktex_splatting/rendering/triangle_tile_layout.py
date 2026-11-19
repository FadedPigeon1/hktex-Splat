"""Build fixed triangle/tile layouts and ordered kernel/triangle incidences.

HKS geometry preparation uses this planner before GPU contribution scheduling.
This module contains layout data and record planning, not a standalone renderer.
Main API: Dispatch, kernel_face_order, dispatch."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Dispatch:
    """Store visible face/tile patches, kernel incidences and bounded pixel-record ranges."""
    pixels: np.ndarray             # Geometry-only face/tile fragment lists
    patch_offsets: np.ndarray
    patch_faces: np.ndarray
    patch_tiles: np.ndarray
    entries: np.ndarray            # Kernel-major map entries, NOT pixels
    patches: np.ndarray
    record_offsets: np.ndarray
    shape: tuple
    tile_size: int

    @property
    def fragments(self):return int(self.record_offsets[-1])

    def batches(self, max_records):
        if max_records < self.tile_size**2:
            raise ValueError('Record bound must fit one full tile')
        start=0
        while start<len(self.entries):
            end=int(np.searchsorted(self.record_offsets,
                self.record_offsets[start]+max_records,side='right')-1)
            end=min(end,len(self.entries))
            if end<=start:raise ValueError('No progress within memory bound')
            yield start,end
            start=end

    def records(self,start,end):
        patches=self.patches[start:end]
        n=np.diff(self.record_offsets[start:end+1])
        local=np.arange(int(n.sum()))-np.repeat(np.r_[0,np.cumsum(n)[:-1]],n)
        index=np.repeat(self.patch_offsets[patches],n)+local
        return self.pixels[index],np.repeat(self.entries[start:end],n)


def kernel_face_order(mapping):
    """Order kernel/triangle incidences and reject duplicates before tile expansion."""
    counts=np.diff(mapping.offsets)
    faces=np.repeat(np.arange(len(counts),dtype=np.int32),counts)
    # Reject duplicate kernel/face incidences: otherwise normalization doubles.
    order=np.lexsort((faces,mapping.kernels))
    if len(order)>1:
        a,b=order[:-1],order[1:]
        if np.any((faces[a]==faces[b])&(mapping.kernels[a]==mapping.kernels[b])):
            raise ValueError('Duplicate kernel-face incidence')
    return faces,order.astype(np.int32)


def dispatch(mapping,face_ids,tile_size=8,order=None,max_dispatch_bytes=512*1024**2):
    """Expand ordered kernel/triangle support over visible triangle/tile patches."""
    ids=np.asarray(face_ids)
    if ids.ndim!=2 or tile_size not in (1,2,4,8,16):raise ValueError('Invalid geometry/tile size')
    h,w=ids.shape;nf=len(mapping.offsets)-1
    if ids.dtype.kind not in 'iu' or np.any(ids>=nf) or np.any(ids < -1):raise ValueError('Face ID outside mesh')
    pixels=np.flatnonzero(ids.ravel()>=0)
    faces=ids.ravel()[pixels]
    ntx=(w+tile_size-1)//tile_size;nty=(h+tile_size-1)//tile_size
    tiles=(pixels//w//tile_size)*ntx+(pixels%w//tile_size)
    keys=faces*(ntx*nty)+tiles
    permutation=np.argsort(keys,kind='stable');keys=keys[permutation]
    pixels=pixels[permutation].astype(np.int32)
    starts=np.flatnonzero(np.r_[True,keys[1:]!=keys[:-1]]) if len(keys) else np.empty(0,np.int64)
    patch_offsets=np.r_[starts,len(keys)].astype(np.int64)
    patch_faces=(keys[starts]//(ntx*nty)).astype(np.int32)
    patch_tiles=(keys[starts]%(ntx*nty)).astype(np.int32)
    face_counts=np.bincount(patch_faces,minlength=nf)
    face_starts=np.r_[0,np.cumsum(face_counts)]
    entry_faces,entry_order=kernel_face_order(mapping) if order is None else order
    repetitions=face_counts[entry_faces[entry_order]]
    ntasks=int(repetitions.sum())
    if ntasks*24>max_dispatch_bytes:raise MemoryError('Forward dispatch exceeds configured CPU bound')
    entries=np.repeat(entry_order,repetitions)
    prefix=np.cumsum(repetitions)-repetitions
    local=np.arange(ntasks)-np.repeat(prefix,repetitions)
    patches=(np.repeat(face_starts[entry_faces[entry_order]],repetitions)+local).astype(np.int32)
    record_offsets=np.r_[0,np.cumsum(np.diff(patch_offsets)[patches])].astype(np.int64)
    return Dispatch(pixels,patch_offsets,patch_faces,patch_tiles,entries,patches,
                    record_offsets,ids.shape,tile_size)
