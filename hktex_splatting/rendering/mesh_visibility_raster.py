"""Fixed-mesh BVH visibility and camera-frustum traversal.

CPU fixed-mesh visibility before spectral rendering.
Main API: BVHRaster."""
import math
import numpy as np
from hktex_splatting.rendering.fast_raster import StaticRaster, _fast_kernel
from hktex_splatting.rendering.camera_raster import perspective_camera_basis


class BVHRaster(StaticRaster):
    """Cull fixed mesh triangles conservatively, then rasterize visible face IDs and barycentrics."""
    def __init__(self,vertices,faces,leaf_size=64):
        super().__init__(vertices,faces)
        triangles=self.vertices[self.faces]
        low,high=triangles.min(1),triangles.max(1)
        self.nodes=[]
        def build(ids):
            lo,hi=low[ids].min(0),high[ids].max(0)
            node=len(self.nodes)
            self.nodes.append(None)
            if len(ids)<=leaf_size:
                self.nodes[node]=(lo,hi,ids,-1,-1)
            else:
                centers=(low[ids]+high[ids])*.5
                axis=int(np.ptp(centers,axis=0).argmax())
                order=np.argsort(centers[:,axis],kind="stable")
                half=len(ids)//2
                left,right=build(ids[order[:half]]),build(ids[order[half:]])
                self.nodes[node]=(lo,hi,ids,left,right)
            return node
        if len(faces):
            build(np.arange(len(faces),dtype=np.int64))
        self.last_counts={}

    def all_buffers(self,camera):
        origin,basis=perspective_camera_basis(camera["position"],camera["target"])
        width,height=camera["width"],camera["height"]
        ty=math.tan(math.radians(camera["fov_y_degrees"])/2)
        tx=ty*width/height
        normals=np.array([[0.,0.,1.],[1.,0.,tx],[-1.,0.,tx],[0.,1.,ty],[0.,-1.,ty]])
        offsets=np.array([self.near,0.,0.,0.,0.])
        world=normals@basis.T
        positive=np.maximum(world,0)
        negative=np.minimum(world,0)
        stack=[0] if self.nodes else []
        selected=[];visited=0
        while stack:
            node=stack.pop();visited+=1
            lo,hi,ids,left,right=self.nodes[node]
            # Upper endpoint of every plane dot product, padded outwards to
            # cover float64 camera transform reassociation at the boundary.
            upper=positive@(hi-origin)+negative@(lo-origin)-offsets
            tolerance=1e-11*(1+np.linalg.norm(hi-origin)+np.linalg.norm(lo-origin))
            if (upper < -tolerance).any():
                continue
            lower=positive@(lo-origin)+negative@(hi-origin)-offsets
            if left<0 or (lower>tolerance).all():
                selected.append(ids)
            else:
                stack.extend((right,left))
        ids=np.sort(np.concatenate(selected)) if selected else np.empty(0,np.int64)
        self.last_counts=dict(raster_bvh_nodes_visited=visited,raster_faces_dispatched=len(ids),mesh_faces=len(self.faces))
        return _fast_kernel(self.vertices,self.faces,ids,origin,basis,width,height,tx,ty,normals,offsets)
