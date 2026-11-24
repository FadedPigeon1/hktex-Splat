"""Fresh CUDA triangle rasterization through optional nvdiffrast.

Clip-space projection and image-row orientation match this repository's camera
basis. GPU raster edge/depth precision differs from the float64 CPU reference.
"""
import math
import numpy as np
import torch


def clip_positions(vertices,camera,near):
    from hktex_splatting.rendering.camera_raster import perspective_camera_basis
    origin,basis=perspective_camera_basis(camera['position'],camera['target'])
    relative=vertices-torch.tensor(origin,device=vertices.device,dtype=vertices.dtype)
    rotation=torch.tensor(basis,device=vertices.device,dtype=vertices.dtype)
    # Camera geometry must not inherit the trainer's TF32 GEMM policy. Keep
    # explicit float32 products/sums rather than a potentially TF32 matmul.
    xyz=(relative[:,0,None]*rotation[0]+relative[:,1,None]*rotation[1])+relative[:,2,None]*rotation[2]
    ty=math.tan(math.radians(camera['fov_y_degrees'])/2);tx=ty*camera['width']/camera['height']
    # Infinite far plane; explicit GPU near plane avoids catastrophic depth
    # quantization from the CPU prepass's extremely small near setting.
    return torch.stack((xyz[:,0]/tx,xyz[:,1]/ty,xyz[:,2]-2*near,xyz[:,2]),1)


class CudaMeshRaster:
    def __init__(self,vertices,faces,device='cuda',near=.01):
        if not math.isfinite(near) or near <= 0:raise ValueError('GPU near plane must be finite and positive')
        if not 0<len(faces)<2**24:raise ValueError('GPU raster requires nonempty float32-exact triangle IDs')
        try:import nvdiffrast.torch as dr
        except ImportError as exc:raise RuntimeError('Install optional nvdiffrast with a compatible CUDA toolkit; no silent CPU fallback') from exc
        self.dr,self.near=dr,near
        self.vertices=torch.tensor(np.asarray(vertices),device=device,dtype=torch.float32)
        self.faces=torch.tensor(np.asarray(faces),device=device,dtype=torch.int32)
        self.context=dr.RasterizeCudaContext(device=self.vertices.device)
        self._versions=(self.vertices._version,self.faces._version)
        self.validation=self.validate()

    @torch.no_grad()
    def validate(self):
        """Small hardware check for row orientation, perspective, occlusion and clipping.

        Only this bounded setup diagnostic downloads raster buffers. Shared-edge
        coverage may differ from the float64 CPU raster; interior faces must agree.
        """
        from hktex_splatting.rendering.camera_raster import rasterize_visible_triangles
        camera=dict(position=[0,0,0],target=[0,0,1],width=32,height=32,fov_y_degrees=60.)
        vertices=np.array([[-.6,-.5,1.],[.6,-.5,1.6],[0,.65,1.2],
                           [-.15,-.15,.5],[.15,-.15,.5],[0,.2,.5]],dtype=np.float32)
        faces=np.array([[0,1,2],[3,4,5]],dtype=np.int32)
        clip=clip_positions(torch.tensor(vertices,device=self.vertices.device),camera,.01)
        rast,_=self.dr.rasterize(self.context,clip[None],torch.tensor(faces,device=clip.device),resolution=[32,32],grad_db=False)
        rast=rast[0].flip(0).cpu().numpy()
        actual=rast[:,:,3].astype(np.int64)-1
        bary=np.concatenate((rast[:,:,:2],1-rast[:,:,:2].sum(-1,keepdims=True)),-1)
        _,_,_,expected,bc=rasterize_visible_triangles(vertices,faces,np.arange(2),
            camera['position'],camera['target'],32,32,60.)
        interior=(expected>=0)&(bc.min(-1)>.05)
        gpu_interior=(actual>=0)&(bary.min(-1)>.05)
        if (not interior.any() or not np.array_equal(actual[interior],expected[interior]) or
                not np.array_equal(actual[gpu_interior],expected[gpu_interior])):
            raise RuntimeError('GPU raster interior visibility disagrees with CPU reference')
        error=float(np.abs(bary[interior]-bc[interior]).max())
        if error>2e-5:raise RuntimeError(f'GPU raster perspective barycentrics disagree with CPU reference; max_abs={error:.9g}, tolerance=2e-5')
        behind=torch.tensor(vertices[:3]*.005,device=clip.device)
        clipped,_=self.dr.rasterize(self.context,clip_positions(behind,camera,.01)[None],
            torch.tensor([[0,1,2]],device=clip.device,dtype=torch.int32),resolution=[32,32],grad_db=False)
        if bool((clipped[:,:,:,3]!=0).any()):raise RuntimeError('GPU raster near-plane clipping failed')
        return dict(passed=True,interior_samples=int(interior.sum()),bary_max_abs=error,
            scope='32-square synthetic GPU setup check; raster edges are not bit-exact CPU coverage')

    @torch.no_grad()
    def buffers(self,camera):
        if self._versions!=(self.vertices._version,self.faces._version):raise ValueError('Fixed raster geometry changed')
        clip=clip_positions(self.vertices,camera,self.near)
        rast,_=self.dr.rasterize(self.context,clip[None].contiguous(),self.faces,
                                resolution=[camera['height'],camera['width']],grad_db=False)
        # nvdiffrast stores bottom-up image rows; existing camera images are top-down.
        rast=rast[0].flip(0)
        ids=rast[:,:,3].long()-1
        bary=torch.cat((rast[:,:,:2],1-rast[:,:,:2].sum(-1,keepdim=True)),-1)
        bary=torch.where((ids>=0)[:,:,None],bary,0.)
        return ids,bary,rast[:,:,2]
