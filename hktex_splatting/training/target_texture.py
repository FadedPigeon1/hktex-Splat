"""Fixed target-image buffers and exact nearest-texel camera supervision.

Fixed pixels and UVs are uploaded once. The CPU-input path retains NumPy UV
interpolation; sample_device uses float64 device interpolation with a bounded
exact RGB check. Learned HKS parameters and shading outputs are never stored here.
"""
import numpy as np
import torch


@torch.no_grad()
def nearest_rgb(pixels, uv, normalization):
    """Trimesh repeat/nearest rule, including round-to-even texel boundaries."""
    height, width = pixels.shape[:2]
    uv = uv.to(device=pixels.device, dtype=torch.float64)
    x = torch.remainder(uv[:, 0]*(width-1), width).round().long().remainder(width)
    y = torch.remainder((1-uv[:, 1])*(height-1), height).round().long().remainder(height)
    return normalization[pixels[y, x].long()]


class TargetTexture:
    """Immutable target textures shared by training and held-out cameras.

    A bounded setup check compares device lookups with the original CPU helper.
    Recreate this object if target pixels/materials change intentionally.
    """
    def __init__(self, mesh, device='cuda', max_bytes=512*1024**2):
        self.device = torch.device(device)
        material = mesh.visual.material
        materials = getattr(material, 'materials', [material])
        assignments = getattr(mesh.visual, 'face_materials', None)
        if len(materials) > 1 and assignments is None:
            raise ValueError('Multiple target materials without face assignments')
        self.assignments = (np.zeros(len(mesh.faces), dtype=np.int64) if assignments is None
                            else np.asarray(assignments, dtype=np.int64).copy())
        if (self.assignments.shape != (len(mesh.faces),) or
                np.any(self.assignments < 0) or np.any(self.assignments >= len(materials))):
            raise ValueError('Invalid target material assignments')
        images = []
        for material in materials:
            image = getattr(material, 'baseColorTexture', None)
            if image is None: image = getattr(material, 'image', None)
            if image is None: raise ValueError('Missing target diffuse texture')
            images.append(image)
        required = sum(im.width*im.height*3 for im in images)+1024+len(mesh.faces)*(6*8+8)
        if required > max_bytes: raise MemoryError('Target texture upload exceeds configured budget')
        self.uv_corners = np.asarray(mesh.original_uv)[np.asarray(mesh.original_faces)].copy()
        self.assignments.flags.writeable = self.uv_corners.flags.writeable = False
        self._assignments, self._uv_corners = self.assignments, self.uv_corners
        self.device_uv = torch.tensor(self.uv_corners,device=self.device,dtype=torch.float64)
        self.device_materials = torch.tensor(self.assignments,device=self.device)
        self.device=self.device_uv.device
        # The CPU reference divides uint8 in float64 then stores float32. A
        # 256-value table avoids introducing division rounding differences.
        self.normalization = torch.tensor((np.arange(256, dtype=np.float64)/255).astype(np.float32), device=self.device)
        self.pixels = [torch.tensor(np.array(im.convert('RGBA'), copy=True)[:, :, :3].copy(),
                                    device=self.device) for im in images]
        self.device_bytes = required
        self._pixels = tuple(self.pixels)
        self._normalization = self.normalization
        self._device_geometry = (self.device_uv,self.device_materials)
        self._versions = tuple(t._version for t in (*self.pixels, self.normalization,*self._device_geometry))
        self.validation = self._validate(images)
        self.device_interpolation_checked = False

    def valid(self):
        return (len(self.pixels) == len(self._pixels) and
                self.assignments is self._assignments and self.uv_corners is self._uv_corners and
                all(a is b for a,b in zip(self.pixels,self._pixels)) and
                self.normalization is self._normalization and
                all(a is b for a,b in zip((self.device_uv,self.device_materials),self._device_geometry)) and
                tuple(t._version for t in (*self.pixels,self.normalization,*self._device_geometry)) == self._versions and
                not self.assignments.flags.writeable and not self.uv_corners.flags.writeable)

    def _validate(self, images):
        import trimesh
        rng = np.random.default_rng(2718)  # Does not consume training RNG.
        for image, pixels in zip(images,self.pixels):
            uv = rng.uniform(-3,3,(4096,2))
            # Explicit wrap endpoints and half-pixel rounding cases.
            ties = np.arange(min(64,image.width), dtype=np.float64)+.5
            special = np.column_stack((ties/max(1,image.width-1),
                                       1-ties/max(1,image.height-1)))
            uv = np.concatenate((uv,special,np.array([[0,0],[1,1],[-1,-1],[2,2]],dtype=np.float64)))
            expected = (trimesh.visual.uv_to_color(uv,image)[:,:3]/255).astype(np.float32)
            actual = nearest_rgb(pixels,torch.tensor(uv,device=self.device),self.normalization).cpu().numpy()
            if not np.array_equal(actual,expected):
                raise RuntimeError('Target texture lookup differs from original Trimesh; use CPU targets')
        return dict(passed=True, exact=True, materials=len(images), device=str(self.device),
                    scope='Setup nearest-texel lookup check; UV interpolation remains the original NumPy einsum')

    @torch.no_grad()
    def sample(self, face_ids, bary):
        if not self.valid(): raise ValueError('Target texture buffers changed; rebuild them')
        face_ids, bary = np.asarray(face_ids), np.asarray(bary)
        if (face_ids.ndim != 1 or face_ids.dtype.kind not in 'iu' or bary.shape != (len(face_ids),3) or
                np.any(face_ids < 0) or np.any(face_ids >= len(self.assignments))):
            raise ValueError('Invalid target face/barycentric samples')
        uv = np.einsum('ij,ijk->ik',bary,self.uv_corners[face_ids])
        if not np.isfinite(uv).all(): raise ValueError('Nonfinite target UVs')
        result = torch.empty((len(face_ids),3),device=self.device,dtype=torch.float32)
        materials = self.assignments[face_ids]
        for index,pixels in enumerate(self.pixels):
            selected = np.flatnonzero(materials == index)
            if not len(selected): continue
            colors = nearest_rgb(pixels,torch.tensor(uv[selected],device=self.device),self.normalization)
            result[torch.tensor(selected,device=self.device)] = colors
        return result

    @torch.no_grad()
    def sample_device(self, face_ids, bary):
        """Interpolate fixed UVs and fetch target RGB without per-frame host arrays.

        The first bounded sample batch is checked against the CPU target helper;
        this diagnostic transfer is recorded separately from production frames.
        """
        if not self.valid(): raise ValueError('Target texture buffers changed; rebuild them')
        if face_ids.device != self.device or bary.device != self.device or bary.shape != (len(face_ids),3):
            raise ValueError('Invalid device target samples')
        torch._assert_async(((face_ids>=0)&(face_ids<len(self.assignments))).all(),'Invalid target faces')
        corners=self.device_uv[face_ids.long()]; b=bary.double()
        uv=(b[:,0,None]*corners[:,0]+b[:,1,None]*corners[:,1])+b[:,2,None]*corners[:,2]
        torch._assert_async(torch.isfinite(uv).all(),'Nonfinite target UVs')
        result=torch.empty((len(face_ids),3),device=self.device,dtype=torch.float32)
        materials=self.device_materials[face_ids.long()]
        for index,pixels in enumerate(self.pixels):
            selected=torch.where(materials==index)[0]
            result[selected]=nearest_rgb(pixels,uv[selected],self.normalization)
        if not self.device_interpolation_checked and len(face_ids):
            count=min(1024,len(face_ids))
            expected=self.sample(face_ids[:count].cpu().numpy(),bary[:count].cpu().numpy())
            if not torch.equal(result[:count],expected):
                raise RuntimeError('Device target interpolation differs from CPU RGB; no diagnostic waiver applies')
            self.device_interpolation_checked=True
            self.validation['device_interpolation']=dict(passed=True,exact=True,samples=count,
                scope='First bounded device frame sample; not an exhaustive texel-boundary proof')
        return result
