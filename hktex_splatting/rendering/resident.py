"""Share fixed dispatch geometry and live triangle coefficient topology with Heat-Kernel Splatting.

The baseline scene builder is retained only for benchmark RGB/gradient/optimizer gates.
The historical filename and Resident* class names are compatibility identifiers
for shared GPU geometry/topology and the internal triangle reference. The main
renderer is Heat-Kernel Splatting in heat_kernel_splatting.py.
Main API: ResidentDispatchGeometry, ResidentTopology, build_scene."""
import torch
import triton
from hktex_splatting.rendering.gpu_dispatch import DeviceDispatchGeometry, DeviceDispatch
from hktex_splatting.rendering.splat_autograd import PreparedScene
from hktex_splatting.knn_heat.triangle_kernel_support import (
    check_sizes,
    _scope,
)

class ResidentDispatchGeometry(DeviceDispatchGeometry):
    """Cache fixed GPU triangle layout shared by HKS and its validation reference."""
    def __init__(self, geometry):
        if max(geometry.face_count, geometry.face_ids.numel(), len(geometry._layout.patch_faces)) >= 2**31:
            raise ValueError('Resident geometry requires int32-indexable faces, pixels and patches')
        super().__init__(geometry)
        self.pixel_counts_device = torch.tensor(self.pixel_counts, device=geometry.bary.device)
        self._pixel_counts = self.pixel_counts_device
        self._pixel_version = self.pixel_counts_device._version

    def valid(self):
        return (self.pixel_counts_device is self._pixel_counts and
                self.pixel_counts_device._version == self._pixel_version and super().valid())

    @property
    def device_bytes(self):
        return super().device_bytes+self.pixel_counts_device.numel()*self.pixel_counts_device.element_size()


class ResidentTopology:
    """Build spectral coefficient indices for ordered kernel-triangle support."""
    def __init__(self, support, mesh, embedding):
        self.support, self.mesh, self.embedding = support, mesh, embedding
        self.scene_kernels = support.kernels
        self.coefficient_kernels = support.kernels.long()
        self.vertices = mesh.faces[support.entry_faces.long()].long()
        self._fixed = (mesh.faces, mesh.verts, embedding)
        self._fixed_versions = tuple(t._version for t in self._fixed)
        self._buffers = (support.counts, support.offsets, support.kernels, support.entry_faces,
                         self.coefficient_kernels, self.vertices)
        self._versions = tuple(t._version for t in self._buffers)

    def matches(self, mapping, mesh, embedding, device):
        requested = torch.device(device)
        return (mapping is self.support and mesh is self.mesh and embedding is self.embedding and
                mapping.valid() and
                requested.type == self.scene_kernels.device.type and
                (requested.index is None or requested == self.scene_kernels.device) and
                mesh.faces is self._fixed[0] and mesh.verts is self._fixed[1] and
                tuple(t._version for t in self._fixed) == self._fixed_versions and
                tuple(t._version for t in self._buffers) == self._versions)

    @property
    def device_bytes(self):
        return sum(t.numel()*t.element_size() for t in self._buffers)


@torch.no_grad()
def build_scene(support, geometry, mesh, embedding, *, max_records=1048576,
                max_dispatch_bytes=512*1024**2, renderer='baseline', profile=False):
    if renderer != 'baseline':
        raise ValueError('Unknown resident renderer')
    fixed = geometry.geometry
    if max_records < fixed.tile_size**2:
        raise ValueError('Record bound must fit one full tile')
    if (not geometry.valid() or not support.valid() or
            support.dispatch_geometry is not geometry or mesh.faces is not support.geometry.faces or
            embedding is not support.geometry.embedding or mesh.verts.device != fixed.bary.device):
        raise ValueError('Resident geometry changed or support belongs to another camera/mesh')
    check_sizes(len(support.kernels), support.maximum, support.tasks, support.fragments,
                max_records, max_dispatch_bytes)
    with _scope(profile, 'stable_kernel_face_order'):
        # CSR is face-major with ascending IDs. Stable sort matches (kernel,face).
        order = torch.argsort(support.kernels, stable=True)
        ordered_faces = support.entry_faces[order].long()
    with _scope(profile, 'task_expansion_and_scan'):
        repetitions = geometry.face_counts[ordered_faces]
        prefix = repetitions.cumsum(0)-repetitions
        entries = torch.repeat_interleave(order.to(torch.int32), repetitions, output_size=support.tasks)
        patches = (torch.repeat_interleave(geometry.face_starts[ordered_faces]-prefix,
                   repetitions, output_size=support.tasks)+torch.arange(support.tasks, device=entries.device)).int()
        offsets = torch.cat((repetitions.new_zeros(1), geometry.lengths[patches.long()].cumsum(0)))
        topology = ResidentTopology(support, mesh, embedding)
    plan = DeviceDispatch(entries, patches, offsets, fixed, support.fragments)
    # Immutable foreground union is safe even for faces with zero support.
    batches = [(0, support.tasks, support.fragments, geometry.foreground)] if support.tasks else []
    scene = PreparedScene(plan, support.kernels, fixed.bary, fixed.face_ids, entries, patches,
        offsets, fixed.patch_offsets, fixed.pixels, batches,
        triton.next_power_of_2(support.maximum+32), max_records, support.source_count, topology,
        renderer_backend=renderer)
    scene.dispatch_stats = dict(gpu_dispatch_used=True, gpu_dispatch_fallback=None,
        dispatch_explicit_d2h_calls=0, dispatch_d2h_bytes=0, dynamic_cache_hits=0,
        dispatch_fixed_extra_device_bytes=geometry.device_bytes,
        compact_bin_bytes=0, renderer_backend=renderer)
    return scene
