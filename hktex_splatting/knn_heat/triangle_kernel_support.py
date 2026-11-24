"""Kernel-centric mesh support and retained spectral-selection reference utilities.

KernelSurfaceSupport builds fresh ordered CSR from source-face graph neighborhoods
or opt-in global spectral pixel KNN. ResidentSupportGeometry/FusedResidentSupportGeometry keep
legacy spectral selection and safe fallback for internal reference workflows.
No learned heat or coefficients are retained by fixed topology lookup tables.
Main API: KernelSurfaceSupport, DeviceSupport."""
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
import triton
from hktex_splatting.knn_heat.packed_support import PreparedSupport
from hktex_splatting.triton import resident_gpu


class KernelSurfaceSupport:
    """Fresh graph neighborhoods or native spectral pixel/triangle support.

    In graph mode a hop limit is a truncation, NOT a conservative spectral influence
    bound. Traversal never stops on heat/distance values. Only fixed connectivity
    is retained across updates; motion, shape changes and density transactions
    always receive fresh pairs and differentiable coefficients.
    """
    def __init__(self, faces, embedding, hops=8, max_bytes=512*1024**2, *, domain='graph'):
        if hops<0 or len(faces)<1:raise ValueError('Expected nonempty mesh and nonnegative hops')
        if domain not in ('graph', 'spectral'):raise ValueError('Unknown support domain')
        self.domain = domain
        self.faces,self.embedding,self.hops=faces,embedding,hops
        self._versions=(faces._version,embedding._version)
        if domain == 'spectral':
            # This mode uses the native Euclidean spectral KNN policy, including
            # across disconnected mesh components. It needs no adjacency table.
            self.adjacency = faces.new_empty((0, 0))
            self._adjacency, self._adjacency_version = self.adjacency, self.adjacency._version
            self.hop_neighbors = self._hop_neighbors = None
            self._hop_version = self._support_faces = None
            return
        # Build edge connectivity once, including all incident faces at a
        # nonmanifold edge; disconnected components remain disconnected.
        mesh=faces.detach().cpu().numpy();edges={}
        for face,vertices in enumerate(mesh):
            for a,b in ((vertices[0],vertices[1]),(vertices[1],vertices[2]),(vertices[2],vertices[0])):
                edges.setdefault((min(int(a),int(b)),max(int(a),int(b))),set()).add(face)
        adjacent=[set() for _ in mesh]
        for incident in edges.values():
            for face in incident:adjacent[face].update(incident-{face})
        width=max(1,max(map(len,adjacent)))
        if len(mesh)*width*8>max_bytes:raise MemoryError('Fixed adjacency exceeds support budget')
        table=np.full((len(mesh),width),-1,np.int64)
        for face,neighbors in enumerate(adjacent):table[face,:len(neighbors)]=sorted(neighbors)
        self.adjacency=torch.tensor(table,device=faces.device)
        self._adjacency_version=self.adjacency._version
        self._adjacency=self.adjacency
        # The hop relation is fixed mesh geometry, not a learned value. Build it
        # once when bounded; querying the CURRENT source face then replaces
        # repeated GPU frontier expansion/sorting without changing support.
        self.hop_neighbors=None
        degree=max(map(len,adjacent))
        bound=1+sum(degree*max(0,degree-1)**i for i in range(hops)) if hops<=8 else None
        fits_bound = bound is not None and len(mesh)*bound*4+self.adjacency.numel()*8<=max_bytes//2
        try_sparse = hops > 8 and len(mesh)*4+self.adjacency.numel()*8<=max_bytes//2
        if fits_bound or try_sparse:
            from scipy.sparse import csr_matrix,eye
            valid=table>=0
            row=np.broadcast_to(np.arange(len(mesh))[:,None],table.shape)[valid]
            graph=csr_matrix((np.ones(int(valid.sum()),dtype=np.bool_),(row,table[valid])),shape=(len(mesh),len(mesh)))
            reach=eye(len(mesh),dtype=np.bool_,format='csr')
            lookup_possible = True
            for _ in range(hops):
                # Bound sparse product/add workspace conservatively, including
                # possible int64 indices. Host preprocessing is bounded separately
                # from the unchanged half-budget for the device lookup table.
                if reach.nnz*(2*degree+2)*9+(len(mesh)+1)*32 > max_bytes*4:
                    lookup_possible = False
                    break
                previous = reach.nnz
                reach=(reach+reach@graph).tocsr()
                if reach.nnz == previous:break  # Monotone graph union has converged.
            if lookup_possible:
                reach.sum_duplicates();reach.sort_indices()
                lengths=np.diff(reach.indptr);width=max(1,int(lengths.max()))
                if len(mesh)*width*4+self.adjacency.numel()*8 <= max_bytes//2:
                    fixed=np.full((len(mesh),width),-1,dtype=np.int32)
                    row=np.repeat(np.arange(len(mesh)),lengths)
                    column=np.arange(reach.nnz)-np.repeat(reach.indptr[:-1],lengths)
                    fixed[row,column]=reach.indices
                    self.hop_neighbors=torch.tensor(fixed,device=faces.device)
        self._hop_neighbors=self.hop_neighbors
        self._hop_version=None if self.hop_neighbors is None else self.hop_neighbors._version
        self._support_faces = None

    @property
    def support_faces(self):
        """Materialize the full source-major support only for explicit inspection."""
        if self._support_faces is None:
            if not self.valid():raise ValueError('Fixed support geometry changed')
            if self.domain == 'spectral':
                faces, kernels = self._spectral_pairs
                self._support_faces = (kernels.long()*len(self.faces)+faces).sort().values % len(self.faces)
                return self._support_faces
            runtime = getattr(self, '_runtime_neighbors', None)
            neighbors = (runtime.sort(dim=1).values if runtime is not None else
                         self.hop_neighbors[self._source_faces_snapshot.long()])
            self._support_faces = neighbors[neighbors >= 0].long()
        return self._support_faces

    @support_faces.setter
    def support_faces(self, value):
        self._support_faces = value

    def valid(self):
        return (self._versions==(self.faces._version,self.embedding._version) and
            self.adjacency is self._adjacency and self.adjacency._version==self._adjacency_version and
            self.hop_neighbors is self._hop_neighbors and
            (self.hop_neighbors is None or self.hop_neighbors._version==self._hop_version))

    @torch.no_grad()
    def select_spectral(self, sources, surface, neighbors, *, dispatch_geometry,
                        max_records, max_dispatch_bytes, pool_cap=0, profile=False, reference=False):
        """Invert native pixel KNN into triangle CSR without a mesh-hop cutoff.

        Each forward searches CURRENT source embeddings with bounded float64
        matrix distances. The independent oracle uses direct cdist instead.
        Coefficients are shared by the union of pixel candidates on each triangle;
        the emitter retains each pixel's own KNN membership. Selection is discrete
        just as in native HKTex; floating coefficients remain differentiable.
        """
        if self.domain != 'spectral' or not self.valid() or not dispatch_geometry.valid():
            raise ValueError('Invalid spectral support geometry')
        n, f = len(sources), len(self.faces)
        if n < 1 or n >= 2**31 or n*f >= 2**63 or neighbors < 1:
            raise ValueError('Invalid spectral source count or neighbor rank')
        if sources.device != self.embedding.device or sources.shape[1:] != self.embedding.shape[1:]:
            raise ValueError('Source embeddings differ from fixed geometry')
        k = min(neighbors, n)
        if len(surface.pixels)*k > max_records or len(surface.pixels)*k*24 > max_dispatch_bytes:
            raise MemoryError('Spectral pixel candidates exceed configured budget')
        with _scope(profile, 'spectral_pixel_candidates'):
            live = sources.detach().double()
            norms = live.square().sum(1)
            candidates = torch.empty((len(surface.pixels), k), device=sources.device, dtype=torch.long)
            chunk = min(4096, max_dispatch_bytes//max(1, n*8*4))
            if chunk < 1:raise MemoryError('Spectral distance scratch exceeds configured budget')
            for begin in range(0, len(surface.pixels), chunk):
                query = surface.queries[begin:begin+chunk].double()
                if reference:
                    distance = torch.cdist(query, live, compute_mode='donot_use_mm_for_euclid_dist')
                else:
                    distance = torch.addmm(query.square().sum(1, keepdim=True)+norms[None, :],
                                           query, live.T, beta=1, alpha=-2).clamp_min_(0)
                candidates[begin:begin+len(query)] = distance.topk(k, dim=1, largest=False,
                    sorted=False).indices.sort(dim=1).values
        with _scope(profile, 'spectral_triangle_union'):
            keys = (surface.face_ids[:, None]*n+candidates).flatten().unique(sorted=True)
            entry_faces, kernels = keys//n, keys%n
            counts = torch.bincount(entry_faces, minlength=f)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            maximum, tasks, fragments = torch.stack((counts.amax(),
                (counts*dispatch_geometry.face_counts).sum(),
                (counts*dispatch_geometry.pixel_counts_device).sum())).cpu().tolist()
            if pool_cap and maximum > pool_cap:
                raise RuntimeError('Spectral triangle union exceeds --support-cap; no pairs dropped')
            check_sizes(len(keys), maximum, tasks, fragments, max_records,
                        max_dispatch_bytes, maximum_limit=None)
            mapping = DeviceSupport(counts, offsets, kernels.int(), entry_faces.int(), n,
                                    maximum, tasks, fragments, self, dispatch_geometry)
            self._spectral_pairs, self._support_faces = (entry_faces, kernels), None
            self.source_offsets = torch.cat((kernels.new_zeros(1),
                torch.bincount(kernels, minlength=n).cumsum(0)))
        return mapping, candidates, dict(backend='spectral_pixel_union', approximate=False,
            source_count=n, neighbor_rank=k, retained_pairs=len(keys),
            maximum_face_support=maximum, dynamic_topology_policy='fresh_spectral_knn_each_forward',
            support_engine='spectral_knn_triangle_union',
            influence_bound='Per-pixel float64 spectral KNN membership; no graph-hop truncation')

    @torch.no_grad()
    def select(self, source_faces, pool_cap=0, *, dispatch_geometry, max_records=1048576,
               max_dispatch_bytes=512*1024**2, profile=False, validation_candidates=None,
               reference=False):
        if self.domain != 'graph' and validation_candidates is None:
            raise ValueError('Spectral support requires select_spectral and current pixel queries')
        if not self.valid() or not dispatch_geometry.valid():raise ValueError('Fixed support geometry changed')
        f=len(self.faces);n=len(source_faces);device=self.faces.device
        if (source_faces.device!=device or source_faces.ndim!=1 or
                source_faces.dtype not in (torch.int32,torch.int64) or n>=2**31 or n*f>=2**63 or
                dispatch_geometry.pixel_counts_device.numel()!=f or dispatch_geometry.face_counts.numel()!=f):
            raise ValueError('Invalid source face table or dispatch geometry')
        torch._assert_async(((source_faces>=0)&(source_faces<f)).all(),'Invalid source triangle')
        limit=min(max_records,max_dispatch_bytes//64)
        frontier_fallback = None
        def bounded(size):
            if size>limit:raise MemoryError('Kernel surface frontier exceeds budget; no pairs silently dropped')
        if self.hop_neighbors is not None and device.type == 'cuda' and not reference and validation_candidates is None:
            bounded(n*self.hop_neighbors.shape[1])
            return self._select_gpu(source_faces, pool_cap, dispatch_geometry,
                max_records, max_dispatch_bytes, profile)
        if (n and f < 2**31 and device.type == 'cuda' and not reference and
                validation_candidates is None and self.hop_neighbors is None):
            # This is per-forward scratch, not cached support or a source-count
            # limit. Any overflowing row falls back to the complete tensor BFS.
            capacity = 1024
            scratch_budget = min(max_dispatch_bytes//4, 256*1024**2)
            while capacity <= 4096 and n*capacity*3*4+n*4 <= scratch_budget:
                with _scope(profile, 'kernel_support_discovery'):
                    neighbors = torch.full((n, capacity), -1, device=device, dtype=torch.int32)
                    seen = torch.full((n, capacity*2), -1, device=device, dtype=torch.int32)
                    overflow = torch.empty(n, device=device, dtype=torch.int32)
                    resident_gpu.graph_surface_frontier[(n,)](
                        self.adjacency, source_faces, seen, neighbors, overflow,
                        self.adjacency.shape[1], capacity, capacity*2, self.hops, 64,
                        num_warps=2)
                result = self._select_gpu(source_faces, pool_cap, dispatch_geometry,
                    max_records, max_dispatch_bytes, profile, neighbors=neighbors, overflow=overflow)
                if result is not None:
                    result[1]['frontier_capacity'] = capacity
                    return result
                frontier_fallback = 'row_overflow'
                # A few wide neighborhoods must not force every source back
                # through repeated global tensor sorts. Retry exact traversal
                # within a bounded scratch budget; never truncate overflowing
                # rows. Geometry and learned state are still refreshed.
                self._runtime_neighbors = None
                del neighbors, seen, overflow
                capacity *= 2
            if frontier_fallback is None:frontier_fallback = 'scratch_budget'
        with _scope(profile,'kernel_support_discovery'):
            visited=torch.arange(n,device=device,dtype=torch.long)*f+source_faces.long()
            frontier=visited
            bounded(len(visited))
            if validation_candidates is not None:
                # Diagnostic only: one query per face, so its native candidate
                # list is exact for that query. No union from other face pixels.
                faces,ids=validation_candidates
                if (faces.ndim!=1 or ids.ndim!=2 or len(faces)!=len(ids) or
                        faces.device!=device or ids.device!=device or
                        faces.dtype not in (torch.int32,torch.int64) or
                        ids.dtype not in (torch.int32,torch.int64) or
                        len(faces.unique())!=len(faces)):
                    raise ValueError('Replay requires one native candidate list per distinct face')
                bounded(ids.numel())
                torch._assert_async(((ids>=0)&(ids<n)).all()&((faces>=0)&(faces<f)).all(),
                    'Invalid native replay candidates')
                visited=(ids.long()*f+faces[:,None]).flatten().unique(sorted=True)
            elif self.hop_neighbors is not None:
                bounded(n*self.hop_neighbors.shape[1])
                neighbors=self.hop_neighbors[source_faces.long()]
                visited=(torch.arange(n,device=device,dtype=torch.long)[:,None]*f+neighbors)[neighbors>=0]
            for _ in range(0 if validation_candidates is not None or self.hop_neighbors is not None else self.hops):
                if not len(frontier):break
                bounded(len(frontier)*self.adjacency.shape[1]+len(visited))
                neighbors=self.adjacency[frontier%f]
                expanded=(frontier[:,None]//f*f+neighbors)[neighbors>=0].unique(sorted=True)
                where=torch.searchsorted(visited,expanded)
                fresh=(where==len(visited))|(visited[where.clamp(max=max(0,len(visited)-1))]!=expanded)
                frontier=expanded[fresh]
                visited=torch.cat((visited,frontier)).sort().values
                bounded(len(visited))
            # Source-major sparse support for this forward only.
            self.source_offsets=torch.cat((visited.new_zeros(1),torch.bincount(visited//f,minlength=n).cumsum(0)))
            self.support_faces=visited%f
        with _scope(profile,'support_visibility_intersection'):
            live=dispatch_geometry.pixel_counts_device[self.support_faces]>0
            keys=(self.support_faces[live]*n+visited[live]//f).sort().values
            entry_faces=keys//max(1,n);kernels=keys%max(1,n)
            counts=torch.bincount(entry_faces,minlength=f)
            offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
            metadata=torch.stack((counts.amax(),(counts*dispatch_geometry.face_counts).sum(),
                                  (counts*dispatch_geometry.pixel_counts_device).sum())).cpu().tolist()
            maximum,tasks,fragments=metadata
            if pool_cap and maximum>pool_cap:
                raise RuntimeError('Kernel surface support exceeds --support-cap; increase it (no silent truncation)')
            check_sizes(len(keys),maximum,tasks,fragments,max_records,max_dispatch_bytes,maximum_limit=None)
            mapping=DeviceSupport(counts,offsets,kernels.int(),entry_faces.int(),n,maximum,tasks,fragments,self,dispatch_geometry)
        return mapping,dict(backend='kernel_centric' if validation_candidates is None else 'native_candidates_validation_only',
            approximate=True,support_hops=self.hops,
            dynamic_topology_policy='fresh_kernel_graph_each_forward',source_count=n,
            fixed_hop_lookup=self.hop_neighbors is not None,
            fixed_hop_bytes=0 if self.hop_neighbors is None else self.hop_neighbors.numel()*self.hop_neighbors.element_size(),
            discovered_pairs=len(visited),retained_pairs=len(keys),maximum_face_support=maximum,
            support_engine='tensor_frontier', frontier_fallback=frontier_fallback,
            influence_bound='Fixed edge-hop truncation; no spectral conservativeness claim')

    def _select_gpu(self, source_faces, pool_cap, dispatch_geometry,
                    max_records, max_dispatch_bytes, profile, *, neighbors=None, overflow=None):
        """Same graph support as the tensor oracle, with one allocation readback."""
        source_indexed = neighbors is not None
        table = neighbors if source_indexed else self.hop_neighbors
        n, f, width = len(source_faces), len(self.faces), table.shape[1]
        device = self.faces.device
        with _scope(profile, 'kernel_support_discovery'):
            self._source_faces_snapshot = source_faces.clone()
            self._support_faces = None
            self._runtime_neighbors = neighbors
            source_counts = torch.empty(n, device=device, dtype=torch.int64)
            visible_counts = torch.empty_like(source_counts)
            counts = torch.zeros(f, device=device, dtype=torch.int64)
            if n:
                resident_gpu.count_kernel_surface_support[(n,)](table,
                    self._source_faces_snapshot, dispatch_geometry.pixel_counts_device,
                    source_counts, visible_counts, counts, width, triton.next_power_of_2(width),
                    SOURCE_INDEXED=source_indexed, num_warps=4)
            self.source_offsets = torch.cat((source_counts.new_zeros(1), source_counts.cumsum(0)))
            visible_offsets = torch.cat((visible_counts.new_zeros(1), visible_counts.cumsum(0)))
        with _scope(profile, 'support_visibility_intersection'):
            values = [visible_offsets[-1], counts.amax(),
                (counts*dispatch_geometry.face_counts).sum(),
                (counts*dispatch_geometry.pixel_counts_device).sum(), self.source_offsets[-1]]
            if overflow is not None:values.append(overflow.amax().long())
            metadata = torch.stack(values).cpu().tolist()
            if overflow is not None and metadata[-1]:return None
            pairs, maximum, tasks, fragments, discovered = metadata[:5]
            if discovered > min(max_records, max_dispatch_bytes//64):
                raise MemoryError('Kernel surface frontier exceeds budget; no pairs silently dropped')
            if pool_cap and maximum > pool_cap:
                raise RuntimeError('Kernel surface support exceeds --support-cap; increase it (no silent truncation)')
            check_sizes(pairs, maximum, tasks, fragments, max_records, max_dispatch_bytes, maximum_limit=None)
            keys = torch.empty(pairs, device=device, dtype=torch.int64)
            if pairs:
                resident_gpu.pack_kernel_surface_support[(n,)](table,
                    self._source_faces_snapshot, dispatch_geometry.pixel_counts_device,
                    visible_offsets, keys, n, width, triton.next_power_of_2(width),
                    SOURCE_INDEXED=source_indexed, num_warps=4)
                keys = keys.sort().values
            entry_faces, kernels = keys//max(1, n), keys % max(1, n)
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            mapping = DeviceSupport(counts, offsets, kernels.int(), entry_faces.int(), n,
                maximum, tasks, fragments, self, dispatch_geometry)
        return mapping, dict(backend='kernel_centric', approximate=True, support_hops=self.hops,
            dynamic_topology_policy='fresh_kernel_graph_each_forward', source_count=n,
            fixed_hop_lookup=not source_indexed,
            fixed_hop_bytes=0 if source_indexed else table.numel()*table.element_size(),
            discovered_pairs=discovered, retained_pairs=pairs, maximum_face_support=maximum,
            support_engine='fused_row_frontier' if source_indexed else 'fused_graph_visibility',
            influence_bound='Fixed edge-hop truncation; no spectral conservativeness claim')

def _scope(enabled, name):
    if hasattr(enabled, 'stage'): return enabled.stage('selector::'+name)
    return torch.profiler.record_function('support::resident_'+name) if enabled else nullcontext()


def check_sizes(pairs, maximum, tasks, fragments, max_records, max_dispatch_bytes, *, maximum_limit=512):
    """Host integers only; check before compact allocations or narrowing IDs."""
    if min(pairs, maximum, tasks, fragments) < 0:
        raise ValueError('Negative resident allocation size')
    if maximum_limit is not None and maximum > maximum_limit:
        raise ValueError('Uncapped support exceeds existing merge width; use pool_cap <=512')
    if max(pairs, tasks, fragments) >= 2**31:
        raise ValueError('Resident support/work requires int32-indexable sizes')
    if max_records < 1 or fragments > max_records:
        raise ValueError('Resident path requires one bounded record batch; increase --max-records')
    # Logical compact support/task/bin footprint; sort workspace is additional.
    if pairs*52+tasks*40 > max_dispatch_bytes:
        raise MemoryError('Resident support/dispatch exceeds configured byte bound')


@dataclass(frozen=True)
class DeviceSupport:
    """Own the current forward's ordered kernel/triangle CSR and allocation metadata."""
    counts: torch.Tensor
    offsets: torch.Tensor
    kernels: torch.Tensor
    entry_faces: torch.Tensor
    source_count: int
    maximum: int
    tasks: int
    fragments: int
    geometry: object
    dispatch_geometry: object

    def __post_init__(self):
        tensors = (self.counts, self.offsets, self.kernels, self.entry_faces)
        object.__setattr__(self, '_versions', tuple(t._version for t in tensors))

    def valid(self):
        return (tuple(t._version for t in (self.counts, self.offsets, self.kernels, self.entry_faces)) == self._versions
                and self.geometry.valid() and self.dispatch_geometry.valid())


class ResidentSupportGeometry(PreparedSupport):
    """Immutable float64 bounds and visible IDs, without support staging reuse."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.face_count >= 2**31 or self.face_ids.size >= 2**31:
            raise ValueError('Resident geometry requires int32-indexable faces and pixels')
        self.face_ids.flags.writeable = False
        self.visible.flags.writeable = False
        self.visible_device = torch.tensor(self.visible, device=self.device, dtype=torch.int64)
        self._fixed = (self.faces, self.embedding, self.visible_device,
                       *(t for _, c, r in self.chunks for t in (c, r)),
                       *(t for _, c, r in (self.execution_chunks if self.execution_chunks is not self.chunks else ()) for t in (c, r)))
        self._fixed_versions = tuple(t._version for t in self._fixed)

    def valid(self):
        current = (self.faces, self.embedding, self.visible_device,
                   *(t for _, c, r in self.chunks for t in (c, r)),
                   *(t for _, c, r in (self.execution_chunks if self.execution_chunks is not self.chunks else ()) for t in (c, r)))
        return (len(current) == len(self._fixed) and
                all(a is b for a, b in zip(current, self._fixed)) and
                tuple(t._version for t in current) == self._fixed_versions)

    @torch.no_grad()
    def select(self, sources, pool_cap=128, guard=1e-5, *, dispatch_geometry,
               max_records=1048576, max_dispatch_bytes=512*1024**2, debug_callback=None):
        if not self.valid() or not dispatch_geometry.valid():
            raise ValueError('Resident geometry changed; rebuild the snapshot')
        fixed = dispatch_geometry.geometry
        if (fixed.face_count != self.face_count or fixed.bary.device != self.device or
                not (self.face_ids is fixed._ids if getattr(self, 'device_visibility', False) else np.array_equal(self.face_ids, fixed._ids))):
            raise ValueError('Support and dispatch geometry differ')
        if max_records < fixed.tile_size**2:
            raise ValueError('Record bound must fit one full tile')
        if not 0 <= pool_cap <= 512 or guard < 0:
            raise ValueError('Invalid support limits')
        if sources.ndim != 2 or sources.shape[1] != self.dim or sources.device != self.device:
            raise ValueError('Source embeddings differ from prepared geometry')
        n = len(sources)
        if n >= 2**31 or (len(self.visible) and not n):
            raise ValueError('Expected a nonempty int32-indexable source table')
        with _scope(self.profile, 'candidate_selection_float64'):
            rows = self.candidate_rows(sources, pool_cap, guard)
        with _scope(self.profile, 'count_and_scan'):
            counts = torch.zeros(self.face_count, dtype=torch.int64, device=self.device)
            counts[self.visible_device] = rows[:, 0].long()
            offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            maximum = counts.amax() if self.face_count else counts.new_zeros(())
            tasks = (counts*dispatch_geometry.face_counts).sum()
            fragments = (counts*dispatch_geometry.pixel_counts_device).sum()
            metadata = torch.stack((offsets[-1], maximum, tasks, fragments))
        with _scope(self.profile, 'allocation_metadata_read'):
            pairs, maximum, tasks, fragments = metadata.cpu().tolist()
        check_sizes(pairs, maximum, tasks, fragments, max_records, max_dispatch_bytes)
        with _scope(self.profile, 'compact_csr'):
            kernels = torch.empty(pairs, device=self.device, dtype=torch.int32)
            faces = torch.empty_like(kernels)
            if len(self.visible) and pairs:
                resident_gpu.pack_support[(len(self.visible),)](
                    rows, self.visible_device, offsets, kernels, faces,
                    rows.shape[1], triton.next_power_of_2(rows.shape[1]-1), num_warps=4)
        stats = dict(backend='resident', dynamic_topology_policy='fresh_each_forward',
            dynamic_cache_hits=0, support_id_d2h_bytes=0, dynamic_h2d_bytes=0,
            explicit_d2h_calls=int(self.device.type == 'cuda'),
            transfer_bytes=32 if self.device.type == 'cuda' else 0,
            allocation_metadata_bytes=32, allocation_syncs=int(self.device.type == 'cuda'),
            candidate_chunks=len(self.chunks), retained_pairs=pairs,
            geometry_reused=self.calls > 0, dispatch_reused=False,
            coefficient_topology_reused=False,
            distance_layout=getattr(self, 'distance_layout', 'native_cdist'))
        self.calls += 1
        mapping = DeviceSupport(counts, offsets, kernels, faces, n, maximum, tasks, fragments,
                                self, dispatch_geometry)
        if debug_callback is not None:
            debug_callback(rows, mapping)
        return mapping, stats


class StableSupportReference(ResidentSupportGeometry):
    """Independent Torch oracle for the wide selector's distance/source-ID ties.

    Used only by correctness diagnostics; legacy selection stays unchanged.
    """
    @torch.no_grad()
    def candidate_rows(self, sources, pool_cap=128, guard=1e-5):
        n = len(sources)
        if not 0 < pool_cap < n:
            raise ValueError('Stable support reference requires a capped source table')
        rows = torch.full((len(self.visible), pool_cap+1), n, dtype=torch.int32, device=self.device)
        for begin, center, radius in self.chunks:
            distance = torch.cdist(center, sources.double(), compute_mode='donot_use_mm_for_euclid_dist')
            # Stable sorting starts from ascending source IDs. NaNs sort last.
            order = torch.argsort(distance, dim=1, stable=True)
            ranked = distance.gather(1, order)
            kth = ranked[:, min(self.neighbors, n)-1]
            bound = kth+2*radius+guard*(1+kth+radius)
            selected = ranked[:, :pool_cap] <= bound[:, None]
            ordered = torch.where(selected, order[:, :pool_cap], n).sort(dim=1).values
            rows[begin:begin+len(center), 0] = selected.sum(1)
            rows[begin:begin+len(center), 1:] = ordered
        return rows


class FusedResidentSupportGeometry(ResidentSupportGeometry):
    """Tiled CUDA distances, bounded selection, and integer CSR scans.

    Small tables keep the audited ATen selection policy. Wide tables merge
    bounded source tiles with explicit distance/source-ID ties. CPU distance
    controls retain native cdist; the small-table control also repairs CPU topk
    ties, without adding CUDA readbacks.
    """
    TORCH_CUDA_REVISION = '449b1768410104d3ed79d3bcfe4ba1d65c7f22c0'
    SOURCE_TILE_SIZE = 1024

    def __init__(self, *args, merge_backend='reference', row_scratch_bytes=32*1024**2, pruning='none', **kwargs):
        if pruning not in ('none','prefix'):raise ValueError('Unknown support pruning policy')
        self.pruning=pruning
        if merge_backend not in ('reference', 'compact', 'radix', 'row-radix', 'row-topk'): raise ValueError('Unknown tiled merge backend')
        if row_scratch_bytes < 1: raise ValueError('Row scratch budget must be positive')
        self.merge_backend = merge_backend
        self.row_scratch_bytes = row_scratch_bytes
        super().__init__(*args, **kwargs)
        face_rows = np.full(self.face_count, -1, np.int32)
        face_rows[self.visible] = np.arange(len(self.visible), dtype=np.int32)
        self.face_rows = torch.tensor(face_rows, device=self.device)
        self._face_rows, self._face_rows_version = self.face_rows, self.face_rows._version

    def valid(self):
        return (super().valid() and self.face_rows is self._face_rows and
                self.face_rows._version == self._face_rows_version)

    @staticmethod
    def single_block_topk(rows, sources):
        # ATen's exact dispatch predicate at TORCH_CUDA_REVISION.
        return not ((rows <= 20 and sources >= 20000) or
            (20 < rows <= 40 and sources >= 10000) or
            (40 < rows <= 80 and sources >= 8000) or
            (80 < rows < 200 and sources >= 5000) or
            (200 <= rows < 800 and sources >= 3000) or
            (800 <= rows <= 4000 and sources >= 800) or
            (rows > 4000 and sources >= 400))

    @torch.no_grad()
    def candidate_rows(self, sources, pool_cap=128, guard=1e-5):
        """Rebuild reference spectral candidates; no persistent association state."""
        n = len(sources)
        self.fused_chunks = 0
        self.pruning_used=False
        self.pruning_reason='unsupported_shape_or_policy' if self.pruning!='none' else None
        self.row_radix_fallback = None
        self.distance_layout = 'cpu_cdist' if sources.device.type == 'cpu' else 'source8'
        chunks = self.execution_chunks if sources.device.type == 'cuda' else self.chunks
        self.fallback_reason = None
        self.native_distance_fallback = self.native_topk_fallback = None
        self.selection_algorithm = 'row_sort'
        self.selection_tie_policy = 'native_single_block'
        if not 0 < pool_cap < n:
            self.fallback_reason = 'uncapped_or_cap_covers_sources'
        elif n > 4096 and min(self.neighbors, n) > 512:
            self.fallback_reason = 'neighbor_rank_exceeds_tiled_capacity'
        elif not 1 <= self.dim <= 256 or sources.dtype not in (torch.float32, torch.float64):
            self.fallback_reason = 'distance_shape_or_dtype_not_audited'
        elif sources.device.type == 'cuda' and (torch.version.hip is not None or
                torch.version.git_version != self.TORCH_CUDA_REVISION):
            self.fallback_reason = 'torch_cuda_revision_not_audited'
        elif n <= 4096 and sources.device.type == 'cuda' and any(not self.single_block_topk(len(c), n) for _, c, _ in self.chunks):
            self.fallback_reason = 'native_multiblock_topk_policy'
        elif (len(self.visible)*(pool_cap+1) >= 2**31 or
              any(len(c)*n >= 2**31 or (len(c)-1)*abs(c.stride(0))+
                  (self.dim-1)*abs(c.stride(1)) >= 2**31 for _, c, _ in chunks) or
              (n-1)*abs(sources.stride(0))+(self.dim-1)*abs(sources.stride(1)) >= 2**31):
            # The tiled programs use int32 relative indexing. Preserve native
            # 64-bit indexing for unusually large or sparse-strided views.
            self.fallback_reason = 'index_span_exceeds_int32'
        if self.fallback_reason:
            self.distance_layout = 'native_cdist'
            self.selection_algorithm = 'legacy'
            self.selection_tie_policy = 'native_topk'
            return super().candidate_rows(sources, pool_cap, guard)
        if n > 4096:
            if self.merge_backend in ('row-radix','row-topk'):
                if n <= 8192 and n*8+(max(min(self.neighbors,n),pool_cap)*16 if self.merge_backend=='row-topk' else 0) <= self.row_scratch_bytes:
                    return self._row_radix_candidate_rows(sources, pool_cap, guard)
                self.row_radix_fallback = ('source_count_exceeds_row_capacity' if n > 8192 else
                                           'row_distance_budget_too_small')
            return self._tiled_candidate_rows(sources, pool_cap, guard)
        rows = torch.empty((len(self.visible), pool_cap+1), device=self.device, dtype=torch.int32)
        cpu = sources.device.type == 'cpu'
        scratch = None
        if self.chunks and not cpu:
            scratch = torch.empty((max(len(c) for _, c, _ in chunks), n),
                                  device=self.device, dtype=torch.float64)
        if cpu:
            self.native_distance_fallback = 'cpu_cdist_reduction_order'
            self.native_topk_fallback = 'cpu_topk_cutoff_tie_repair'
            self.selection_tie_policy = 'cpu_native_topk'
        for begin, center, radius in chunks:
            with _scope(self.profile, 'fused_distance'):
                if cpu:
                    distance = torch.cdist(center, sources.double(), compute_mode='donot_use_mm_for_euclid_dist')
                else:
                    distance = scratch[:len(center)]
                    resident_gpu.support_distances[(triton.cdiv(n, 8), len(center))](
                        center, sources, distance, n, *center.stride(), *sources.stride(), self.dim,
                        max(32, triton.next_power_of_2(self.dim)), 8, True,
                        num_warps=4, enable_fp_fusion=False)
            kth = distance.new_empty(len(center)) if cpu else distance
            ties = rows.new_empty(len(center)) if cpu else rows
            with _scope(self.profile, 'fused_threshold_topk_mask_output'):
                resident_gpu.select_support_rows[(len(center),)](
                    distance, radius, rows, kth, ties, n, min(self.neighbors, n), pool_cap, begin, pool_cap+1,
                    guard, triton.next_power_of_2(n), cpu, num_warps=4, enable_fp_fusion=False)
            if cpu:
                # CPU-native tie identities differ from CUDA sbtopk. Only tied
                # rows are repaired; no Tensor read or CPU fallback runs on CUDA.
                bound = kth[:, None]+2*radius[:, None]+guard*(1+kth[:, None]+radius[:, None])
                nearest = distance.masked_fill(~(distance <= bound), float('inf')).topk(
                    pool_cap, dim=1, largest=False).indices
                resident_gpu.pack_support_rows[(len(center),)](
                    distance, kth, radius, nearest, rows, n, pool_cap, begin, pool_cap+1,
                    guard, triton.next_power_of_2(pool_cap), REPAIR=ties, REPAIR_ONLY=True,
                    num_warps=4, enable_fp_fusion=False)
            self.fused_chunks += 1
        return rows

    @torch.no_grad()
    def _row_radix_candidate_rows(self, sources, pool_cap, guard):
        """Bounded whole-row exact cutoffs; wide inputs retain the tiled path."""
        n = len(sources)
        rows = torch.empty((len(self.visible), pool_cap+1), device=self.device, dtype=torch.int32)
        self.selection_algorithm = 'row_topk' if self.merge_backend=='row-topk' else 'row_radix'
        self.selection_tie_policy = 'distance_then_source_id'
        cpu = self.device.type == 'cpu'
        chunks = self.chunks if cpu else self.execution_chunks
        if cpu: self.native_distance_fallback = 'cpu_cdist_reduction_order'
        if not chunks: return rows
        eligible=(not cpu and self.dim==64 and sources.dtype==torch.float32 and self.pruning=='prefix')
        topk_bytes=max(min(self.neighbors,n),pool_cap)*16 if self.merge_backend=='row-topk' else 0
        seed_bytes=(min(self.neighbors,n)+1)*8
        prune_enabled=eligible and n*8+seed_bytes+topk_bytes<=self.row_scratch_bytes
        if eligible and not prune_enabled:self.pruning_reason='seed_scratch_budget'
        row_bytes=n*8+topk_bytes+(seed_bytes if prune_enabled else 0)
        batch = min(max(len(c) for _,c,_ in chunks), self.row_scratch_bytes//row_bytes)
        scratch = torch.empty((batch,n), device=self.device, dtype=torch.float64)
        # More independent pairs amortize CTA launch/center loads at the common
        # 64-D embedding. Keep the old width for larger register footprints.
        pairs = 32 if self.dim <= 64 else 8
        self.distance_layout = ('cpu_cdist' if cpu else 'face4_source16'
                                if self.dim == 64 and sources.dtype == torch.float32 else 'source'+str(pairs))
        for begin, centers, radii in chunks:
            for offset in range(0,len(centers),batch):
                center = centers[offset:offset+batch]
                radius = radii[offset:offset+batch]
                with _scope(self.profile, 'row_distance'):
                    if cpu:
                        distance = torch.cdist(center,sources.double(),compute_mode='donot_use_mm_for_euclid_dist')
                    else:
                        distance = scratch[:len(center)]
                        if self.dim == 64 and sources.dtype == torch.float32:
                            upper=None
                            prune=prune_enabled
                            if prune:
                                rank=min(self.neighbors,n)
                                seed=torch.empty((len(center),rank),device=self.device,dtype=torch.float64)
                                upper=torch.empty(len(center),device=self.device,dtype=torch.float64)
                                resident_gpu.support_distances[(triton.cdiv(rank,8),len(center))](
                                    center,sources,seed,rank,*center.stride(),*sources.stride(),64,64,8,True,
                                    num_warps=4,enable_fp_fusion=False)
                                resident_gpu.support_seed_upper[(len(center),)](seed,radius,upper,rank,guard,
                                    triton.next_power_of_2(rank),num_warps=4,enable_fp_fusion=False)
                                self.pruning_used=True
                                self.pruning_reason=None
                            resident_gpu.support_distances_blocked[(triton.cdiv(n,16),triton.cdiv(len(center),4))](
                                center,sources,distance,len(center),n,*center.stride(),*sources.stride(),
                                64,64,4,16,True,UPPER=upper,PRUNE=prune,
                                num_warps=4,enable_fp_fusion=False)
                        else:
                            resident_gpu.support_distances[(triton.cdiv(n,pairs),len(center))](
                                center,sources,distance,n,*center.stride(),*sources.stride(),self.dim,
                                max(32,triton.next_power_of_2(self.dim)),pairs,True,
                                num_warps=4,enable_fp_fusion=False)
                with _scope(self.profile, 'row_cutoffs_and_output'):
                    if self.merge_backend=='row-topk':
                        rank=min(self.neighbors,n)
                        values=distance.topk(max(rank,pool_cap),dim=1,largest=False,sorted=True).values
                        kth,cutoff=values[:,rank-1],values[:,pool_cap-1]
                        resident_gpu.pack_support_cutoffs[(len(center),)](
                            distance,radius,kth,cutoff,rows,n,pool_cap,begin+offset,pool_cap+1,
                            kth.stride(0),cutoff.stride(0),guard,triton.next_power_of_2(n),
                            num_warps=8,enable_fp_fusion=False)
                    else:
                        resident_gpu.select_support_row_radix[(len(center),)](
                            distance,radius,rows,n,min(self.neighbors,n),pool_cap,begin+offset,pool_cap+1,
                            guard,triton.next_power_of_2(n),num_warps=8,enable_fp_fusion=False)
                self.fused_chunks += 1
        return rows

    @torch.no_grad()
    def _tiled_candidate_rows(self, sources, pool_cap, guard):
        """Stream source tiles through bounded GPU lists, without an M-by-N table.

        Every source is examined. Exact ties use ascending source IDs; the
        final support rows keep the existing ascending-ID CSR contract.
        """
        n, tile = len(sources), self.SOURCE_TILE_SIZE
        keep = triton.next_power_of_2(max(min(self.neighbors, n), pool_cap))
        block = triton.next_power_of_2(tile+keep)
        rows = torch.empty((len(self.visible), pool_cap+1), device=self.device, dtype=torch.int32)
        self.selection_algorithm = {'reference':'tiled_merge', 'compact':'tiled_compact_merge',
                                    'radix':'tiled_radix_merge', 'row-radix':'tiled_radix_merge','row-topk':'tiled_radix_merge'}[self.merge_backend]
        self.selection_tie_policy = 'distance_then_source_id'
        cpu = self.device.type == 'cpu'
        chunks = self.chunks if cpu else self.execution_chunks
        if cpu:
            self.native_distance_fallback = 'cpu_cdist_reduction_order'
        if not self.chunks:
            return rows
        face_rows = max(len(c) for _, c, _ in chunks)
        scratch = torch.empty(face_rows*tile, device=self.device, dtype=torch.float64)
        compact = self.merge_backend in ('compact', 'radix', 'row-radix','row-topk')
        key_scratch = torch.empty((face_rows, keep), device=self.device, dtype=torch.int64) if compact else None
        id_scratch = torch.empty((face_rows, keep), device=self.device, dtype=torch.int32) if compact else None
        local_keys = torch.empty_like(key_scratch) if self.merge_backend in ('radix', 'row-radix','row-topk') else None
        local_ids = torch.empty_like(id_scratch) if self.merge_backend in ('radix', 'row-radix','row-topk') else None
        for begin, center, radius in chunks:
            if compact:
                # Private to this selection call; first tile rewrites every lane.
                # finish_support_tiles consumes the lists on the same stream
                # before the next face batch reuses their storage.
                keys = key_scratch[:len(center)]
                ids = id_scratch[:len(center)]
            else:
                keys = torch.full((len(center), keep), -1, device=self.device, dtype=torch.int64)
                ids = torch.full((len(center), keep), n, device=self.device, dtype=torch.int32)
            for start in range(0, n, tile):
                source = sources[start:start+tile]
                width = len(source)
                with _scope(self.profile, 'fused_distance'):
                    if cpu:
                        distance = torch.cdist(center, source.double(), compute_mode='donot_use_mm_for_euclid_dist')
                    else:
                        distance = scratch[:len(center)*width].view(len(center), width)
                        resident_gpu.support_distances[(triton.cdiv(width, 8), len(center))](
                            center, source, distance, width, *center.stride(), *source.stride(), self.dim,
                            max(32, triton.next_power_of_2(self.dim)), 8, True,
                            num_warps=4, enable_fp_fusion=False)
                with _scope(self.profile, 'fused_tile_merge'):
                    if self.merge_backend in ('radix', 'row-radix','row-topk'):
                        resident_gpu.merge_support_tile_radix[(len(center),)](
                            distance, local_keys, local_ids, keys, ids, n, width, start, tile, keep,
                            keep.bit_length()-1, (2*keep).bit_length()-1,
                            num_warps=4, enable_fp_fusion=False)
                    elif compact:
                        resident_gpu.merge_support_tile_compact[(len(center),)](
                            distance, keys, ids, n, width, start, tile, keep,
                            tile.bit_length()-1, (2*keep).bit_length()-1,
                            num_warps=4, enable_fp_fusion=False)
                    else:
                        resident_gpu.merge_support_tile[(len(center),)](
                            distance, keys, ids, n, width, start, tile, keep, block, block.bit_length()-1,
                            num_warps=4, enable_fp_fusion=False)
            with _scope(self.profile, 'fused_threshold_mask_output'):
                resident_gpu.finish_support_tiles[(len(center),)](
                    keys, ids, radius, rows, n, min(self.neighbors, n), pool_cap, begin, pool_cap+1,
                    guard, keep, num_warps=4, enable_fp_fusion=False)
            self.fused_chunks += 1
        return rows

    @torch.no_grad()
    def select(self, sources, pool_cap=128, guard=1e-5, *, dispatch_geometry,
               max_records=1048576, max_dispatch_bytes=512*1024**2, debug_callback=None):
        if not self.valid() or not dispatch_geometry.valid():
            raise ValueError('Resident geometry changed; rebuild the snapshot')
        fixed = dispatch_geometry.geometry
        if (fixed.face_count != self.face_count or fixed.bary.device != self.device or
                not (self.face_ids is fixed._ids if getattr(self, 'device_visibility', False) else np.array_equal(self.face_ids, fixed._ids))):
            raise ValueError('Support and dispatch geometry differ')
        if max_records < fixed.tile_size**2:
            raise ValueError('Record bound must fit one full tile')
        if not 0 <= pool_cap <= 512 or guard < 0:
            raise ValueError('Invalid support limits')
        if sources.ndim != 2 or sources.shape[1] != self.dim or sources.device != self.device:
            raise ValueError('Source embeddings differ from prepared geometry')
        n = len(sources)
        if n >= 2**31 or (len(self.visible) and not n):
            raise ValueError('Expected a nonempty int32-indexable source table')
        if self.face_count > 1048576:
            # This scan bound depends on face count, not total source count.
            mapping, stats = super().select(sources, pool_cap, guard, dispatch_geometry=dispatch_geometry,
                max_records=max_records, max_dispatch_bytes=max_dispatch_bytes, debug_callback=debug_callback)
            stats['metadata_backend'] = 'legacy_large_face_table'
        else:
            with _scope(self.profile, 'candidate_selection_float64'):
                rows = self.candidate_rows(sources, pool_cap, guard)
            with _scope(self.profile, 'count_and_scan'):
                counts = torch.empty(self.face_count, device=self.device, dtype=torch.int64)
                offsets = torch.empty(self.face_count+1, device=self.device, dtype=torch.int64)
                blocks = max(1, triton.cdiv(self.face_count, 256))
                partial = torch.empty((blocks, 4), device=self.device, dtype=torch.int64)
                starts = torch.empty(blocks, device=self.device, dtype=torch.int64)
                metadata = torch.empty(4, device=self.device, dtype=torch.int64)
                resident_gpu.support_scan_blocks[(blocks,)](
                    rows, self.face_rows, dispatch_geometry.face_counts, dispatch_geometry.pixel_counts_device,
                    counts, offsets, partial, self.face_count, rows.shape[1], 256, num_warps=4)
                resident_gpu.support_scan_totals[(1,)](
                    partial, starts, metadata, blocks, triton.next_power_of_2(blocks), num_warps=4)
                resident_gpu.support_scan_offsets[(blocks,)](offsets, starts, self.face_count, 256, num_warps=4)
            with _scope(self.profile, 'allocation_metadata_read'):
                pairs, maximum, tasks, fragments = metadata.cpu().tolist()
            check_sizes(pairs, maximum, tasks, fragments, max_records, max_dispatch_bytes)
            with _scope(self.profile, 'compact_csr'):
                kernels = torch.empty(pairs, device=self.device, dtype=torch.int32)
                faces = torch.empty_like(kernels)
                if len(self.visible) and pairs:
                    resident_gpu.pack_support[(len(self.visible),)](rows, self.visible_device, offsets, kernels, faces,
                        rows.shape[1], triton.next_power_of_2(rows.shape[1]-1), num_warps=4)
            stats = dict(backend='resident', dynamic_topology_policy='fresh_each_forward',
                dynamic_cache_hits=0, support_id_d2h_bytes=0, dynamic_h2d_bytes=0,
                explicit_d2h_calls=int(self.device.type == 'cuda'),
                transfer_bytes=32 if self.device.type == 'cuda' else 0, allocation_metadata_bytes=32,
                allocation_syncs=int(self.device.type == 'cuda'), candidate_chunks=len(self.chunks),
                retained_pairs=pairs, geometry_reused=self.calls > 0, dispatch_reused=False,
                coefficient_topology_reused=False, metadata_backend='triton_block_scan')
            mapping = DeviceSupport(counts, offsets, kernels, faces, n, maximum, tasks, fragments, self, dispatch_geometry)
            self.calls += 1
            if debug_callback is not None:
                debug_callback(rows, mapping)
        stats.update(support_selector='fused', fused_chunks=self.fused_chunks,
                     source_count=n,distance_layout=self.distance_layout,row_radix_fallback=self.row_radix_fallback,
                     pruning_requested=self.pruning, pruning_used=self.pruning_used,
                     pruning_unused_reason=self.pruning_reason,
                     selector_fallback=self.fallback_reason, native_distance_fallback=self.native_distance_fallback,
                     native_topk_fallback=self.native_topk_fallback,
                     selection_algorithm=self.selection_algorithm, selection_tie_policy=self.selection_tie_policy,
                     source_tile_size=self.SOURCE_TILE_SIZE if self.selection_algorithm == 'tiled_merge' else None)
        return mapping, stats

