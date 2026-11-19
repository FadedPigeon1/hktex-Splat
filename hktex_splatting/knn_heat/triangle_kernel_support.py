"""Fresh triangle candidate selection and device CSR, including fused selector fallback.

Candidate selection and ordered support packing for HKS and its internal triangle reference.
Resident* names are retained identifiers for this shared support machinery.
Main API: ResidentSupportGeometry, FusedResidentSupportGeometry, DeviceSupport."""
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
import triton
from hktex_splatting.knn_heat.packed_support import PreparedSupport
from hktex_splatting.triton import resident_gpu

def _scope(enabled, name):
    return torch.profiler.record_function('support::resident_'+name) if enabled else nullcontext()


def check_sizes(pairs, maximum, tasks, fragments, max_records, max_dispatch_bytes):
    """Host integers only; check before compact allocations or narrowing IDs."""
    if min(pairs, maximum, tasks, fragments) < 0:
        raise ValueError('Negative resident allocation size')
    if maximum > 512:
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
                       *(t for _, c, r in self.chunks for t in (c, r)))
        self._fixed_versions = tuple(t._version for t in self._fixed)

    def valid(self):
        current = (self.faces, self.embedding, self.visible_device,
                   *(t for _, c, r in self.chunks for t in (c, r)))
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
                not np.array_equal(self.face_ids, fixed._ids)):
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
            coefficient_topology_reused=False)
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

    def __init__(self, *args, **kwargs):
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
        n = len(sources)
        self.fused_chunks = 0
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
                  (self.dim-1)*abs(c.stride(1)) >= 2**31 for _, c, _ in self.chunks) or
              (n-1)*abs(sources.stride(0))+(self.dim-1)*abs(sources.stride(1)) >= 2**31):
            # The tiled programs use int32 relative indexing. Preserve native
            # 64-bit indexing for unusually large or sparse-strided views.
            self.fallback_reason = 'index_span_exceeds_int32'
        if self.fallback_reason:
            self.selection_algorithm = 'legacy'
            self.selection_tie_policy = 'native_topk'
            return super().candidate_rows(sources, pool_cap, guard)
        if n > 4096:
            return self._tiled_candidate_rows(sources, pool_cap, guard)
        rows = torch.empty((len(self.visible), pool_cap+1), device=self.device, dtype=torch.int32)
        cpu = sources.device.type == 'cpu'
        scratch = None
        if self.chunks and not cpu:
            scratch = torch.empty((max(len(c) for _, c, _ in self.chunks), n),
                                  device=self.device, dtype=torch.float64)
        if cpu:
            self.native_distance_fallback = 'cpu_cdist_reduction_order'
            self.native_topk_fallback = 'cpu_topk_cutoff_tie_repair'
            self.selection_tie_policy = 'cpu_native_topk'
        for begin, center, radius in self.chunks:
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
    def _tiled_candidate_rows(self, sources, pool_cap, guard):
        """Stream source tiles through bounded GPU lists, without an M-by-N table.

        Every source is examined. Exact ties use ascending source IDs; the
        final support rows keep the existing ascending-ID CSR contract.
        """
        n, tile = len(sources), self.SOURCE_TILE_SIZE
        keep = triton.next_power_of_2(max(min(self.neighbors, n), pool_cap))
        block = triton.next_power_of_2(tile+keep)
        rows = torch.empty((len(self.visible), pool_cap+1), device=self.device, dtype=torch.int32)
        self.selection_algorithm = 'tiled_merge'
        self.selection_tie_policy = 'distance_then_source_id'
        cpu = self.device.type == 'cpu'
        if cpu:
            self.native_distance_fallback = 'cpu_cdist_reduction_order'
        if not self.chunks:
            return rows
        face_rows = max(len(c) for _, c, _ in self.chunks)
        scratch = torch.empty(face_rows*tile, device=self.device, dtype=torch.float64)
        for begin, center, radius in self.chunks:
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
                not np.array_equal(self.face_ids, fixed._ids)):
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
                     selector_fallback=self.fallback_reason, native_distance_fallback=self.native_distance_fallback,
                     native_topk_fallback=self.native_topk_fallback,
                     selection_algorithm=self.selection_algorithm, selection_tie_policy=self.selection_tie_policy,
                     source_tile_size=self.SOURCE_TILE_SIZE if self.selection_algorithm == 'tiled_merge' else None)
        return mapping, stats

