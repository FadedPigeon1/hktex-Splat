"""Group supported triangle work and emit HKS pixel contributions with Triton.

GPU rendering uses three-corner coefficients and visible-pixel barycentrics.
Main API: expand_triangle_work, group_sorted_triangle_work, _hybrid_emit, hybrid_forward.
Task expansion preserves the reference order and changes no floating field values."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=['P', 'IMAGE', 'F'], do_not_specialize_on_alignment=['P', 'IMAGE', 'F'])
def prepare_camera_surface(IDS, BARY, PIXELS, FACE_IDS, POINT_BARY, ORDINALS,
                           TILES, TILE_LENGTHS, TILE_PIXELS, FACE_PIXELS, FLAGS,
                           P, IMAGE, F, WIDTH: tl.constexpr, TILES_X: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Pack fresh visible-pixel geometry and integer histograms, validating all input IDs."""
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    image_valid = i < IMAGE
    face = tl.load(IDS+i, image_valid, -1)
    bad_image = image_valid & ((face < -1) | (face >= F))
    count = tl.sum((image_valid & (face >= 0)).to(tl.int32), 0)
    tl.atomic_add(FLAGS+1, count)
    point_valid = i < P
    pixel = tl.load(PIXELS+i, point_valid, -1)
    previous = tl.load(PIXELS+i-1, point_valid & (i > 0), -1)
    in_bounds = (pixel >= 0) & (pixel < IMAGE)
    selected = tl.load(IDS+pixel, point_valid & in_bounds, -1)
    covered = (selected >= 0) & (selected < F)
    bad_pixel = point_valid & (~in_bounds | ~covered | ((i > 0) & (pixel <= previous)))
    good = point_valid & in_bounds & covered
    b0 = tl.load(BARY+pixel*3, good, 0.)
    b1 = tl.load(BARY+pixel*3+1, good, 0.)
    b2 = tl.load(BARY+pixel*3+2, good, 0.)
    finite = (b0 == b0) & (b1 == b1) & (b2 == b2)
    finite &= (tl.abs(b0) < float('inf')) & (tl.abs(b1) < float('inf')) & (tl.abs(b2) < float('inf'))
    bary_valid = finite & (b0 >= -2e-6) & (b1 >= -2e-6) & (b2 >= -2e-6)
    bary_valid &= tl.abs((b0+b2)+b1-1.) <= 2e-6
    bad = tl.sum((bad_image | bad_pixel | (good & ~bary_valid)).to(tl.int32), 0) > 0
    tl.atomic_or(FLAGS, bad.to(tl.int32))
    tl.store(FACE_IDS+i, tl.where(covered, selected, 0), point_valid)
    tl.store(POINT_BARY+i*3, b0, point_valid)
    tl.store(POINT_BARY+i*3+1, b1, point_valid)
    tl.store(POINT_BARY+i*3+2, b2, point_valid)
    tl.store(ORDINALS+i, i, point_valid)
    tile = pixel//WIDTH//8*TILES_X+pixel%WIDTH//8
    lane = (pixel//WIDTH % 8)*8+pixel % WIDTH % 8
    tl.store(TILES+i, tile, point_valid)
    tl.atomic_add(TILE_LENGTHS+tile, 1, good)
    tl.store(TILE_PIXELS+tile*64+lane, i, good)
    tl.atomic_add(FACE_PIXELS+selected, 1, good)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _triangle_group_scan(KEYS, LOCAL_GROUP, BLOCK_COUNTS, N, BLOCK: tl.constexpr):
    """Count sorted-key run starts and scan ranks within each independent block."""
    block = tl.program_id(0)
    i = block*BLOCK+tl.arange(0, BLOCK)
    key = tl.load(KEYS+i, i < N, 9223372036854775807)
    previous = tl.load(KEYS+i-1, (i < N) & (i > 0), -1)
    first = (i < N) & (key != 9223372036854775807) & ((i == 0) | (key != previous))
    rank = tl.cumsum(first.to(tl.int32), 0)
    tl.store(LOCAL_GROUP+i, rank, i < N)
    tl.store(BLOCK_COUNTS+block, tl.sum(first.to(tl.int32), 0))


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _triangle_group_pack(KEYS, LOCAL_GROUP, BLOCK_ENDS, QUEUE, RANGES,
                         N, BLOCK: tl.constexpr):
    """Pack exact ascending queue keys and run endpoints without atomics."""
    block = tl.program_id(0)
    i = block*BLOCK+tl.arange(0, BLOCK)
    key = tl.load(KEYS+i, i < N, 9223372036854775807)
    valid = (i < N) & (key != 9223372036854775807)
    previous = tl.load(KEYS+i-1, valid & (i > 0), -1)
    following = tl.load(KEYS+i+1, valid & (i+1 < N), -1)
    first = valid & ((i == 0) | (key != previous))
    last = valid & ((i+1 == N) | (key != following))
    before = tl.load(BLOCK_ENDS+block-1, block > 0, 0)
    group = before+tl.load(LOCAL_GROUP+i, i < N, 0)-1
    tl.store(QUEUE+group, key, first)
    tl.store(RANGES+group*2, i, first)
    tl.store(RANGES+group*2+1, i+1, last)


def group_sorted_triangle_work(keys, capacity):
    """Exact stable run grouping with block-sized scan metadata on the device.

    Caller supplies sorted nonnegative kernel/tile keys and an upper bound on
    unique keys. INT64_MAX padding is ignored, as in the reference grouping.
    No device count is read back, and no learned tensor enters this scheduler.
    """
    n = keys.numel()
    if keys.ndim != 1 or keys.dtype != torch.int64 or not 0 <= capacity <= n or n >= 2**31:
        raise ValueError('Invalid sorted tile keys or queue capacity')
    queue = torch.full((capacity,), -1, device=keys.device, dtype=torch.int64)
    ranges = torch.zeros((capacity, 2), device=keys.device, dtype=torch.int32)
    if not n:
        return queue, ranges, keys.new_zeros(())
    block = 256
    local = torch.empty(n, device=keys.device, dtype=torch.int32)
    counts = torch.empty(triton.cdiv(n, block), device=keys.device, dtype=torch.int32)
    _triangle_group_scan[(len(counts),)](keys, local, counts, n, block, num_warps=4)
    ends = counts.cumsum(0, dtype=torch.int64)
    torch._assert_async(ends[-1] <= capacity, 'Tile queue capacity exceeded')
    _triangle_group_pack[(len(counts),)](keys, local, ends, queue, ranges, n, block, num_warps=4)
    return queue, ranges, ends[-1]


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _expand_triangle_work(FACES, KERNELS, OFFSETS, FACE_COUNTS, FACE_STARTS,
                           TASK_STARTS, PATCH_TILES, KEYS, ENTRIES, N,
                           TILES: tl.constexpr, BLOCK: tl.constexpr):
    """Emit the original face/kernel/patch order without repeated-index tensors."""
    entry = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = entry < N
    face = tl.load(FACES+entry, valid, 0)
    kernel = tl.load(KERNELS+entry, valid, 0).to(tl.int64)
    count = tl.load(FACE_COUNTS+face, valid, 0)
    ordinal = entry-tl.load(OFFSETS+face, valid, 0)
    base = tl.load(TASK_STARTS+face, valid, 0)+ordinal*count
    patch = tl.load(FACE_STARTS+face, valid, 0)
    i = 0
    while i < tl.max(count, 0):
        active = valid & (i < count)
        tile = tl.load(PATCH_TILES+patch+i, active, 0)
        tl.store(KEYS+base+i, kernel*TILES+tile, active)
        tl.store(ENTRIES+base+i, entry.to(tl.int32), active)
        i += 1


def expand_triangle_work(support, geometry, patch_tiles, tile_count):
    """Preserve exact task order; scan fixed faces instead of all support pairs."""
    counts = support.counts*geometry.face_counts
    starts = counts.cumsum(0)-counts
    keys = torch.empty(support.tasks, device=support.kernels.device, dtype=torch.int64)
    entries = torch.empty(support.tasks, device=support.kernels.device, dtype=torch.int32)
    if len(support.kernels):
        _expand_triangle_work[(triton.cdiv(len(support.kernels), 32),)](
            support.entry_faces, support.kernels, support.offsets, geometry.face_counts,
            geometry.face_starts, starts, patch_tiles, keys, entries,
            len(support.kernels), tile_count, 32, num_warps=1)
    return keys, entries


@triton.jit
def _hybrid_emit(QUEUE, RANGES, COUNT, ENTRIES, ENTRY_FACES, CSR_OFFSETS,
                 TILE_PIXELS, FACE_IDS, PIXEL_OFFSETS, BARY, HEAT, METRIC,
                 EPSILON, SHARPNESS, REC_ENTRY, REC_WEIGHT, RESPONSE,
                 TILES: tl.constexpr, AREA: tl.constexpr, BP: tl.constexpr,
                 SEARCH_STEPS: tl.constexpr, SIGMA: tl.constexpr, DIST_EPS: tl.constexpr,
                 CAPTURE_RESPONSE: tl.constexpr, PIXEL_CANDIDATES=None,
                 CANDIDATES: tl.constexpr=0):
    task = tl.program_id(0)
    if task < tl.load(COUNT):
        key = tl.load(QUEUE+task)
        kernel, tile = key//TILES, key % TILES
        begin = tl.load(RANGES+task*2)
        end = tl.load(RANGES+task*2+1)
        lane = tl.program_id(1)*BP+tl.arange(0, BP)
        p = tl.load(TILE_PIXELS+tile*AREA+lane, lane < AREA, -1)
        valid = (lane < AREA) & (p >= 0)
        face = tl.load(FACE_IDS+p, valid, -1)
        # The kernel's supported triangles within this tile are sorted by face.
        # A visible pixel shades only the coefficient entry for its winning face.
        lo = tl.full((BP,), 0, tl.int32)+begin
        hi = tl.full((BP,), 0, tl.int32)+end
        for _ in tl.static_range(SEARCH_STEPS):
            mid = (lo+hi)//2
            entry = tl.load(ENTRIES+mid, (mid < end) & valid, 0)
            candidate = tl.load(ENTRY_FACES+entry, (mid < end) & valid, 2147483647)
            right = candidate < face
            lo = tl.where(right, mid+1, lo)
            hi = tl.where(right, hi, mid)
        entry = tl.load(ENTRIES+lo, (lo < end) & valid, 0)
        found = valid & (lo < end) & (tl.load(ENTRY_FACES+entry, valid & (lo < end), -1) == face)
        if CANDIDATES:
            # A triangle stores a union of its pixels' candidates. Do not let
            # a neighbor belonging to another pixel enter this pixel's top-30.
            left = tl.full((BP,), 0, tl.int32)
            right = tl.full((BP,), CANDIDATES, tl.int32)
            for _ in tl.static_range(CANDIDATES.bit_length()):
                mid = (left+right)//2
                candidate = tl.load(PIXEL_CANDIDATES+p*CANDIDATES+mid,
                                    found & (mid < CANDIDATES), 2147483647)
                advance = candidate < kernel
                left = tl.where(advance, mid+1, left)
                right = tl.where(advance, right, mid)
            admitted = tl.load(PIXEL_CANDIDATES+p*CANDIDATES+left,
                               found & (left < CANDIDATES), -1) == kernel
            found &= admitted
        b0 = tl.load(BARY+p*3, found, 0.)
        b1 = tl.load(BARY+p*3+1, found, 0.)
        b2 = tl.load(BARY+p*3+2, found, 0.)
        h = (tl.load(HEAT+entry*3, found, 0.)*b0+
             tl.load(HEAT+entry*3+1, found, 0.)*b1+
             tl.load(HEAT+entry*3+2, found, 0.)*b2)
        g0 = tl.load(METRIC+entry*6, found, 0.)
        g1 = tl.load(METRIC+entry*6+1, found, 0.)
        g2 = tl.load(METRIC+entry*6+2, found, 0.)
        g3 = tl.load(METRIC+entry*6+3, found, 0.)
        g4 = tl.load(METRIC+entry*6+4, found, 0.)
        g5 = tl.load(METRIC+entry*6+5, found, 0.)
        d = tl.maximum(g0*b0*b0+g1*b1*b1+g2*b2*b2+2*(g3*b0*b1+g4*b0*b2+g5*b1*b2), 0.)
        r = h*tl.exp(-(d+DIST_EPS)/(2*SIGMA*SIGMA))
        ep = tl.load(EPSILON+kernel)
        sh = tl.load(SHARPNESS+kernel)
        # Match the Torch float32 filter's rounded division. Approximate
        # reciprocal can add one ULP near saturation and swap top-30 members.
        y = libdevice.div_rn(1.,1.+libdevice.exp(-sh*(r-ep)))
        y0 = libdevice.div_rn(1.,1.+libdevice.exp(sh*ep))
        y1 = libdevice.div_rn(1.,1.+libdevice.exp(-sh*(1.-ep)))
        weight = tl.minimum(tl.maximum(libdevice.div_rn(y-y0,y1-y0+1e-8), 0.), 1.)
        # Unique (kernel,tile) keys + unique CSR incidences + disjoint visible
        # tile pixels give exclusive ownership of every compact record slot.
        slot = tl.load(PIXEL_OFFSETS+p, found, 0)+entry-tl.load(CSR_OFFSETS+face, found, 0)
        tl.store(REC_ENTRY+slot, entry, found)
        tl.store(REC_WEIGHT+slot, weight, found)
        if CAPTURE_RESPONSE:
            tl.store(RESPONSE+slot, r, found)


def hybrid_forward(heat, metric, colors, base, epsilon, sharpness, work, inner, sigma, distance_eps):
    from contextlib import nullcontext
    from hktex_splatting.triton import splat_contributions as gpu, splat_reconstruction as runtime_counts
    g, support = work.surface, work.support
    p, fragments = len(g.pixels), support.fragments
    def stage(name):
        return torch.profiler.record_function('splat::hybrid_'+name) if getattr(work, 'profile', False) else nullcontext()
    with stage('pixel_emission'):
        re = torch.full((fragments,), -1, device=heat.device, dtype=torch.int32)
        rw = heat.new_zeros(fragments)
        capture = getattr(work, 'capture_response', True)
        response = heat.new_zeros(fragments) if capture else rw
        candidates = getattr(work, 'pixel_candidates', None)
        if work.keys.numel():
            # Polynomial interpolation needs few registers: cover a full 8x8
            # tile per program instead of the small spectral-gather groups.
            group = min(64, g.tile_size**2)
            _hybrid_emit[(work.keys.numel(), triton.cdiv(g.tile_size**2, group))](
                work.keys, work.ranges, work.count, work.entries, support.entry_faces, support.offsets,
                g.tile_pixels, g.face_ids, work.pixel_offsets, g.bary, heat, metric, epsilon, sharpness,
                re, rw, response, g.tile_count, g.tile_size**2, group, (g.tile_size**2).bit_length(),
                sigma, distance_eps, capture,
                PIXEL_CANDIDATES=candidates, CANDIDATES=0 if candidates is None else candidates.shape[1],
                num_warps=2, enable_fp_fusion=False)
    with stage('top30_reduction'):
        se = torch.full((p, 32), 2147483647, device=heat.device, dtype=torch.int32)
        sw = heat.new_zeros((p, 32))
        # Compact records are already segmented by pixel, so no record sort or
        # histogram is needed. Reuse the validated shared top-30 primitive.
        # The compile-time identity branch never dereferences this pointer.
        # Avoid allocating/writing/reading an int64 permutation per record.
        identity = work.pixel_offsets
        active = g.active_ordinals
        if p and support.maximum<=512:
            gpu.merge[(p,)](work.pixel_offsets, identity, re, rw, active, se, sw, p,
                triton.next_power_of_2(support.maximum+32), inner, 32,
                IDENTITY_INDEX=True, num_warps=4, enable_fp_fusion=False)
        elif p:
            # Exact segmented top-k identity: topk(A union B) equals
            # topk(topk(A) union B), with the same global entry-ID tie key.
            # Process only overflowing pixels after the first bounded chunk.
            counts=support.counts[g.face_ids]
            for start in range(0,support.maximum,512):
                current=active if start==0 else torch.where(counts>start)[0].int()
                if len(current):
                    gpu.merge[(len(current),)](work.pixel_offsets,identity,re,rw,current,se,sw,p,
                        1024,inner,32,start,512,IDENTITY_INDEX=True,num_warps=4,enable_fp_fusion=False)
        rgb = heat.new_empty((p, 3))
        if p:
            runtime_counts.reconstruct[(p,)](se, sw, support.kernels, colors, base,
                g.face_ids, rgb, p, len(support.kernels), 32, num_warps=4, enable_fp_fusion=False)
    work.selected_entries, work.selected_weights = se, sw
    work.record_entries, work.responses = re, response if capture else None
    if capture:work.record_weights=rw
    return rgb, se, sw
