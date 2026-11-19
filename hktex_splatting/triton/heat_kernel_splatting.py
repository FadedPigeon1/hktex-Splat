"""Group supported triangle work and emit HKS pixel contributions with Triton.

GPU rendering uses three-corner coefficients and visible-pixel barycentrics.
Main API: group_triangle_work, _hybrid_emit, hybrid_forward."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _group_triangle_work(KEYS, FIRST, LAST, GROUP, QUEUE, RANGES,
                         N, BLOCK: tl.constexpr):
    # Task counts change with source movement. Specializing N compiles/cache-
    # loads a new variant for each count beyond the warmed training trajectory.
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < N
    first = tl.load(FIRST+i, valid, 0)
    last = tl.load(LAST+i, valid, 0)
    group = tl.load(GROUP+i, valid, 0)
    key = tl.load(KEYS+i, valid, -1)
    tl.store(QUEUE+group, key, valid & first)
    tl.store(RANGES+group*2, i, valid & first)
    tl.store(RANGES+group*2+1, i+1, valid & last)


def group_triangle_work(keys, first, last, groups, queue, ranges):
    if keys.numel():
        _group_triangle_work[(triton.cdiv(keys.numel(), 256),)](
            keys, first, last, groups, queue, ranges, keys.numel(), 256, num_warps=4)


@triton.jit
def _hybrid_emit(QUEUE, RANGES, COUNT, ENTRIES, ENTRY_FACES, CSR_OFFSETS,
                 TILE_PIXELS, FACE_IDS, PIXEL_OFFSETS, BARY, HEAT, METRIC,
                 EPSILON, SHARPNESS, REC_ENTRY, REC_WEIGHT, RESPONSE,
                 TILES: tl.constexpr, AREA: tl.constexpr, BP: tl.constexpr,
                 SEARCH_STEPS: tl.constexpr, SIGMA: tl.constexpr, DIST_EPS: tl.constexpr,
                 CAPTURE_RESPONSE: tl.constexpr):
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
        y = 1./(1.+tl.exp(-sh*(r-ep)))
        y0 = 1./(1.+tl.exp(sh*ep))
        y1 = 1./(1.+tl.exp(-sh*(1.-ep)))
        weight = tl.minimum(tl.maximum((y-y0)/(y1-y0+1e-8), 0.), 1.)
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
        if work.keys.numel():
            # Polynomial interpolation needs few registers: cover a full 8x8
            # tile per program instead of the small spectral-gather groups.
            group = min(64, g.tile_size**2)
            _hybrid_emit[(work.keys.numel(), triton.cdiv(g.tile_size**2, group))](
                work.keys, work.ranges, work.count, work.entries, support.entry_faces, support.offsets,
                g.tile_pixels, g.face_ids, work.pixel_offsets, g.bary, heat, metric, epsilon, sharpness,
                re, rw, response, g.tile_count, g.tile_size**2, group, (g.tile_size**2).bit_length(),
                sigma, distance_eps, capture, num_warps=4, enable_fp_fusion=False)
    with stage('top30_reduction'):
        se = torch.full((p, 32), 2147483647, device=heat.device, dtype=torch.int32)
        sw = heat.new_zeros((p, 32))
        # Compact records are already segmented by pixel, so no record sort or
        # histogram is needed. Reuse the validated shared top-30 primitive.
        identity = torch.arange(fragments, device=heat.device, dtype=torch.int64)
        active = g.active_ordinals
        if p:
            gpu.merge[(p,)](work.pixel_offsets, identity, re, rw, active, se, sw, p,
                triton.next_power_of_2(support.maximum+32), inner, 32,
                num_warps=4, enable_fp_fusion=False)
        rgb = heat.new_empty((p, 3))
        if p:
            runtime_counts.reconstruct[(p,)](se, sw, support.kernels, colors, base,
                g.face_ids, rgb, p, len(support.kernels), 32, num_warps=4, enable_fp_fusion=False)
    work.selected_entries, work.selected_weights = se, sw
    work.record_entries, work.responses = re, response if capture else None
    return rgb, se, sw
