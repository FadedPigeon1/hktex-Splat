"""Select, scan and pack fresh ordered triangle/kernel support on the GPU.

Shared HKS support construction: audited small-row selection and wide tiled merging.
Main API: support_distances, select_support_rows, merge_support_tile,
finish_support_tiles, pack_support and support_scan_* kernels."""
import triton
import triton.language as tl


@triton.jit
def pack_support(ROWS, VISIBLE, OFFSETS, KERNELS, FACES,
                 STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    face = tl.load(VISIBLE+row)
    count = tl.load(ROWS+row*STRIDE)
    start = tl.load(OFFSETS+face)
    lane = tl.arange(0, BLOCK)
    kernel = tl.load(ROWS+row*STRIDE+1+lane, lane<count, 0)
    tl.store(KERNELS+start+lane, kernel, lane<count)
    tl.store(FACES+start+lane, face, lane<count)


@triton.jit
def _support_limit(kth, radius, GUARD: tl.constexpr):
    # Keep the legacy sequence of float64 elementwise operations. In
    # particular, do not round the Python guard through a float32 scalar.
    guard = tl.full((), GUARD, tl.float64)
    return (kth+2*radius)+guard*((1+kth)+radius)


@triton.jit(do_not_specialize=['N', 'CAP', 'BASE', 'STRIDE'],
            do_not_specialize_on_alignment=['N', 'CAP', 'BASE', 'STRIDE'])
def pack_support_rows(DISTANCE, KTH, RADIUS, NEAREST, ROWS, N, CAP, BASE, STRIDE,
                      GUARD: tl.constexpr, BLOCK: tl.constexpr,
                      REPAIR=None, REPAIR_ONLY: tl.constexpr=False):
    row = tl.program_id(0)
    active = True
    if REPAIR_ONLY:
        active = tl.load(REPAIR+row) != 0
    if active:
        lane = tl.arange(0, BLOCK)
        source = tl.load(NEAREST+row*CAP+lane, lane < CAP, 0)
        distance = tl.load(DISTANCE+row*N+source, lane < CAP, float('inf'))
        kth = tl.load(KTH+row)
        radius = tl.load(RADIUS+row)
        valid = (lane < CAP) & (distance <= _support_limit(kth, radius, GUARD))
        ordered = tl.sort(tl.where(valid, source, N).to(tl.int32), descending=False)
        count = tl.sum(valid.to(tl.int32), 0)
        tl.store(ROWS+(BASE+row)*STRIDE, count)
        tl.store(ROWS+(BASE+row)*STRIDE+1+lane, ordered, lane < CAP)


@triton.jit
def _warp_sum_rows(values, WIDTH: tl.constexpr, PAIRS: tl.constexpr):
    lane = tl.arange(0, WIDTH)
    for shift in tl.static_range(5):
        offset = 16 >> shift
        peer = tl.where(lane % 32+offset < 32, lane+offset, lane)
        values = values+tl.gather(values, tl.broadcast_to(peer[None, :], (PAIRS, WIDTH)), 1)
    return values


@triton.jit(do_not_specialize=['N', 'CS0', 'CS1', 'SS0', 'SS1'],
            do_not_specialize_on_alignment=['N', 'CS0', 'CS1', 'SS0', 'SS1'])
def support_distances(CENTER, SOURCE, DISTANCE, N, CS0, CS1, SS0, SS1,
                      DIM: tl.constexpr, BD: tl.constexpr, PAIRS: tl.constexpr,
                      CUDA_SQRT: tl.constexpr):
    # Match ATen's 256-lane CUDA cdist reduction for DIM<=256. Each native
    # lane has at most one product, so no multiply-accumulate rounding differs.
    row = tl.program_id(1)
    source = tl.program_id(0)*PAIRS+tl.arange(0, PAIRS)
    dim = tl.arange(0, BD)
    c = tl.load(CENTER+row*CS0+dim*CS1, dim < DIM, 0.).to(tl.float64)
    s = tl.load(SOURCE+source[:, None]*SS0+dim[None, :]*SS1,
                (source[:, None] < N) & (dim[None, :] < DIM), 0.).to(tl.float64)
    delta = tl.abs(c[None, :]-s)
    partial = tl.where(dim[None, :] < DIM, delta*delta, 0.)
    partial = _warp_sum_rows(partial, BD, PAIRS)
    warp = tl.arange(0, 32)
    start = tl.where(warp < BD//32, warp*32, 0)
    warp_values = tl.gather(partial, tl.broadcast_to(start[None, :], (PAIRS, 32)), 1)
    warp_values = tl.where(warp[None, :] < BD//32, warp_values, 0.)
    total = _warp_sum_rows(warp_values, 32, PAIRS)
    total = tl.gather(total, tl.full((PAIRS, 1), 0, tl.int32), 1).reshape((PAIRS,))
    if CUDA_SQRT:
        value = cuda_libdevice.sqrt_rn(total)
    else:
        # Interpreter's np.sqrt(float64); CUDA always uses explicit RN libdevice.
        value = tl.sqrt(total)
    tl.store(DISTANCE+row*N+source, value, source < N)


@triton.jit(do_not_specialize=['N', 'K', 'CAP', 'BASE', 'STRIDE'],
            do_not_specialize_on_alignment=['N', 'K', 'CAP', 'BASE', 'STRIDE'])
def select_support_rows(DISTANCE, RADIUS, ROWS, KTH, TIES, N, K, CAP, BASE, STRIDE,
                        GUARD: tl.constexpr, BLOCK: tl.constexpr, CAPTURE: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    distance = tl.load(DISTANCE+row*N+lane, lane < N, float('nan'))
    # Distances are nonnegative. NaNs sort after infinity, as ATen radix kthvalue.
    key = tl.where((lane < N) & (distance == distance), distance.to(tl.uint64, bitcast=True),
                   tl.full((), 0xffffffffffffffff, tl.uint64))
    ordered = tl.sort(key, descending=False)
    kth_key = tl.sum(tl.where(lane == K-1, ordered, 0).to(tl.uint64), 0)
    kth = kth_key.to(tl.float64, bitcast=True)
    radius = tl.load(RADIUS+row)
    bound = _support_limit(kth, radius, GUARD)
    admitted = (lane < N) & (distance <= bound)
    raw_cut_key = tl.sum(tl.where(lane == CAP-1, ordered, 0).to(tl.uint64), 0)
    raw_cut = raw_cut_key.to(tl.float64, bitcast=True)
    infinity = tl.full((), 0x7ff0000000000000, tl.uint64)
    # Filtering is monotone: reuse the same raw order statistic for the cap.
    # Clamp to masked infinity if the raw cutoff is not admitted (including NaN).
    cutoff = tl.where(raw_cut <= bound, raw_cut_key, infinity)
    masked_key = tl.where(admitted, key, infinity)
    less = (lane < N) & (masked_key < cutoff)
    equal = (lane < N) & (masked_key == cutoff)
    strict_count = tl.sum(less.to(tl.int32), 0)
    equal_prefix = tl.cumsum(equal.to(tl.int32), 0)
    keep = admitted & (less | (equal & (equal_prefix <= CAP-strict_count)))
    count = tl.sum(keep.to(tl.int32), 0)
    destination = tl.cumsum(keep.to(tl.int32), 0)-1
    start = (BASE+row)*STRIDE
    tl.store(ROWS+start, count)
    # Padding and selected slots are disjoint, avoiding cross-warp store races.
    tl.store(ROWS+start+1+lane, N, (lane >= count) & (lane < CAP))
    tl.store(ROWS+start+1+destination, lane, keep)
    if CAPTURE:
        tl.store(KTH+row, kth)
        tl.store(TIES+row, tl.sum(equal.to(tl.int32), 0) > CAP-strict_count)


@triton.jit
def _sort_distance_ids(keys, sources, BLOCK: tl.constexpr, LOG_BLOCK: tl.constexpr):
    """Bitonic lexicographic order without reducing float64 distance precision."""
    lane = tl.arange(0, BLOCK)
    for stage in tl.static_range(1, LOG_BLOCK+1):
        for step in tl.static_range(stage):
            stride = 1 << (stage-1-step)
            peer = lane ^ stride
            other_key = tl.gather(keys, peer, 0)
            other_source = tl.gather(sources, peer, 0)
            greater = (keys > other_key) | ((keys == other_key) & (sources > other_source))
            less = (keys < other_key) | ((keys == other_key) & (sources < other_source))
            take_min = ((lane & (1 << stage)) == 0) == ((lane & stride) == 0)
            swap = tl.where(take_min, greater, less)
            keys = tl.where(swap, other_key, keys)
            sources = tl.where(swap, other_source, sources)
    return keys, sources


@triton.jit(do_not_specialize=['N', 'TILE_N', 'START'],
            do_not_specialize_on_alignment=['N', 'TILE_N', 'START'])
def merge_support_tile(DISTANCE, KEYS, SOURCES, N, TILE_N, START,
                       TILE: tl.constexpr, KEEP: tl.constexpr, BLOCK: tl.constexpr,
                       LOG_BLOCK: tl.constexpr):
    """Merge one source tile into a bounded nearest-candidate list per face."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    sentinel = tl.full((), 0xffffffffffffffff, tl.uint64)
    distance = tl.load(DISTANCE+row*TILE_N+lane, lane < TILE_N, float('nan'))
    fresh_key = tl.where((lane < TILE_N) & (distance == distance),
                         distance.to(tl.uint64, bitcast=True), sentinel)
    old = (lane >= TILE) & (lane < TILE+KEEP)
    old_key = tl.load(KEYS+row*KEEP+lane-TILE, old, -1).to(tl.uint64, bitcast=True)
    old_source = tl.load(SOURCES+row*KEEP+lane-TILE, old, N)
    keys = tl.where(old, old_key, fresh_key)
    sources = tl.where(old, old_source, tl.where(lane < TILE_N, START+lane, N))
    keys, sources = _sort_distance_ids(keys, sources, BLOCK, LOG_BLOCK)
    tl.store(KEYS+row*KEEP+lane, keys, lane < KEEP)
    tl.store(SOURCES+row*KEEP+lane, sources, lane < KEEP)


@triton.jit(do_not_specialize=['N', 'K', 'CAP', 'BASE', 'STRIDE'],
            do_not_specialize_on_alignment=['N', 'K', 'CAP', 'BASE', 'STRIDE'])
def finish_support_tiles(KEYS, SOURCES, RADIUS, ROWS, N, K, CAP, BASE, STRIDE,
                         GUARD: tl.constexpr, KEEP: tl.constexpr):
    """Apply the unchanged spectral bound and emit ascending source IDs."""
    row = tl.program_id(0)
    lane = tl.arange(0, KEEP)
    key = tl.load(KEYS+row*KEEP+lane).to(tl.uint64, bitcast=True)
    source = tl.load(SOURCES+row*KEEP+lane)
    kth_key = tl.sum(tl.where(lane == K-1, key, 0).to(tl.uint64), 0)
    bound = _support_limit(kth_key.to(tl.float64, bitcast=True), tl.load(RADIUS+row), GUARD)
    admitted = (lane < CAP) & (source < N) & (key.to(tl.float64, bitcast=True) <= bound)
    ordered = tl.sort(tl.where(admitted, source, N), descending=False)
    tl.store(ROWS+(BASE+row)*STRIDE, tl.sum(admitted.to(tl.int32), 0))
    tl.store(ROWS+(BASE+row)*STRIDE+1+lane, ordered, lane < CAP)


@triton.jit(do_not_specialize=['FACES', 'STRIDE'],
            do_not_specialize_on_alignment=['FACES', 'STRIDE'])
def support_scan_blocks(ROWS, FACE_ROWS, TILE_COUNTS, PIXEL_COUNTS, COUNTS, OFFSETS, STATS,
                        FACES, STRIDE, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    face = block*BLOCK+tl.arange(0, BLOCK)
    row = tl.load(FACE_ROWS+face, face < FACES, -1)
    count = tl.load(ROWS+row*STRIDE, (face < FACES) & (row >= 0), 0).to(tl.int64)
    tiles = tl.load(TILE_COUNTS+face, face < FACES, 0)
    pixels = tl.load(PIXEL_COUNTS+face, face < FACES, 0)
    tl.store(COUNTS+face, count, face < FACES)
    tl.store(OFFSETS+face+1, tl.cumsum(count, 0), face < FACES)
    tl.store(STATS+block*4, tl.sum(count, 0))
    tl.store(STATS+block*4+1, tl.max(count, 0))
    tl.store(STATS+block*4+2, tl.sum(count*tiles, 0))
    tl.store(STATS+block*4+3, tl.sum(count*pixels, 0))


@triton.jit(do_not_specialize=['BLOCKS'], do_not_specialize_on_alignment=['BLOCKS'])
def support_scan_totals(STATS, STARTS, METADATA, BLOCKS, WIDTH: tl.constexpr):
    block = tl.arange(0, WIDTH)
    pairs = tl.load(STATS+block*4, block < BLOCKS, 0)
    maximum = tl.load(STATS+block*4+1, block < BLOCKS, 0)
    tasks = tl.load(STATS+block*4+2, block < BLOCKS, 0)
    fragments = tl.load(STATS+block*4+3, block < BLOCKS, 0)
    tl.store(STARTS+block, tl.cumsum(pairs, 0)-pairs, block < BLOCKS)
    tl.store(METADATA, tl.sum(pairs, 0))
    tl.store(METADATA+1, tl.max(maximum, 0))
    tl.store(METADATA+2, tl.sum(tasks, 0))
    tl.store(METADATA+3, tl.sum(fragments, 0))


@triton.jit(do_not_specialize=['FACES'], do_not_specialize_on_alignment=['FACES'])
def support_scan_offsets(OFFSETS, STARTS, FACES, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    face = block*BLOCK+tl.arange(0, BLOCK)
    start = tl.load(STARTS+block)
    local = tl.load(OFFSETS+face+1, face < FACES, 0)
    tl.store(OFFSETS+face+1, start+local, face < FACES)
    if block == 0:
        tl.store(OFFSETS, 0)


from triton.language.extra.cuda import libdevice as cuda_libdevice
