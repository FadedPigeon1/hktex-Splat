"""Select, scan and pack fresh ordered triangle/kernel support on the GPU.

Current HKS uses graph_surface_frontier and support count/pack kernels.
Retained reference utilities include small-row selection and wide tiled merging.
Main API: support_distances, select_support_rows, merge_support_tile[_compact],
finish_support_tiles, pack_support and support_scan_* kernels."""
import triton
import triton.language as tl
# Historical imports remain valid for benchmark commands and cached callers.
from hktex_splatting.triton.spectral_support_distances import (
    _support_limit, support_distances, support_distances_blocked, support_seed_upper,
)


@triton.jit
def graph_surface_frontier(ADJACENCY, SOURCE_FACES, SEEN, QUEUE, OVERFLOW,
                           DEGREE: tl.constexpr, CAPACITY: tl.constexpr,
                           HASH_SIZE: tl.constexpr, HOPS: tl.constexpr,
                           BLOCK: tl.constexpr):
    """Exact bounded BFS per source; overflow requests the unbounded tensor oracle.

    Hash entries and queue rows belong exclusively to one GPU program. Every
    face at most HOPS edges away is visited, including invisible intermediates.
    No heat, learned parameter, or visibility cutoff controls traversal.
    """
    kernel = tl.program_id(0).to(tl.int64)
    seed = tl.load(SOURCE_FACES+kernel).to(tl.int32)
    queue_base = kernel*CAPACITY
    hash_base = kernel*HASH_SIZE
    seed_slot = (seed.to(tl.uint32)*2654435761) & (HASH_SIZE-1)
    tl.store(SEEN+hash_base+seed_slot, seed)
    tl.store(QUEUE+queue_base, seed)
    tl.debug_barrier()
    lane = tl.arange(0, BLOCK)
    head = 0
    tail = 1
    overflow = 0
    for _ in range(HOPS):
        end = tail
        cursor = head
        while (cursor < end) & (overflow == 0):
            index = cursor+lane
            face = tl.load(QUEUE+queue_base+index, index < end, 0)
            for edge in range(DEGREE):
                if overflow == 0:
                    neighbor = tl.load(ADJACENCY+face.to(tl.int64)*DEGREE+edge,
                                       index < end, -1).to(tl.int32)
                    pending = (index < end) & (neighbor >= 0)
                    slot = (neighbor.to(tl.uint32)*2654435761) & (HASH_SIZE-1)
                    inserted = tl.full((BLOCK,), False, tl.int1)
                    probes = 0
                    while (tl.sum(pending.to(tl.int32), 0) > 0) & (probes < HASH_SIZE):
                        # atomic_cas has no mask. -2 can never occur in this
                        # table, so inactive lanes cannot modify an entry.
                        compare = tl.where(pending, -1, -2).to(tl.int32)
                        previous = tl.atomic_cas(SEEN+hash_base+slot, compare, neighbor)
                        inserted = inserted | (pending & (previous == -1))
                        pending = pending & (previous != -1) & (previous != neighbor)
                        slot = (slot+1) & (HASH_SIZE-1)
                        probes += 1
                    added = tl.sum(inserted.to(tl.int32), 0)
                    rank = tl.cumsum(inserted.to(tl.int32), 0)-1
                    tl.store(QUEUE+queue_base+tail+rank, neighbor,
                             inserted & (tail+rank < CAPACITY))
                    overflow = tl.where((tail+added > CAPACITY) |
                                        (tl.sum(pending.to(tl.int32), 0) > 0), 1, overflow)
                    tail += added
                    tl.debug_barrier()
            cursor += BLOCK
        head = end
    tl.store(OVERFLOW+kernel, overflow)


@triton.jit
def count_kernel_surface_support(NEIGHBORS, SOURCE_FACES, PIXEL_COUNTS,
                                 SOURCE_COUNTS, VISIBLE_COUNTS, FACE_COUNTS,
                                 WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                                 SOURCE_INDEXED: tl.constexpr = False):
    """Count the exact fixed-hop relation and its current visible intersection."""
    kernel = tl.program_id(0)
    source_face = kernel if SOURCE_INDEXED else tl.load(SOURCE_FACES+kernel)
    lane = tl.arange(0, BLOCK)
    face = tl.load(NEIGHBORS+source_face*WIDTH+lane, lane < WIDTH, -1)
    valid = (lane < WIDTH) & (face >= 0)
    visible = valid & (tl.load(PIXEL_COUNTS+face, valid, 0) > 0)
    tl.store(SOURCE_COUNTS+kernel, tl.sum(valid.to(tl.int32), 0))
    tl.store(VISIBLE_COUNTS+kernel, tl.sum(visible.to(tl.int32), 0))
    tl.atomic_add(FACE_COUNTS+face, 1, visible)


@triton.jit(do_not_specialize=['SOURCES'], do_not_specialize_on_alignment=['SOURCES'])
def pack_kernel_surface_support(NEIGHBORS, SOURCE_FACES, PIXEL_COUNTS,
                                SOURCE_OFFSETS, KEYS, SOURCES,
                                WIDTH: tl.constexpr, BLOCK: tl.constexpr,
                                SOURCE_INDEXED: tl.constexpr = False):
    """Pack only visible incidences; global face/source sorting stays unchanged."""
    kernel = tl.program_id(0)
    source_face = kernel if SOURCE_INDEXED else tl.load(SOURCE_FACES+kernel)
    lane = tl.arange(0, BLOCK)
    face = tl.load(NEIGHBORS+source_face*WIDTH+lane, lane < WIDTH, -1)
    valid = (lane < WIDTH) & (face >= 0)
    visible = valid & (tl.load(PIXEL_COUNTS+face, valid, 0) > 0)
    rank = tl.cumsum(visible.to(tl.int32), axis=0)-1
    offset = tl.load(SOURCE_OFFSETS+kernel)
    key = face.to(tl.int64)*SOURCES+kernel
    tl.store(KEYS+offset+rank, key, visible)


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


@triton.jit
def _exact_radix_rank(keys, RANK, BLOCK: tl.constexpr):
    """Exact unsigned key order statistic; no approximate pivot termination."""
    active = tl.full((BLOCK,), True, tl.int1)
    rank = RANK
    cutoff = tl.full((), 0, tl.uint64)
    done = tl.full((), False, tl.int1)
    bins = tl.arange(0, 256)
    byte = tl.full((), 0, tl.int32)
    # A bounded GPU loop avoids replicating the large row's code eight times.
    while (byte < 8) & (~done):
        low = tl.min(tl.where(active, keys, tl.full((), 0xffffffffffffffff, tl.uint64)), 0)
        high = tl.max(tl.where(active, keys, 0), 0)
        if low == high:
            cutoff = low
            done = True
        else:
            shift = 56-byte*8
            low_digit = (low >> shift) & 255
            high_digit = (high >> shift) & 255
            if low_digit == high_digit:
                # All surviving keys share this byte: no histogram or rank
                # update is needed. This is an exact prefix shortcut.
                cutoff = cutoff | (low_digit << shift)
            else:
                digit = ((keys >> shift) & 255).to(tl.int32)
                histogram = tl.histogram(digit, 256, mask=active)
                cumulative = tl.cumsum(histogram, 0)
                bucket = tl.min(tl.where(cumulative >= rank, bins, 256), 0)
                rank = rank-tl.sum(tl.where(bins < bucket, histogram, 0), 0)
                cutoff = cutoff | (bucket.to(tl.uint64) << shift)
                active = active & (digit == bucket)
        byte += 1
    return cutoff


@triton.jit(do_not_specialize=['N', 'K', 'CAP', 'BASE', 'STRIDE'],
            do_not_specialize_on_alignment=['N', 'K', 'CAP', 'BASE', 'STRIDE'])
def select_support_row_radix(DISTANCE, RADIUS, ROWS, N, K, CAP, BASE, STRIDE,
                             GUARD: tl.constexpr, BLOCK: tl.constexpr):
    """Select wide-row neighbor/cap cutoffs directly, without source-tile merges."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    distance = tl.load(DISTANCE+row*N+lane, lane < N, float('nan'))
    key = tl.where((lane < N) & (distance == distance), distance.to(tl.uint64, bitcast=True),
                   tl.full((), 0xffffffffffffffff, tl.uint64))
    kth = _exact_radix_rank(key, K, BLOCK).to(tl.float64, bitcast=True)
    cutoff = _exact_radix_rank(key, CAP, BLOCK)
    less = (lane < N) & (key < cutoff)
    equal = (lane < N) & (key == cutoff)
    nearest = less | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= CAP-tl.sum(less.to(tl.int32), 0)))
    bound = _support_limit(kth, tl.load(RADIUS+row), GUARD)
    admitted = nearest & (key.to(tl.float64, bitcast=True) <= bound)
    count = tl.sum(admitted.to(tl.int32), 0)
    destination = tl.cumsum(admitted.to(tl.int32), 0)-1
    start = (BASE+row)*STRIDE
    tl.store(ROWS+start, count)
    tl.store(ROWS+start+1+lane, N, (lane >= count) & (lane < CAP))
    tl.store(ROWS+start+1+destination, lane, admitted)


@triton.jit
def pack_support_cutoffs(DISTANCE,RADIUS,KTH,CUTOFF,ROWS,N,CAP,BASE,STRIDE,KS,CS,
                         GUARD:tl.constexpr,BLOCK:tl.constexpr):
    """Pack exact topk value cutoffs; boundary ties use ascending source IDs."""
    row=tl.program_id(0);lane=tl.arange(0,BLOCK)
    distance=tl.load(DISTANCE+row*N+lane,lane<N,float('nan'))
    sentinel=tl.full((),0xffffffffffffffff,tl.uint64)
    key=tl.where((lane<N)&(distance==distance),distance.to(tl.uint64,bitcast=True),sentinel)
    kth=tl.load(KTH+row*KS);raw=tl.load(CUTOFF+row*CS)
    cutoff=tl.where(raw==raw,raw.to(tl.uint64,bitcast=True),sentinel)
    less=(lane<N)&(key<cutoff);equal=(lane<N)&(key==cutoff)
    nearest=less|(equal&(tl.cumsum(equal.to(tl.int32),0)<=CAP-tl.sum(less.to(tl.int32),0)))
    admitted=nearest&(distance<=_support_limit(kth,tl.load(RADIUS+row),GUARD))
    count=tl.sum(admitted.to(tl.int32),0);destination=tl.cumsum(admitted.to(tl.int32),0)-1
    start=(BASE+row)*STRIDE
    tl.store(ROWS+start,count)
    tl.store(ROWS+start+1+lane,N,(lane>=count)&(lane<CAP))
    tl.store(ROWS+start+1+destination,lane,admitted)


@triton.jit(do_not_specialize=['N', 'TILE_N', 'START'],
            do_not_specialize_on_alignment=['N', 'TILE_N', 'START'])
def merge_support_tile_compact(DISTANCE, KEYS, SOURCES, N, TILE_N, START,
                               TILE: tl.constexpr, KEEP: tl.constexpr,
                               LOG_TILE: tl.constexpr, LOG_MERGE: tl.constexpr):
    """Exact local top-KEEP and merge of two sorted KEEP lists.

    Every source is examined; float64 key bits and source-ID ties are retained.
    The first tile initializes the lists, avoiding host sentinel fills.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    distance = tl.load(DISTANCE+row*TILE_N+lane, lane < TILE_N, float('nan'))
    sentinel = tl.full((), 0xffffffffffffffff, tl.uint64)
    keys = tl.where((lane < TILE_N) & (distance == distance),
                    distance.to(tl.uint64, bitcast=True), sentinel)
    sources = tl.where(lane < TILE_N, START+lane, N)
    keys, sources = _sort_distance_ids(keys, sources, TILE, LOG_TILE)
    if START == 0:
        initial_lane = tl.arange(0, KEEP)
        tl.store(KEYS+row*KEEP+initial_lane, tl.gather(keys, initial_lane, 0))
        tl.store(SOURCES+row*KEEP+initial_lane, tl.gather(sources, initial_lane, 0))
    else:
        out = tl.arange(0, 2*KEEP)
        fresh = tl.minimum(out, KEEP-1)
        old = 2*KEEP-1-out
        old_keys = tl.load(KEYS+row*KEEP+old, out >= KEEP, -1).to(tl.uint64, bitcast=True)
        old_sources = tl.load(SOURCES+row*KEEP+old, out >= KEEP, N)
        joined_keys = tl.where(out < KEEP, tl.gather(keys, fresh, 0), old_keys)
        joined_sources = tl.where(out < KEEP, tl.gather(sources, fresh, 0), old_sources)
        # Ascending fresh prefix + descending old list form a bitonic sequence.
        for step in tl.static_range(LOG_MERGE):
            stride = KEEP >> step
            peer = out ^ stride
            other_key = tl.gather(joined_keys, peer, 0)
            other_source = tl.gather(joined_sources, peer, 0)
            greater = (joined_keys > other_key) | ((joined_keys == other_key) & (joined_sources > other_source))
            less = (joined_keys < other_key) | ((joined_keys == other_key) & (joined_sources < other_source))
            swap = tl.where((out & stride) == 0, greater, less)
            joined_keys = tl.where(swap, other_key, joined_keys)
            joined_sources = tl.where(swap, other_source, joined_sources)
        tl.store(KEYS+row*KEEP+out, joined_keys, out < KEEP)
        tl.store(SOURCES+row*KEEP+out, joined_sources, out < KEEP)


@triton.jit(do_not_specialize=['N', 'TILE_N', 'START'],
            do_not_specialize_on_alignment=['N', 'TILE_N', 'START'])
def merge_support_tile_radix(DISTANCE, LOCAL_KEYS, LOCAL_SOURCES, KEYS, SOURCES, N, TILE_N, START,
                               TILE: tl.constexpr, KEEP: tl.constexpr,
                               LOG_KEEP: tl.constexpr, LOG_MERGE: tl.constexpr):
    """Exact byte-radix local selection and small sorted-list merge.

    Every source is examined; float64 key bits and source-ID ties are retained.
    The first tile initializes the lists, avoiding host sentinel fills.
    """
    row = tl.program_id(0)
    lane = tl.arange(0, TILE)
    distance = tl.load(DISTANCE+row*TILE_N+lane, lane < TILE_N, float('nan'))
    sentinel = tl.full((), 0xffffffffffffffff, tl.uint64)
    keys = tl.where((lane < TILE_N) & (distance == distance),
                    distance.to(tl.uint64, bitcast=True), sentinel)
    sources = tl.where(lane < TILE_N, START+lane, N)
    active = tl.full((TILE,), True, tl.int1)
    rank = tl.full((), KEEP, tl.int32)
    cutoff = tl.full((), 0, tl.uint64)
    done = tl.full((), False, tl.int1)
    bins = tl.arange(0, 256)
    # Most-significant-byte selection, entirely within this GPU program.
    # Early exit is exact only when all surviving keys are identical.
    for byte in tl.static_range(8):
        if not done:
            low = tl.min(tl.where(active, keys, sentinel), 0)
            high = tl.max(tl.where(active, keys, 0), 0)
            if low == high:
                cutoff = low
                done = True
            else:
                shift = 56-byte*8
                digit = ((keys >> shift) & 255).to(tl.int32)
                histogram = tl.histogram(digit, 256, mask=active)
                cumulative = tl.cumsum(histogram, 0)
                bucket = tl.min(tl.where(cumulative >= rank, bins, 256), 0)
                rank -= tl.sum(tl.where(bins < bucket, histogram, 0), 0)
                cutoff = cutoff | (bucket.to(tl.uint64) << shift)
                active = active & (digit == bucket)
    radix_less = keys < cutoff
    radix_equal = keys == cutoff
    retained = radix_less | (radix_equal & (tl.cumsum(radix_equal.to(tl.int32), 0) <= KEEP-tl.sum(radix_less.to(tl.int32), 0)))
    destination = tl.cumsum(retained.to(tl.int32), 0)-1
    tl.store(LOCAL_KEYS+row*KEEP+destination, keys, retained)
    tl.store(LOCAL_SOURCES+row*KEEP+destination, sources, retained)
    # Unique compact destinations are written by this CTA. Synchronize before
    # its other warps gather the compact prefix; no cross-CTA sharing occurs.
    tl.debug_barrier()
    prefix_lane = tl.arange(0, KEEP)
    prefix_keys = tl.load(LOCAL_KEYS+row*KEEP+prefix_lane).to(tl.uint64, bitcast=True)
    prefix_sources = tl.load(LOCAL_SOURCES+row*KEEP+prefix_lane)
    keys, sources = _sort_distance_ids(prefix_keys, prefix_sources, KEEP, LOG_KEEP)
    if START == 0:
        initial_lane = tl.arange(0, KEEP)
        tl.store(KEYS+row*KEEP+initial_lane, tl.gather(keys, initial_lane, 0))
        tl.store(SOURCES+row*KEEP+initial_lane, tl.gather(sources, initial_lane, 0))
    else:
        out = tl.arange(0, 2*KEEP)
        fresh = tl.minimum(out, KEEP-1)
        old = 2*KEEP-1-out
        old_keys = tl.load(KEYS+row*KEEP+old, out >= KEEP, -1).to(tl.uint64, bitcast=True)
        old_sources = tl.load(SOURCES+row*KEEP+old, out >= KEEP, N)
        joined_keys = tl.where(out < KEEP, tl.gather(keys, fresh, 0), old_keys)
        joined_sources = tl.where(out < KEEP, tl.gather(sources, fresh, 0), old_sources)
        # Ascending fresh prefix + descending old list form a bitonic sequence.
        for step in tl.static_range(LOG_MERGE):
            stride = KEEP >> step
            peer = out ^ stride
            other_key = tl.gather(joined_keys, peer, 0)
            other_source = tl.gather(joined_sources, peer, 0)
            greater = (joined_keys > other_key) | ((joined_keys == other_key) & (joined_sources > other_source))
            less = (joined_keys < other_key) | ((joined_keys == other_key) & (joined_sources < other_source))
            swap = tl.where((out & stride) == 0, greater, less)
            joined_keys = tl.where(swap, other_key, joined_keys)
            joined_sources = tl.where(swap, other_source, joined_sources)
        tl.store(KEYS+row*KEEP+out, joined_keys, out < KEEP)
        tl.store(SOURCES+row*KEEP+out, joined_sources, out < KEEP)
