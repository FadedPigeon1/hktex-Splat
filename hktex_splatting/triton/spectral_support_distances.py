"""Exact spectral support distances, conservative pruning and reduction oracles.

Rendering/training candidate selection uses float64 differences and the native
warp addition order. COMPACT=False retains the old gather reduction solely for
numerical/performance comparisons; it is not a different heat-kernel formula.
"""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice as cuda_libdevice

@triton.jit
def _support_limit(kth, radius, GUARD: tl.constexpr):
    # Keep the legacy sequence of float64 elementwise operations. In
    # particular, do not round the Python guard through a float32 scalar.
    guard = tl.full((), GUARD, tl.float64)
    return (kth+2*radius)+guard*((1+kth)+radius)


@triton.jit
def _warp_sum_rows(values, WIDTH: tl.constexpr, PAIRS: tl.constexpr):
    lane = tl.arange(0, WIDTH)
    for shift in tl.static_range(5):
        offset = 16 >> shift
        peer = tl.where(lane % 32+offset < 32, lane+offset, lane)
        values = values+tl.gather(values, tl.broadcast_to(peer[None, :], (PAIRS, WIDTH)), 1)
    return values


@triton.jit
def _add_halves(values, ROWS: tl.constexpr, WIDTH: tl.constexpr):
    left,right=tl.split(values.reshape((ROWS,2,WIDTH//2)).permute((0,2,1)))
    return left+right


@triton.jit
def _distance_sum(values, WIDTH: tl.constexpr, PAIRS: tl.constexpr, COMPACT: tl.constexpr):
    if COMPACT:
        # Keep precisely the lane-0 dependency tree: offsets 16,8,4,2,1
        # inside each warp, then the same tree over its warp totals. Unlike
        # gather-to-every-lane, no unused suffix sums or 32-lane padding survive.
        groups:tl.constexpr=WIDTH//32
        rows:tl.constexpr=PAIRS*groups
        value=values.reshape((rows,32))
        value=_add_halves(value,rows,32)
        value=_add_halves(value,rows,16)
        value=_add_halves(value,rows,8)
        value=_add_halves(value,rows,4)
        value=_add_halves(value,rows,2)
        value=value.reshape((PAIRS,groups))
        if groups==8:value=_add_halves(value,PAIRS,8)
        if groups>=4:value=_add_halves(value,PAIRS,4)
        if groups>=2:value=_add_halves(value,PAIRS,2)
        return value.reshape((PAIRS,))
    else:
        partial=_warp_sum_rows(values,WIDTH,PAIRS)
        warp=tl.arange(0,32)
        start=tl.where(warp<WIDTH//32,warp*32,0)
        total=tl.gather(partial,tl.broadcast_to(start[None,:],(PAIRS,32)),1)
        total=tl.where(warp[None,:]<WIDTH//32,total,0.)
        total=_warp_sum_rows(total,32,PAIRS)
        return tl.gather(total,tl.full((PAIRS,1),0,tl.int32),1).reshape((PAIRS,))


@triton.jit(do_not_specialize=['N', 'CS0', 'CS1', 'SS0', 'SS1'],
            do_not_specialize_on_alignment=['N', 'CS0', 'CS1', 'SS0', 'SS1'])
def support_distances(CENTER, SOURCE, DISTANCE, N, CS0, CS1, SS0, SS1,
                      DIM: tl.constexpr, BD: tl.constexpr, PAIRS: tl.constexpr,
                      CUDA_SQRT: tl.constexpr, COMPACT: tl.constexpr=True):
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
    total = _distance_sum(partial, BD, PAIRS, COMPACT)
    if CUDA_SQRT:
        value = cuda_libdevice.sqrt_rn(total)
    else:
        # Interpreter's np.sqrt(float64); CUDA always uses explicit RN libdevice.
        value = tl.sqrt(total)
    tl.store(DISTANCE+row*N+source, value, source < N)


@triton.jit
def support_distances_blocked(CENTER, SOURCE, DISTANCE, M, N, CS0, CS1, SS0, SS1,
                              DIM: tl.constexpr, BD: tl.constexpr,
                              BF: tl.constexpr, BS: tl.constexpr,
                              CUDA_SQRT: tl.constexpr, UPPER=None,
                              PRUNE: tl.constexpr=False, PREFIX: tl.constexpr=16,
                              COMPACT: tl.constexpr=True):
    """Experimental exact face/source tiles; share inputs across several pairs.

    Preserve the audited explicit warp reduction and RN sqrt. This is not a
    dot-product distance identity or a reduced-precision approximation.
    """
    face = tl.program_id(1)*BF+tl.arange(0, BF)
    source = tl.program_id(0)*BS+tl.arange(0, BS)
    dim = tl.arange(0, BD)
    c = tl.load(CENTER+face[:, None]*CS0+dim[None, :]*CS1,
                (face[:, None] < M) & (dim[None, :] < DIM), 0.).to(tl.float64)
    admitted=tl.full((BF,BS),True,tl.int1)
    if PRUNE:
        # Nonnegative partial sums use the SAME reduction tree as full cdist.
        # Monotonic float64 arithmetic makes this a lower bound, including RN
        # sqrt. The seed bound is >= the true kth-distance support bound.
        prefix=tl.load(SOURCE+source[:,None]*SS0+dim[None,:]*SS1,
            (source[:,None]<N)&(dim[None,:]<PREFIX),0.).to(tl.float64)
        delta_prefix=c[:,None,:]-prefix[None,:,:]
        lower=tl.where(dim[None,None,:]<PREFIX,delta_prefix*delta_prefix,0.).reshape((BF*BS,BD))
        lower=_distance_sum(lower,BD,BF*BS,COMPACT).reshape((BF,BS))
        lower=cuda_libdevice.sqrt_rn(lower) if CUDA_SQRT else tl.sqrt(lower)
        upper=tl.load(UPPER+face,face<M,float('inf'))
        admitted=(~(lower>upper[:,None]))&(face[:,None]<M)&(source[None,:]<N)  # NaNs stay conservative.
    needed=tl.sum(admitted.to(tl.int32),0)>0
    compute=tl.full((),True,tl.int1)
    if PRUNE:compute=tl.sum(needed.to(tl.int32),0)>0
    if compute:
        s = tl.load(SOURCE+source[:, None]*SS0+dim[None, :]*SS1,
                    (source[:, None] < N) & (dim[None, :] < DIM)&needed[:,None], 0.).to(tl.float64)
        delta = tl.abs(c[:, None, :]-s[None, :, :])
        partial = tl.where(dim[None, None, :] < DIM, delta*delta, 0.).reshape((BF*BS, BD))
        total = _distance_sum(partial,BD,BF*BS,COMPACT).reshape((BF,BS))
        value = cuda_libdevice.sqrt_rn(total) if CUDA_SQRT else tl.sqrt(total)
    else:value=tl.full((BF,BS),float('inf'),tl.float64)
    if PRUNE:value=tl.where(admitted,value,float('inf'))
    tl.store(DISTANCE+face[:, None]*N+source[None, :], value,
             (face[:, None] < M) & (source[None, :] < N))


@triton.jit
def support_seed_upper(DISTANCE,RADIUS,UPPER,K,GUARD:tl.constexpr,BLOCK:tl.constexpr):
    """Any K full source distances give a conservative kth-support upper bound."""
    row=tl.program_id(0);lane=tl.arange(0,BLOCK)
    value=tl.load(DISTANCE+row*K+lane,lane<K,0.)
    # A nonfinite seed disables pruning for this row.
    finite=tl.sum(((lane<K)&((value!=value)|(value==float('inf')))).to(tl.int32),0)==0
    bound=_support_limit(tl.max(value,0),tl.load(RADIUS+row),GUARD)
    tl.store(UPPER+row,tl.where(finite,bound,float('inf')))

