"""Triton live-source preparation, corner heat/Gram contractions and their adjoints.

GPU corner heat evaluation and source-state adjoints for rendering/training.
Main API: source_grid_forward/backward, source_forward/backward,
coefficients_forward/parallel, coefficients_backward. Mesh and spectral arrays stay fixed."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _grid_lower_bound(AXIS, query, SIZE: tl.constexpr, LOG: tl.constexpr, BLOCK: tl.constexpr):
    lo = tl.full((BLOCK,), 0, tl.int32)
    hi = tl.full((BLOCK,), SIZE, tl.int32)
    for _ in tl.static_range(SIZE.bit_length()):
        mid = (lo+hi)//2
        value = tl.load(AXIS+mid, mid < SIZE, float('inf'))
        if LOG:value = tl.log(1.+value)
        right = value < query
        lo = tl.where(right, mid+1, lo)
        hi = tl.where(right, hi, mid)
    return tl.minimum(tl.maximum(lo, 1), SIZE-1)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def source_grid_forward(ROT, SCALE_AXIS, ANGLE, SCALE, GI, GW, N,
                        R: tl.constexpr, A: tl.constexpr, LOG: tl.constexpr,
                        ROT_MAJOR: tl.constexpr, EPS: tl.constexpr,
                        CLAMP: tl.constexpr, BLOCK: tl.constexpr):
    """Native four-cell bilinear source mixture, without source-count specialization."""
    k = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    angle = tl.load(ANGLE+k, k < N, 0.)
    scale = tl.load(SCALE+k, k < N, 1.)
    u = tl.log(1.+scale) if LOG else scale
    r1 = _grid_lower_bound(ROT, angle, R, False, BLOCK)
    a1 = _grid_lower_bound(SCALE_AXIS, u, A, LOG, BLOCK)
    r0 = r1-1; a0 = a1-1
    x0 = tl.load(ROT+r0); x1 = tl.load(ROT+r1)
    y0 = tl.load(SCALE_AXIS+a0); y1 = tl.load(SCALE_AXIS+a1)
    if LOG:y0 = tl.log(1.+y0); y1 = tl.log(1.+y1)
    t = (angle-x0)/tl.maximum(x1-x0, EPS)
    s = (u-y0)/tl.maximum(y1-y0, EPS)
    if CLAMP:
        t = tl.minimum(tl.maximum(t, 0.), 1.)
        s = tl.minimum(tl.maximum(s, 0.), 1.)
    if ROT_MAJOR:
        i0 = r0*A+a0; i1 = r0*A+a1; i2 = r1*A+a0; i3 = r1*A+a1
    else:
        i0 = a0*R+r0; i1 = a1*R+r0; i2 = a0*R+r1; i3 = a1*R+r1
    tl.store(GI+k*4, i0, k < N); tl.store(GI+k*4+1, i1, k < N)
    tl.store(GI+k*4+2, i2, k < N); tl.store(GI+k*4+3, i3, k < N)
    tl.store(GW+k*4, (1.-t)*(1.-s), k < N); tl.store(GW+k*4+1, (1.-t)*s, k < N)
    tl.store(GW+k*4+2, t*(1.-s), k < N); tl.store(GW+k*4+3, t*s, k < N)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def source_grid_backward(ROT, SCALE_AXIS, ANGLE, SCALE, GI, GRAD, GA, GS, N,
                         R: tl.constexpr, A: tl.constexpr, LOG: tl.constexpr,
                         ROT_MAJOR: tl.constexpr, EPS: tl.constexpr,
                         CLAMP: tl.constexpr, NEED_ANGLE: tl.constexpr,
                         NEED_SCALE: tl.constexpr, BLOCK: tl.constexpr):
    k = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    i = tl.load(GI+k*4, k < N, 0)
    r0 = i//A if ROT_MAJOR else i%R
    a0 = i%A if ROT_MAJOR else i//R
    x0 = tl.load(ROT+r0); x1 = tl.load(ROT+r0+1)
    y0 = tl.load(SCALE_AXIS+a0); y1 = tl.load(SCALE_AXIS+a0+1)
    scale = tl.load(SCALE+k, k < N, 1.)
    u = tl.log(1.+scale) if LOG else scale
    if LOG:y0 = tl.log(1.+y0); y1 = tl.log(1.+y1)
    dx = tl.maximum(x1-x0, EPS); dy = tl.maximum(y1-y0, EPS)
    tr = (tl.load(ANGLE+k, k < N, 0.)-x0)/dx
    sr = (u-y0)/dy
    t = tr; s = sr
    if CLAMP:
        t = tl.minimum(tl.maximum(t, 0.), 1.)
        s = tl.minimum(tl.maximum(s, 0.), 1.)
    g0 = tl.load(GRAD+k*4, k < N, 0.); g1 = tl.load(GRAD+k*4+1, k < N, 0.)
    g2 = tl.load(GRAD+k*4+2, k < N, 0.); g3 = tl.load(GRAD+k*4+3, k < N, 0.)
    if NEED_ANGLE:
        dr = ((g2-g0)*(1.-s)+(g3-g1)*s)/dx
        if CLAMP:dr = tl.where((tr >= 0.) & (tr <= 1.), dr, 0.)
        tl.store(GA+k, dr, k < N)
    if NEED_SCALE:
        ds = ((g1-g0)*(1.-t)+(g3-g2)*t)/dy
        if LOG:ds = ds/(1.+scale)
        if CLAMP:ds = tl.where((sr >= 0.) & (sr <= 1.), ds, 0.)
        tl.store(GS+k, ds, k < N)


@triton.jit
def source_forward(VERT, BARY, GI, GW, TIME, EVAL, EVEC, EMBED,
                   DECAY, PHI, NORM, SOURCE,
                   MODES: tl.constexpr, DIM: tl.constexpr,
                   LS0: tl.constexpr, LS1: tl.constexpr,
                   ES0: tl.constexpr, ES1: tl.constexpr, ES2: tl.constexpr,
                   IS0: tl.constexpr, IS1: tl.constexpr,
                   BM: tl.constexpr, BD: tl.constexpr, EPS: tl.constexpr):
    """Gather each live source once; no learned state survives this forward."""
    k = tl.program_id(0)
    mode = tl.arange(0, BM)
    grid = tl.arange(0, 4)
    gi = tl.load(GI+k*4+grid)
    gw = tl.load(GW+k*4+grid)
    v0 = tl.load(VERT+k*3); v1 = tl.load(VERT+k*3+1); v2 = tl.load(VERT+k*3+2)
    b0 = tl.load(BARY+k*3); b1 = tl.load(BARY+k*3+1); b2 = tl.load(BARY+k*3+2)
    index = gi[:, None]*ES0+mode[None, :]*ES2
    r0 = tl.load(EVEC+index+v0*ES1, mode[None, :] < MODES, 0.)
    r1 = tl.load(EVEC+index+v1*ES1, mode[None, :] < MODES, 0.)
    r2 = tl.load(EVEC+index+v2*ES1, mode[None, :] < MODES, 0.)
    phi = tl.sum(((r0*b0+r1*b1)+r2*b2)*gw[:, None], 0)
    ev = tl.load(EVAL+gi[:, None]*LS0+mode[None, :]*LS1, mode[None, :] < MODES, 0.)
    vals = tl.sum(ev*gw[:, None], 0)
    t = tl.load(TIME+k)
    decay = libdevice.exp(-vals*t)
    tl.store(PHI+k*MODES+mode, phi, mode < MODES)
    tl.store(DECAY+k*MODES+mode, decay, mode < MODES)
    tl.store(NORM+k, tl.sum(tl.where(mode < MODES, decay*phi*phi, 0.), 0)+EPS)
    dim = tl.arange(0, BD)
    z0 = tl.load(EMBED+v0*IS0+dim*IS1, dim < DIM, 0.)
    z1 = tl.load(EMBED+v1*IS0+dim*IS1, dim < DIM, 0.)
    z2 = tl.load(EMBED+v2*IS0+dim*IS1, dim < DIM, 0.)
    tl.store(SOURCE+k*DIM+dim, (z0*b0+z1*b1)+z2*b2, dim < DIM)


@triton.jit
def source_backward(VERT, BARY, GI, GW, TIME, EVAL, EVEC, EMBED,
                    DECAY, PHI, GD, GP, GN, GZ, GB, GGW, GT,
                    MODES: tl.constexpr, DIM: tl.constexpr,
                    LS0: tl.constexpr, LS1: tl.constexpr,
                    ES0: tl.constexpr, ES1: tl.constexpr, ES2: tl.constexpr,
                    IS0: tl.constexpr, IS1: tl.constexpr,
                    BM: tl.constexpr, BD: tl.constexpr,
                    HAS_D: tl.constexpr, HAS_P: tl.constexpr,
                    HAS_N: tl.constexpr, HAS_Z: tl.constexpr,
                    NEED_B: tl.constexpr, NEED_W: tl.constexpr, NEED_T: tl.constexpr):
    """Source-local adjoints have exclusive stores, without spectral atomics."""
    k = tl.program_id(0)
    mode = tl.arange(0, BM); grid = tl.arange(0, 4)
    gi = tl.load(GI+k*4+grid); gw = tl.load(GW+k*4+grid)
    v0 = tl.load(VERT+k*3); v1 = tl.load(VERT+k*3+1); v2 = tl.load(VERT+k*3+2)
    b0 = tl.load(BARY+k*3); b1 = tl.load(BARY+k*3+1); b2 = tl.load(BARY+k*3+2)
    d = tl.load(DECAY+k*MODES+mode, mode < MODES, 0.)
    p = tl.load(PHI+k*MODES+mode, mode < MODES, 0.)
    gd = tl.load(GD+k*MODES+mode, mode < MODES, 0.) if HAS_D else tl.full((BM,), 0., tl.float32)
    gp = tl.load(GP+k*MODES+mode, mode < MODES, 0.) if HAS_P else tl.full((BM,), 0., tl.float32)
    gn = tl.load(GN+k) if HAS_N else 0.
    gd = gd+gn*p*p
    gp = gp+gn*2*d*p
    t = tl.load(TIME+k)
    gl = -t*d*gd
    ev = tl.load(EVAL+gi[:, None]*LS0+mode[None, :]*LS1, mode[None, :] < MODES, 0.)
    index = gi[:, None]*ES0+mode[None, :]*ES2
    r0 = tl.load(EVEC+index+v0*ES1, mode[None, :] < MODES, 0.)
    r1 = tl.load(EVEC+index+v1*ES1, mode[None, :] < MODES, 0.)
    r2 = tl.load(EVEC+index+v2*ES1, mode[None, :] < MODES, 0.)
    if NEED_W:
        dw = tl.sum(gl[None, :]*ev+gp[None, :]*((r0*b0+r1*b1)+r2*b2), 1)
        tl.store(GGW+k*4+grid, dw)
    if NEED_T:
        vals = tl.sum(ev*gw[:, None], 0)
        tl.store(GT+k, tl.sum(-vals*d*gd, 0))
    if NEED_B:
        db0 = tl.sum(gp*tl.sum(r0*gw[:, None], 0), 0)
        db1 = tl.sum(gp*tl.sum(r1*gw[:, None], 0), 0)
        db2 = tl.sum(gp*tl.sum(r2*gw[:, None], 0), 0)
        if HAS_Z:
            dim = tl.arange(0, BD)
            gz = tl.load(GZ+k*DIM+dim, dim < DIM, 0.)
            z0 = tl.load(EMBED+v0*IS0+dim*IS1, dim < DIM, 0.)
            z1 = tl.load(EMBED+v1*IS0+dim*IS1, dim < DIM, 0.)
            z2 = tl.load(EMBED+v2*IS0+dim*IS1, dim < DIM, 0.)
            db0 += tl.sum(gz*z0, 0); db1 += tl.sum(gz*z1, 0); db2 += tl.sum(gz*z2, 0)
        tl.store(GB+k*3, db0); tl.store(GB+k*3+1, db1); tl.store(GB+k*3+2, db2)


@triton.jit
def coefficients_forward(VERT, KERNEL, GI, GW, DECAY, PHI, NORM, SOURCE,
                         EVEC, EMBED, HEAT, METRIC,
                         MODES: tl.constexpr, DIM: tl.constexpr,
                         ES0: tl.constexpr, ES1: tl.constexpr, ES2: tl.constexpr,
                         IS0: tl.constexpr, IS1: tl.constexpr,
                         BM: tl.constexpr, BD: tl.constexpr):
    entry = tl.program_id(0)
    k = tl.load(KERNEL+entry)
    mode = tl.arange(0, BM)
    grid = tl.arange(0, 4)
    gi = tl.load(GI+k*4+grid)
    gw = tl.load(GW+k*4+grid)
    decay = tl.load(DECAY+k*MODES+mode, mode < MODES, 0.)
    phi = tl.load(PHI+k*MODES+mode, mode < MODES, 0.)
    norm = tl.load(NORM+k)
    for c in tl.static_range(3):
        v = tl.load(VERT+entry*3+c)
        raw = tl.load(EVEC+gi[:, None]*ES0+v*ES1+mode[None, :]*ES2,
                      mode[None, :] < MODES, 0.)
        mixed = tl.sum(raw*gw[:, None], 0)
        h = tl.sum(decay*mixed*phi, 0)/norm
        tl.store(HEAT+entry*3+c, h)
    dim = tl.arange(0, BD)
    source = tl.load(SOURCE+k*DIM+dim, dim < DIM, 0.).to(tl.float64)
    v0 = tl.load(VERT+entry*3)
    v1 = tl.load(VERT+entry*3+1)
    v2 = tl.load(VERT+entry*3+2)
    d0 = tl.load(EMBED+v0*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
    d1 = tl.load(EMBED+v1*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
    d2 = tl.load(EMBED+v2*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
    tl.store(METRIC+entry*6, tl.sum(d0*d0, 0))
    tl.store(METRIC+entry*6+1, tl.sum(d1*d1, 0))
    tl.store(METRIC+entry*6+2, tl.sum(d2*d2, 0))
    tl.store(METRIC+entry*6+3, tl.sum(d0*d1, 0))
    tl.store(METRIC+entry*6+4, tl.sum(d0*d2, 0))
    tl.store(METRIC+entry*6+5, tl.sum(d1*d2, 0))


@triton.jit
def coefficients_backward(VERT, KERNEL, GI, GW, DECAY, PHI, NORM, SOURCE,
                          EVEC, EMBED, HEAT, GH, GM, GGW, GD, GP, GN, GS,
                          MODES: tl.constexpr, DIM: tl.constexpr,
                          ES0: tl.constexpr, ES1: tl.constexpr, ES2: tl.constexpr,
                          IS0: tl.constexpr, IS1: tl.constexpr,
                          BM: tl.constexpr, BD: tl.constexpr,
                          NEED_GW: tl.constexpr, NEED_D: tl.constexpr,
                          NEED_P: tl.constexpr, NEED_N: tl.constexpr,
                          NEED_S: tl.constexpr, HAS_GH: tl.constexpr,
                          HAS_GM: tl.constexpr):
    entry = tl.program_id(0)
    k = tl.load(KERNEL+entry)
    h0 = tl.load(GH+entry*3) if HAS_GH else 0.
    h1 = tl.load(GH+entry*3+1) if HAS_GH else 0.
    h2 = tl.load(GH+entry*3+2) if HAS_GH else 0.
    # Discarded/zero-adjoint entries require no spectral gather or recompute.
    # This branch is uniform for the entire program.
    if (h0 != 0.) | (h1 != 0.) | (h2 != 0.):
        mode = tl.arange(0, BM)
        grid = tl.arange(0, 4)
        gi = tl.load(GI+k*4+grid)
        gw = tl.load(GW+k*4+grid)
        decay = tl.load(DECAY+k*MODES+mode, mode < MODES, 0.)
        phi = tl.load(PHI+k*MODES+mode, mode < MODES, 0.)
        norm = tl.load(NORM+k)
        gd = tl.full((BM,), 0., tl.float32)
        gp = tl.full((BM,), 0., tl.float32)
        ggrid = tl.full((4,), 0., tl.float32)
        gnorm = tl.full((), 0., tl.float32)
        for c in tl.static_range(3):
            gh = tl.load(GH+entry*3+c)
            v = tl.load(VERT+entry*3+c)
            raw = tl.load(EVEC+gi[:, None]*ES0+v*ES1+mode[None, :]*ES2,
                          mode[None, :] < MODES, 0.)
            mixed = tl.sum(raw*gw[:, None], 0)
            scale = gh/norm
            gd += scale*mixed*phi
            gp += scale*decay*mixed
            ggrid += tl.sum(raw*(scale*decay*phi)[None, :], 1)
            gnorm -= gh*tl.load(HEAT+entry*3+c)/norm
        if NEED_D:
            tl.atomic_add(GD+k*MODES+mode, gd, mode < MODES)
        if NEED_P:
            tl.atomic_add(GP+k*MODES+mode, gp, mode < MODES)
        if NEED_GW:
            tl.atomic_add(GGW+k*4+grid, ggrid)
        if NEED_N:
            tl.atomic_add(GN+k, gnorm)
    if NEED_S and HAS_GM:
        lane = tl.arange(0, 8)
        gm = tl.load(GM+entry*6+lane, lane < 6, 0.)
        if tl.sum(tl.abs(gm), 0) != 0.:
            dim = tl.arange(0, BD)
            source = tl.load(SOURCE+k*DIM+dim, dim < DIM, 0.).to(tl.float64)
            v0 = tl.load(VERT+entry*3)
            v1 = tl.load(VERT+entry*3+1)
            v2 = tl.load(VERT+entry*3+2)
            d0 = tl.load(EMBED+v0*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
            d1 = tl.load(EMBED+v1*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
            d2 = tl.load(EMBED+v2*IS0+dim*IS1, dim < DIM, 0.).to(tl.float64)-source
            g0 = tl.load(GM+entry*6).to(tl.float64)
            g1 = tl.load(GM+entry*6+1).to(tl.float64)
            g2 = tl.load(GM+entry*6+2).to(tl.float64)
            g3 = tl.load(GM+entry*6+3).to(tl.float64)
            g4 = tl.load(GM+entry*6+4).to(tl.float64)
            g5 = tl.load(GM+entry*6+5).to(tl.float64)
            ds = -(2*g0*d0+2*g1*d1+2*g2*d2+g3*(d0+d1)+g4*(d0+d2)+g5*(d1+d2))
            tl.atomic_add(GS+k*DIM+dim, ds.to(tl.float32), dim < DIM)


@triton.jit
def coefficients_forward_parallel(VERT,KERNEL,GI,GW,DECAY,PHI,NORM,SOURCE,
                                  EVEC,EMBED,HEAT,METRIC,
                                  MODES:tl.constexpr,DIM:tl.constexpr,
                                  ES0:tl.constexpr,ES1:tl.constexpr,ES2:tl.constexpr,
                                  IS0:tl.constexpr,IS1:tl.constexpr,
                                  BM:tl.constexpr,BD:tl.constexpr):
    """Evaluate independent corners/Gram entries in parallel, with unchanged reductions."""
    entry=tl.program_id(0);k=tl.load(KERNEL+entry)
    mode=tl.arange(0,BM);grid=tl.arange(0,4);corner=tl.arange(0,4)
    gi=tl.load(GI+k*4+grid);gw=tl.load(GW+k*4+grid)
    decay=tl.load(DECAY+k*MODES+mode,mode<MODES,0.)
    phi=tl.load(PHI+k*MODES+mode,mode<MODES,0.)
    v=tl.load(VERT+entry*3+corner,corner<3,0)
    raw=tl.load(EVEC+gi[None,:,None]*ES0+v[:,None,None]*ES1+mode[None,None,:]*ES2,
                (corner[:,None,None]<3)&(mode[None,None,:]<MODES),0.)
    mixed=tl.sum(raw*gw[None,:,None],1)
    heat=tl.sum(decay[None,:]*mixed*phi[None,:],1)/tl.load(NORM+k)
    tl.store(HEAT+entry*3+corner,heat,corner<3)
    dim=tl.arange(0,BD)
    source=tl.load(SOURCE+k*DIM+dim,dim<DIM,0.).to(tl.float64)
    v0=tl.load(VERT+entry*3);v1=tl.load(VERT+entry*3+1);v2=tl.load(VERT+entry*3+2)
    d0=tl.load(EMBED+v0*IS0+dim*IS1,dim<DIM,0.).to(tl.float64)-source
    d1=tl.load(EMBED+v1*IS0+dim*IS1,dim<DIM,0.).to(tl.float64)-source
    d2=tl.load(EMBED+v2*IS0+dim*IS1,dim<DIM,0.).to(tl.float64)-source
    pair=tl.arange(0,8)
    left=tl.where(((pair==0)|(pair==3)|(pair==4))[:,None],d0[None,:],
                  tl.where(((pair==1)|(pair==5))[:,None],d1[None,:],d2[None,:]))
    right=tl.where((pair==0)[:,None],d0[None,:],
                   tl.where(((pair==1)|(pair==3))[:,None],d1[None,:],d2[None,:]))
    metric=tl.sum(left*right,1)
    tl.store(METRIC+entry*6+pair,metric,pair<6)
