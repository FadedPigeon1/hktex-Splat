"""Triton corner heat/Gram contractions and source-state adjoints.

GPU corner heat evaluation and source-state adjoints for rendering/training.
Main API: coefficients_forward, coefficients_backward."""
import triton
import triton.language as tl


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
