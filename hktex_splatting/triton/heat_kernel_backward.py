"""Selected contribution and RGB adjoints with compile-time pair counts.

GPU backward propagation with fixed visibility and selected contribution IDs.
Main API: reconstruct_backward, emit_backward."""
import triton
import triton.language as tl


@triton.jit
def reconstruct_backward(SE, SW, KERNELS, COLORS, BASE, IDS, GRAD,
                         GW, GC, GB, PAIRS: tl.constexpr, S: tl.constexpr,
                         NEED_COLORS: tl.constexpr, NEED_BASE: tl.constexpr,
                         NEED_WEIGHT: tl.constexpr):
    p = tl.program_id(0)
    lane = tl.arange(0, S)
    entry = tl.load(SE+p*S+lane)
    w = tl.load(SW+p*S+lane)
    valid = (entry >= 0) & (entry < PAIRS) & (w > 0)
    k = tl.load(KERNELS+entry, valid, 0)
    total = tl.sum(w, 0)
    den = tl.maximum(total, 1.)
    foreground = tl.load(IDS+p) >= 0
    dw = tl.full((S,), 0., tl.float32)
    for c in tl.static_range(3):
        color = tl.load(COLORS+k*3+c, valid, 0.)
        numerator = tl.sum(w*color, 0)
        pre = tl.load(BASE+c)+numerator/den
        # Inclusive endpoints match torch.clamp's chosen subgradient.
        g = tl.load(GRAD+p*3+c)*foreground*(pre >= 0.)*(pre <= 1.)
        if NEED_BASE:
            tl.atomic_add(GB+c, g)
        if NEED_COLORS:
            tl.atomic_add(GC+k*3+c, g*w/den, valid)
        if NEED_WEIGHT:
            dw += g*(color/den-tl.where(total >= 1., numerator/(den*den), 0.))
    if NEED_WEIGHT:
        tl.store(GW+p*S+lane, tl.where(valid, dw, 0.))


@triton.jit
def emit_backward(SE, GW, BARY, KERNELS, HEAT, METRIC, EPS, SHARP,
                  GH, GM, GE, GS, PAIRS: tl.constexpr, S: tl.constexpr,
                  SIGMA: tl.constexpr, DIST_EPS: tl.constexpr, POWER: tl.constexpr,
                  NEED_HEAT: tl.constexpr, NEED_METRIC: tl.constexpr,
                  NEED_EPS: tl.constexpr, NEED_SHARP: tl.constexpr):
    p = tl.program_id(0)
    lane = tl.arange(0, S)
    entry = tl.load(SE+p*S+lane)
    upstream = tl.load(GW+p*S+lane)
    valid = (entry >= 0) & (entry < PAIRS) & (upstream != 0.)
    k = tl.load(KERNELS+entry, valid, 0)
    b0 = tl.load(BARY+p*3)
    b1 = tl.load(BARY+p*3+1)
    b2 = tl.load(BARY+p*3+2)
    h = (tl.load(HEAT+entry*3, valid, 0.)*b0 +
         tl.load(HEAT+entry*3+1, valid, 0.)*b1 +
         tl.load(HEAT+entry*3+2, valid, 0.)*b2)
    g0 = tl.load(METRIC+entry*6, valid, 0.)
    g1 = tl.load(METRIC+entry*6+1, valid, 0.)
    g2 = tl.load(METRIC+entry*6+2, valid, 0.)
    g3 = tl.load(METRIC+entry*6+3, valid, 0.)
    g4 = tl.load(METRIC+entry*6+4, valid, 0.)
    g5 = tl.load(METRIC+entry*6+5, valid, 0.)
    d = g0*b0*b0+g1*b1*b1+g2*b2*b2+2*(g3*b0*b1+g4*b0*b2+g5*b1*b2)
    attenuation = tl.exp(-(tl.maximum(d, 0.)+DIST_EPS)/(2*SIGMA*SIGMA))
    response = h*attenuation
    r = response  # Training API explicitly requires the audited POWER=1.
    e = tl.load(EPS+k, valid, 0.)
    s = tl.load(SHARP+k, valid, 1.)
    y = 1./(1.+tl.exp(-s*(r-e)))
    y0 = 1./(1.+tl.exp(s*e))
    y1 = 1./(1.+tl.exp(-s*(1.-e)))
    den = y1-y0+1e-8
    num = y-y0
    raw = num/den
    u = tl.where(valid & (raw >= 0.) & (raw <= 1.), upstream, 0.)
    a = y*(1.-y)
    a0 = y0*(1.-y0)
    a1 = y1*(1.-y1)
    dr = u*s*a/den
    de = u*((-s*a+s*a0)/den-num*(-s*a1+s*a0)/(den*den))
    ds = u*(((r-e)*a+e*a0)/den-num*((1.-e)*a1+e*a0)/(den*den))
    dh = dr*attenuation
    dd = tl.where(d >= 0., -dr*response/(2*SIGMA*SIGMA), 0.)
    if NEED_HEAT:
        tl.atomic_add(GH+entry*3, dh*b0, valid)
        tl.atomic_add(GH+entry*3+1, dh*b1, valid)
        tl.atomic_add(GH+entry*3+2, dh*b2, valid)
    if NEED_METRIC:
        tl.atomic_add(GM+entry*6, dd*b0*b0, valid)
        tl.atomic_add(GM+entry*6+1, dd*b1*b1, valid)
        tl.atomic_add(GM+entry*6+2, dd*b2*b2, valid)
        tl.atomic_add(GM+entry*6+3, dd*2*b0*b1, valid)
        tl.atomic_add(GM+entry*6+4, dd*2*b0*b2, valid)
        tl.atomic_add(GM+entry*6+5, dd*2*b1*b2, valid)
    if NEED_EPS:
        tl.atomic_add(GE+k, de, valid)
    if NEED_SHARP:
        tl.atomic_add(GS+k, ds, valid)
