"""Reconstruct HKS RGB and differentiate selected contributions.

Rendering/training uses these RGB, color, heat and filter adjoints with runtime
pair bounds to avoid recompiling when the supported pair count changes.
Main API: reconstruct, reconstruct_backward, emit_backward."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit(do_not_specialize=['N', 'DX', 'DY', 'DZ', 'AMBIENT', 'INTENSITY', 'ALPHA2', 'F0'],
            do_not_specialize_on_alignment=['N'])
def material_lighting_forward(ALBEDO, NORMAL, VIEW, RGB, MULTIPLIER, N,
                              DX, DY, DZ, AMBIENT, INTENSITY, ALPHA2, F0,
                              GGX: tl.constexpr, SAVE: tl.constexpr, BLOCK: tl.constexpr):
    """One pixel per lane; fixed light/material settings remain runtime inputs."""
    p = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = p < N
    nx = tl.load(NORMAL+p*3, valid, 0.)
    ny = tl.load(NORMAL+p*3+1, valid, 0.)
    nz = tl.load(NORMAL+p*3+2, valid, 0.)
    vx = tl.load(VIEW+p*3, valid, 0.)
    vy = tl.load(VIEW+p*3+1, valid, 0.)
    vz = tl.load(VIEW+p*3+2, valid, 0.)
    # Match Torch CUDA's three-component tree reduction: (x+z)+y.
    nn = tl.maximum(libdevice.sqrt_rn((nx*nx+nz*nz)+ny*ny), 1e-6)
    vn = tl.maximum(libdevice.sqrt_rn((vx*vx+vz*vz)+vy*vy), 1e-6)
    ln = tl.maximum(libdevice.sqrt_rn((DX*DX+DZ*DZ)+DY*DY), 1e-6)
    nx, ny, nz = tl.div_rn(nx, nn), tl.div_rn(ny, nn), tl.div_rn(nz, nn)
    vx, vy, vz = tl.div_rn(vx, vn), tl.div_rn(vy, vn), tl.div_rn(vz, vn)
    lx, ly, lz = tl.div_rn(DX, ln), tl.div_rn(DY, ln), tl.div_rn(DZ, ln)
    nl = tl.minimum(tl.maximum((nx*lx+nz*lz)+ny*ly, 0.), 1.)
    nv = tl.minimum(tl.maximum((nx*vx+nz*vz)+ny*vy, 0.), 1.)
    fresnel = tl.full((BLOCK,), 0., tl.float32)
    highlight = tl.full((BLOCK,), 0., tl.float32)
    if GGX:
        hx, hy, hz = lx+vx, ly+vy, lz+vz
        hn = tl.maximum(libdevice.sqrt_rn((hx*hx+hz*hz)+hy*hy), 1e-6)
        hx, hy, hz = tl.div_rn(hx, hn), tl.div_rn(hy, hn), tl.div_rn(hz, hn)
        nh = tl.minimum(tl.maximum((nx*hx+nz*hz)+ny*hy, 0.), 1.)
        vh = tl.minimum(tl.maximum((vx*hx+vz*hz)+vy*hy, 0.), 1.)
        denominator = nh*nh*(ALPHA2-1.)+1.
        distribution = tl.div_rn(ALPHA2, 3.141592653589793*(denominator*denominator))
        gl = tl.div_rn(2.*nl, nl+libdevice.sqrt_rn(ALPHA2+(1.-ALPHA2)*(nl*nl)))
        gv = tl.div_rn(2.*nv, nv+libdevice.sqrt_rn(ALPHA2+(1.-ALPHA2)*(nv*nv)))
        fresnel = F0+(1.-F0)*libdevice.pow(1.-vh, 5.)
        highlight = tl.div_rn(distribution*gl*gv*fresnel, 4.*tl.maximum(nv, 1e-6))
    active = (nl > 0.) & (nv > 0.)
    factor = AMBIENT+tl.where(active, INTENSITY*((1.-fresnel)/3.141592653589793)*nl, 0.)
    if SAVE:
        tl.store(MULTIPLIER+p, factor, valid)
    for c in tl.static_range(3):
        a = tl.load(ALBEDO+p*3+c, valid, 0.)
        linear = tl.where(a <= .04045, a/12.92,
                          libdevice.pow(tl.maximum(a+.055, 0.)/1.055, 2.4))
        # Match the tensor oracle's forward operation order; factor is only its adjoint.
        diffuse = linear/3.141592653589793
        if GGX:
            diffuse = diffuse*(1.-fresnel)
        result = AMBIENT*linear+INTENSITY*tl.where(active, diffuse*nl+highlight, 0.)
        tl.store(RGB+p*3+c, result, valid)


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def material_lighting_backward(ALBEDO, MULTIPLIER, GRAD, RESULT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    valid = i < N*3
    a = tl.load(ALBEDO+i, valid, 0.)
    factor = tl.load(MULTIPLIER+i//3, valid, 0.)
    derivative = tl.where(a <= .04045, 1./12.92,
        (2.4/1.055)*libdevice.pow(tl.maximum(a+.055, 0.)/1.055, 1.4))
    tl.store(RESULT+i, tl.load(GRAD+i, valid, 0.)*factor*derivative, valid)

@triton.jit(do_not_specialize=["PAIRS", "N"], do_not_specialize_on_alignment=["PAIRS", "N"])
def reconstruct(STATE_ENTRY,STATE_WEIGHT,KERNELS,COLORS,BASE,FACE_IDS,RGB,
                N,PAIRS,S:tl.constexpr):
    pixel=tl.program_id(0);lane=tl.arange(0,S)
    entry=tl.load(STATE_ENTRY+pixel*S+lane);w=tl.load(STATE_WEIGHT+pixel*S+lane)
    valid=(entry>=0)&(entry<PAIRS)&(w>0)
    source=tl.load(KERNELS+entry,valid,0)
    den=tl.maximum(tl.sum(w,0),1.)
    foreground=tl.load(FACE_IDS+pixel)>=0
    for c in tl.static_range(3):
        color=tl.load(COLORS+source*3+c,valid,0.)
        value=tl.load(BASE+c)+tl.sum(w*color,0)/den
        tl.store(RGB+pixel*3+c,tl.where(foreground,tl.minimum(tl.maximum(value,0.),1.),0.))


@triton.jit(do_not_specialize=["PAIRS"], do_not_specialize_on_alignment=["PAIRS"])
def reconstruct_backward(SE, SW, KERNELS, COLORS, BASE, IDS, GRAD,
                         GW, GC, GB, PAIRS, S: tl.constexpr,
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


@triton.jit(do_not_specialize=["PAIRS"], do_not_specialize_on_alignment=["PAIRS"])
def emit_backward(SE, GW, BARY, KERNELS, HEAT, METRIC, EPS, SHARP,
                  GH, GM, GE, GS, PAIRS, S: tl.constexpr,
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

