"""Shared contribution emission, strongest-weight selection and reference RGB.

HKS uses merge for top-30 selection after its tile emitter. The generic emit and
reconstruct kernels support the retained correctness reference.
Main API: emit, merge, reconstruct."""
import triton
import triton.language as tl


@triton.jit
def emit(ENTRIES,PATCHES,RECORD_OFF,PATCH_OFF,PIXELS,BARY,KERNELS,HEAT,METRIC,
         EPS,SHARP,OUT_PIXEL,OUT_ENTRY,OUT_WEIGHT,START,
         ORIGIN,BLOCK:tl.constexpr,SIGMA:tl.constexpr,
         DIST_EPS:tl.constexpr,POWER:tl.constexpr,DISABLED):
    task=START+tl.program_id(0)
    entry=tl.load(ENTRIES+task);patch=tl.load(PATCHES+task)
    begin=tl.load(PATCH_OFF+patch);end=tl.load(PATCH_OFF+patch+1)
    lane=tl.arange(0,BLOCK);valid=begin+lane<end
    pixel=tl.load(PIXELS+begin+lane,valid,0)
    b0=tl.load(BARY+pixel*3,valid,0.);b1=tl.load(BARY+pixel*3+1,valid,0.)
    b2=tl.load(BARY+pixel*3+2,valid,0.)
    h=tl.load(HEAT+entry*3)*b0+tl.load(HEAT+entry*3+1)*b1+tl.load(HEAT+entry*3+2)*b2
    g0=tl.load(METRIC+entry*6);g1=tl.load(METRIC+entry*6+1);g2=tl.load(METRIC+entry*6+2)
    g3=tl.load(METRIC+entry*6+3);g4=tl.load(METRIC+entry*6+4);g5=tl.load(METRIC+entry*6+5)
    d=tl.maximum(g0*b0*b0+g1*b1*b1+g2*b2*b2+2*(g3*b0*b1+g4*b0*b2+g5*b1*b2),0.)
    r=h*tl.exp(-(d+DIST_EPS)/(2*SIGMA*SIGMA))
    if POWER!=1:r=tl.pow(r,POWER)
    source=tl.load(KERNELS+entry);e=tl.load(EPS+source);s=tl.load(SHARP+source)
    y=1./(1.+tl.exp(-s*(r-e)));y0=1./(1.+tl.exp(s*e));y1=1./(1.+tl.exp(-s*(1.-e)))
    weight=tl.minimum(tl.maximum((y-y0)/(y1-y0+1e-8),0.),1.)
    weight=tl.where(source==DISABLED,0.,weight)
    out=tl.load(RECORD_OFF+task)-ORIGIN+lane
    tl.store(OUT_PIXEL+out,pixel,valid);tl.store(OUT_ENTRY+out,entry,valid)
    tl.store(OUT_WEIGHT+out,weight,valid)


@triton.jit(do_not_specialize=['ORIGIN', 'N'], do_not_specialize_on_alignment=['N'])
def merge(SEGMENT_OFF,SORTED_INDEX,REC_ENTRY,REC_WEIGHT,ACTIVE,STATE_ENTRY,STATE_WEIGHT,
          N,WIDTH:tl.constexpr,INNER:tl.constexpr,S:tl.constexpr,
          ORIGIN=0,RECORD_LIMIT:tl.constexpr=0,IDENTITY_INDEX:tl.constexpr=False):
    i=tl.program_id(0);pixel=tl.load(ACTIVE+i)
    begin=tl.load(SEGMENT_OFF+pixel)+ORIGIN;end=tl.load(SEGMENT_OFF+pixel+1)
    lane=tl.arange(0,WIDTH)
    previous=lane<S
    old_entry=tl.load(STATE_ENTRY+pixel*S+lane,previous,2147483647)
    old_weight=tl.load(STATE_WEIGHT+pixel*S+lane,previous,0.)
    new_slot=lane-S;valid=(new_slot>=0)&(begin+new_slot<end)
    if RECORD_LIMIT>0:valid=valid&(new_slot<RECORD_LIMIT)
    # HKS records are already pixel segmented: their permutation is identity.
    # Reference renderers retain the explicit permutation by default.
    if IDENTITY_INDEX:
        record=begin+new_slot
    else:
        record=tl.load(SORTED_INDEX+begin+new_slot,valid,0)
    new_entry=tl.load(REC_ENTRY+record,valid,2147483647)
    new_weight=tl.load(REC_WEIGHT+record,valid,0.)
    entry=tl.where(previous,old_entry,new_entry)
    weight=tl.where(previous,old_weight,new_weight)
    # Nonnegative float weights sort by IEEE bits. Low bits impose the same
    # deterministic support-order ties as the old shader, across batches too.
    key=(weight.to(tl.uint32,bitcast=True).to(tl.uint64)<<32)|(
         0xffffffff-entry.to(tl.uint32)).to(tl.uint64)
    ordered=tl.sort(key,descending=True)
    selected=lane<INNER
    picked_entry=(0xffffffff-(ordered&0xffffffff).to(tl.uint32)).to(tl.int32)
    picked_weight=(ordered>>32).to(tl.uint32).to(tl.float32,bitcast=True)
    tl.store(STATE_ENTRY+pixel*S+lane,tl.where(selected,picked_entry,2147483647),lane<S)
    tl.store(STATE_WEIGHT+pixel*S+lane,tl.where(selected,picked_weight,0.),lane<S)


@triton.jit
def reconstruct(STATE_ENTRY,STATE_WEIGHT,KERNELS,COLORS,BASE,FACE_IDS,RGB,
                N:tl.constexpr,PAIRS:tl.constexpr,S:tl.constexpr):
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
