"""Autograd wrappers for live spectral sources and three-corner heat/Gram coefficients.

Corner heat evaluation and first-order source gradients during rendering/training.
Main API: fused_source_grid, fused_source_state, fused_coefficients.
All learned values are refreshed each forward; only fixed spectral assets are reused."""
import torch
from torch.autograd.function import once_differentiable
import triton
from hktex_splatting.triton import spectral_heat_coefficients as coefficient_gpu


class _SourceGrid(torch.autograd.Function):
    """Preserve native bilinear cell routing and floating angle/scale derivatives."""
    @staticmethod
    def forward(ctx, rot, axis, angle, scale, logarithmic, rot_major, eps, clamp):
        n = len(angle)
        gi = torch.empty((n, 4), dtype=torch.int64, device=angle.device)
        gw = angle.new_empty((n, 4))
        constants = (len(rot), len(axis), logarithmic, rot_major, eps, clamp)
        if n:
            coefficient_gpu.source_grid_forward[(triton.cdiv(n, 128),)](
                rot, axis, angle, scale, gi, gw, n, *constants, 128,
                num_warps=4, enable_fp_fusion=False)
        ctx.mark_non_differentiable(gi)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(rot, axis, angle, scale, gi)
        ctx.constants = constants
        return gi, gw

    @staticmethod
    @once_differentiable
    def backward(ctx, _, grad):
        if grad is None:
            return (None,)*8
        rot, axis, angle, scale, gi = ctx.saved_tensors
        needs = ctx.needs_input_grad[2:4]
        ga = torch.empty_like(angle) if needs[0] else None
        gs = torch.empty_like(scale) if needs[1] else None
        if grad is not None and len(angle) and any(needs):
            dummy = angle.new_empty(1)
            coefficient_gpu.source_grid_backward[(triton.cdiv(len(angle), 128),)](
                rot, axis, angle, scale, gi, grad.contiguous(),
                ga if ga is not None else dummy, gs if gs is not None else dummy,
                len(angle), *ctx.constants, *needs, 128, num_warps=4, enable_fp_fusion=False)
        return None, None, ga, gs, None, None, None, None


def fused_source_grid(rot, axis, angle, scale, *, scale_map='log1p',
                      grid_layout='rot_major', eps=1e-10, clamp_unit=True):
    """Four native bilinear orientation/anisotropy grid weights for current sources."""
    if scale_map not in ('log1p', 'identity') or grid_layout not in ('rot_major', 'scale_major'):
        raise ValueError('Unsupported source grid mapping/layout')
    if (rot.ndim != 1 or axis.ndim != 1 or min(len(rot), len(axis)) < 2 or
            angle.ndim != 1 or angle.shape != scale.shape):
        raise ValueError('Invalid source grid axes or angle/scale shapes')
    if any(t.dtype != torch.float32 or t.device != angle.device for t in (rot, axis, angle, scale)):
        raise ValueError('Source grid requires float32 tensors on one device')
    if rot.requires_grad or axis.requires_grad:
        raise ValueError('Spectral grid axes must remain fixed')
    return _SourceGrid.apply(*(t.contiguous() for t in (rot, axis, angle, scale)),
        scale_map == 'log1p', grid_layout == 'rot_major', eps, clamp_unit)


class _SourceState(torch.autograd.Function):
    """Fuse current source gathers and their first-order floating-input adjoints."""
    @staticmethod
    def forward(ctx, vertices, bary, gi, gw, time, evals, evecs, embedding, eps):
        n, modes, dim = len(vertices), evecs.shape[-1], embedding.shape[-1]
        decay = bary.new_empty((n, modes)); phi = torch.empty_like(decay)
        norm = bary.new_empty(n); source = bary.new_empty((n, dim))
        constants = (modes, dim, *evals.stride(), *evecs.stride(), *embedding.stride(),
                     triton.next_power_of_2(modes), triton.next_power_of_2(dim))
        if n:
            coefficient_gpu.source_forward[(n,)](vertices, bary, gi, gw, time,
                evals, evecs, embedding, decay, phi, norm, source, *constants, eps,
                num_warps=4, enable_fp_fusion=False)
        ctx.set_materialize_grads(False)
        bary_grad, grid_grad, time_grad = (ctx.needs_input_grad[i] for i in (1, 3, 4))
        nondifferentiable = []
        if not (grid_grad or time_grad):nondifferentiable.append(decay)
        if not (bary_grad or grid_grad):nondifferentiable.append(phi)
        if not (bary_grad or grid_grad or time_grad):nondifferentiable.append(norm)
        if not bary_grad:nondifferentiable.append(source)
        if nondifferentiable:ctx.mark_non_differentiable(*nondifferentiable)
        ctx.save_for_backward(vertices, bary, gi, gw, time, evals, evecs, embedding, decay, phi)
        ctx.constants = constants
        return decay, phi, norm, source

    @staticmethod
    @once_differentiable
    def backward(ctx, gd, gp, gn, gz):
        vertices, bary, gi, gw, time, evals, evecs, embedding, decay, phi = ctx.saved_tensors
        needs = tuple(ctx.needs_input_grad[i] for i in (1, 3, 4))
        grads = [torch.empty_like(t) if need else None for t, need in zip((bary, gw, time), needs)]
        dummy = bary.new_empty(1)
        incoming = tuple(g.contiguous() if g is not None else dummy for g in (gd, gp, gn, gz))
        if len(vertices) and any(needs):
            coefficient_gpu.source_backward[(len(vertices),)](vertices, bary, gi, gw, time,
                evals, evecs, embedding, decay, phi, *incoming,
                *(g if g is not None else dummy for g in grads), *ctx.constants,
                *(g is not None for g in (gd, gp, gn, gz)), *needs,
                num_warps=4, enable_fp_fusion=False)
        return None, grads[0], None, grads[1], grads[2], None, None, None, None


def fused_source_state(vertices, bary, gi, gw, time, eigen):
    """Fresh decay/phi/normalizer/embedding with gradients to barycentrics, grid weights and time.

    Fixed spectral assets are borrowed, never copied or differentiated. Integer
    grid/face selection follows the existing native policy. Higher derivatives
    are unsupported, like the downstream fused coefficients.
    """
    n = len(vertices)
    evals, evecs, embedding = eigen._eigen_val_knn, eigen._eigen_vec_knn, eigen._iso_embeddings
    if (vertices.shape != (n, 3) or bary.shape != (n, 3) or gi.shape != (n, 4) or
            gw.shape != (n, 4) or time.shape != (n,)):
        raise ValueError('Invalid live source geometry/grid/time shapes')
    if any(t.dtype != torch.float32 or t.device != bary.device for t in (bary, gw, time, evals, evecs, embedding)):
        raise ValueError('Fused source preparation requires float32 tensors on one device')
    if any(t.requires_grad for t in (evals, evecs, embedding)):
        raise ValueError('Spectral assets must remain fixed')
    if any(t.dtype not in (torch.int32, torch.int64) or t.device != bary.device for t in (vertices, gi)):
        raise ValueError('Source/grid indices must be integer tensors on the source device')
    return _SourceState.apply(vertices.contiguous(), bary.contiguous(), gi.contiguous(), gw.contiguous(),
        time.contiguous(), evals, evecs, embedding, eigen.knn_heat.config.eps)


class _Coefficients(torch.autograd.Function):
    """Connect fused corner heat/Gram kernels to first-order source-parameter gradients."""
    @staticmethod
    def forward(ctx, vertices, kernels, gi, gw, decay, phi, norm, source, evec, embedding, parallel):
        pairs = kernels.numel()
        heat = phi.new_empty((pairs, 3))
        metric = phi.new_empty((pairs, 6))
        modes, dim = phi.shape[1], source.shape[1]
        constants = (modes, dim, *evec.stride(), *embedding.stride(),
                     triton.next_power_of_2(modes), triton.next_power_of_2(dim))
        if pairs:
            forward=coefficient_gpu.coefficients_forward_parallel if parallel else coefficient_gpu.coefficients_forward
            forward[(pairs,)](
                vertices, kernels, gi, gw, decay, phi, norm, source,
                evec, embedding, heat, metric, *constants,
                num_warps=4, enable_fp_fusion=False)
        ctx.set_materialize_grads(False)
        nondifferentiable = []
        if not any(ctx.needs_input_grad[3:7]):
            nondifferentiable.append(heat)
        if not ctx.needs_input_grad[7]:
            nondifferentiable.append(metric)
        if nondifferentiable:
            ctx.mark_non_differentiable(*nondifferentiable)
        ctx.save_for_backward(vertices, kernels, gi, gw, decay, phi, norm, source, evec, embedding, heat)
        ctx.constants = constants
        return heat, metric

    @staticmethod
    @once_differentiable
    def backward(ctx, gh, gm):
        vertices, kernels, gi, gw, decay, phi, norm, source, evec, embedding, heat = ctx.saved_tensors
        tensors = (gw, decay, phi, norm, source)
        needs = tuple(needed and (gh is not None if i < 4 else gm is not None)
                      for i, needed in enumerate(ctx.needs_input_grad[3:8]))
        gradients = [torch.zeros_like(t) if needed else None for t, needed in zip(tensors, needs)]
        dummy = heat.new_empty(1)
        pointers = [g if g is not None else dummy for g in gradients]
        gh_pointer = gh.contiguous() if gh is not None else dummy
        gm_pointer = gm.contiguous() if gm is not None else dummy
        if kernels.numel() and any(needs) and (gh is not None or gm is not None):
            with torch.profiler.record_function('spectral::fused_backward'):
                coefficient_gpu.coefficients_backward[(kernels.numel(),)](
                    vertices, kernels, gi, gw, decay, phi, norm, source, evec, embedding,
                    heat, gh_pointer, gm_pointer, *pointers, *ctx.constants, *needs,
                    gh is not None, gm is not None, num_warps=4, enable_fp_fusion=False)
        return None, None, None, *gradients, None, None, None


def fused_coefficients(vertices, kernels, state, eigen, *, layout='serial'):
    """Exact same intrinsic formulation; float32 outputs, first-order gradients.

    Floating-point reduction order can differ from eager PyTorch. The spectral
    archive/embedding are fixed, as in native HKTex. Their gradients and higher
    derivatives are explicitly unsupported.
    """
    if layout not in ('serial','parallel'):raise ValueError('Unknown coefficient layout')
    tensors = (state.grid_weights, state.decay, state.phi, state.normalizer,
               state.embedding, eigen._eigen_vec_knn, eigen._iso_embeddings)
    if any(t.dtype != torch.float32 or t.device != state.phi.device for t in tensors):
        raise ValueError('Fused coefficients require float32 tensors on one device')
    if state.phi.ndim != 2 or state.embedding.ndim != 2 or not state.phi.shape[1] or not state.embedding.shape[1]:
        raise ValueError('Expected nonempty spectral and embedding dimensions')
    for t in (vertices,kernels,state.grid_ids):
        if t.device != state.phi.device or t.dtype not in (torch.int32,torch.int64):
            raise ValueError('Topology/grid indices must be integers on the source-state device')
    if (eigen._eigen_vec_knn.ndim != 3 or eigen._eigen_vec_knn.shape[2] != state.phi.shape[1]
        or eigen._iso_embeddings.ndim != 2 or eigen._iso_embeddings.shape[1] != state.embedding.shape[1]):
        raise ValueError('Spectral archive/embedding dimensions differ from the live state')
    if eigen._eigen_vec_knn.requires_grad or eigen._iso_embeddings.requires_grad:
        raise ValueError('Spectral archive/mesh embedding must be fixed')
    if state.grid_ids.shape != (state.phi.shape[0], 4) or state.grid_weights.shape != state.grid_ids.shape:
        raise ValueError('Expected the native four-grid source interpolation')
    if state.phi.shape != state.decay.shape or state.normalizer.shape != (state.phi.shape[0],):
        raise ValueError('Invalid live source-state shapes')
    if vertices.shape != (kernels.numel(), 3) or state.embedding.shape[0] != state.phi.shape[0]:
        raise ValueError('Invalid coefficient/source geometry')
    return _Coefficients.apply(vertices.contiguous(), kernels.contiguous(), state.grid_ids.contiguous(),
        *(t.contiguous() for t in tensors[:5]), eigen._eigen_vec_knn, eigen._iso_embeddings, layout=='parallel')
