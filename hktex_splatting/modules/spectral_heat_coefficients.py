"""Autograd wrapper for fused spectral heat and Gram coefficients.

Corner heat evaluation and first-order source gradients during rendering/training.
Main API: _Coefficients, fused_coefficients."""
import torch
from torch.autograd.function import once_differentiable
import triton
from hktex_splatting.triton import spectral_heat_coefficients as coefficient_gpu


class _Coefficients(torch.autograd.Function):
    """Connect fused corner heat/Gram kernels to first-order source-parameter gradients."""
    @staticmethod
    def forward(ctx, vertices, kernels, gi, gw, decay, phi, norm, source, evec, embedding):
        pairs = kernels.numel()
        heat = phi.new_empty((pairs, 3))
        metric = phi.new_empty((pairs, 6))
        modes, dim = phi.shape[1], source.shape[1]
        constants = (modes, dim, *evec.stride(), *embedding.stride(),
                     triton.next_power_of_2(modes), triton.next_power_of_2(dim))
        if pairs:
            coefficient_gpu.coefficients_forward[(pairs,)](
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
        return None, None, None, *gradients, None, None


def fused_coefficients(vertices, kernels, state, eigen):
    """Exact same intrinsic formulation; float32 outputs, first-order gradients.

    Floating-point reduction order can differ from eager PyTorch. The spectral
    archive/embedding are fixed, as in native HKTex. Their gradients and higher
    derivatives are explicitly unsupported.
    """
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
        *(t.contiguous() for t in tensors[:5]), eigen._eigen_vec_knn, eigen._iso_embeddings)
