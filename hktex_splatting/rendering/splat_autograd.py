"""Shared splat autograd contract and backward used by Heat-Kernel Splatting.

Shared training forward/backward; visibility and support decisions are fixed per call.
Main API: PreparedScene, _Splat, splat."""
from dataclasses import dataclass
import torch
from torch.autograd.function import once_differentiable

from hktex_splatting.rendering.triangle_tile_layout import Dispatch
from hktex_splatting.triton import (
    splat_contributions as gpu,
    heat_kernel_backward as backward,
    splat_reconstruction as runtime_kernels,
)


@dataclass
class PreparedScene:
    """Own fixed visibility and the current support/task snapshot consumed by splat autograd."""
    plan: Dispatch
    kernels: torch.Tensor
    bary: torch.Tensor
    face_ids: torch.Tensor
    entries: torch.Tensor
    patches: torch.Tensor
    record_offsets: torch.Tensor
    patch_offsets: torch.Tensor
    pixels: torch.Tensor
    batches: list
    width: int
    max_records: int
    required_sources: int
    coefficient_topology: object = None
    work_bins: object = None
    renderer_backend: str = 'baseline'


    @property
    def device_bytes(self):
        tensors = (self.kernels, self.bary, self.face_ids, self.entries,
                   self.patches, self.record_offsets, self.patch_offsets, self.pixels)
        extra = (0 if self.coefficient_topology is None else self.coefficient_topology.device_bytes
                 - self.kernels.numel()*self.kernels.element_size())
        return sum(t.numel()*t.element_size() for t in tensors) + sum(
            b[3].numel()*b[3].element_size() for b in self.batches) + extra + (
                0 if self.work_bins is None else self.work_bins.device_bytes)

    def record_origin(self, start):
        # Device plans carry launch metadata without downloading task offsets.
        if hasattr(self.plan, 'record_origin'):
            return self.plan.record_origin(start)
        return int(self.plan.record_offsets[start])


class _Splat(torch.autograd.Function):
    """Differentiate selected spectral contributions; HKS uses this backward unchanged."""
    @staticmethod
    def forward(ctx, heat, metric, colors, base, epsilon, sharpness, scene,
                inner_k, sigma, distance_eps, power, runtime_counts):
        n = scene.face_ids.numel()
        se = torch.full((n, 32), 2147483647, device=heat.device, dtype=torch.int32)
        sw = torch.zeros((n, 32), device=heat.device, dtype=torch.float32)
        batches = scene.batches
        for start, end, count, active in batches:
            rp = torch.empty(count, device=heat.device, dtype=torch.int32)
            re = torch.empty_like(rp)
            rw = heat.new_empty(count)
            gpu.emit[(end-start,)](
                scene.entries, scene.patches, scene.record_offsets, scene.patch_offsets,
                scene.pixels, scene.bary, scene.kernels, heat, metric, epsilon,
                sharpness, rp, re, rw, start, scene.record_origin(start),
                scene.plan.tile_size**2, sigma, distance_eps, power, -1,
                num_warps=4, enable_fp_fusion=False)
            permutation = torch.argsort(rp, stable=False)
            histogram = torch.zeros(n, device=heat.device, dtype=torch.int64)
            histogram.scatter_add_(0, rp.long(), torch.ones_like(rp, dtype=torch.int64))
            offsets = torch.cat((histogram.new_zeros(1), histogram.cumsum(0)))
            gpu.merge[(active.numel(),)](
                offsets, permutation, re, rw, active, se, sw, n, scene.width,
                inner_k, 32, num_warps=4, enable_fp_fusion=False)
        rgb = heat.new_empty((n, 3))
        reconstruction = runtime_kernels.reconstruct if runtime_counts else gpu.reconstruct
        reconstruction[(n,)](se, sw, scene.kernels, colors, base, scene.face_ids,
            rgb, n, scene.kernels.numel(), 32, num_warps=4, enable_fp_fusion=False)
        ctx.save_for_backward(heat, metric, colors, base, epsilon, sharpness,
                              se, sw, scene.kernels, scene.bary, scene.face_ids)
        ctx.recipe = (sigma, distance_eps, power)
        ctx.runtime_counts = runtime_counts
        return rgb.reshape(*scene.plan.shape, 3)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        heat, metric, colors, base, epsilon, sharpness, se, sw, kernels, bary, ids = ctx.saved_tensors
        needs = ctx.needs_input_grad[:6]
        gradients = [torch.zeros_like(t) if need else None for t, need in zip(
            (heat, metric, colors, base, epsilon, sharpness), needs)]
        # Frozen field/color inputs need no gradient allocation or atomic work.
        dummy = heat.new_empty(1)
        gh, gm, gc, gb, ge, gs = [g if g is not None else dummy for g in gradients]
        need_weight = any(needs[i] for i in (0, 1, 4, 5))
        gw = torch.empty_like(sw) if need_weight else dummy
        n = ids.numel()
        kernels_impl = runtime_kernels if ctx.runtime_counts else backward
        kernels_impl.reconstruct_backward[(n,)](
            se, sw, kernels, colors, base, ids, grad.contiguous(), gw, gc, gb,
            kernels.numel(), 32, needs[2], needs[3], need_weight,
            num_warps=4, enable_fp_fusion=False)
        if need_weight:
            kernels_impl.emit_backward[(n,)](
                se, gw, bary, kernels, heat, metric, epsilon, sharpness,
                gh, gm, ge, gs, kernels.numel(), 32, *ctx.recipe,
                needs[0], needs[1], needs[4], needs[5],
                num_warps=4, enable_fp_fusion=False)
        return (*gradients, None, None, None, None, None, None)


def splat(scene, heat, metric, colors, base, epsilon, sharpness, *, inner_k=30,
          sigma=.05, distance_eps=1e-8, power=1, runtime_counts=False) -> torch.Tensor:
    """Device RGB with first-order gradients to all six tensor inputs.

    Inputs can be model properties/activation outputs: contiguous() preserves
    their graphs. The API does not convert learned parameters through NumPy.
    Geometry/camera gradients and second derivatives are unsupported.
    """
    if not 1 <= inner_k <= 32 or sigma <= 0 or distance_eps < 0:
        raise ValueError('Invalid field recipe')
    if power != 1:
        raise ValueError('Training currently supports the audited HKTex power=1 recipe')
    tensors = (heat, metric, colors, base, epsilon, sharpness)
    if any(not isinstance(t, torch.Tensor) or t.dtype != torch.float32 or t.device != scene.bary.device
           for t in tensors):
        raise ValueError('Live float32 tensors must share the scene device')
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError('Expected a source-by-RGB color table')
    pairs = scene.kernels.numel()
    nk = colors.shape[0]
    if nk < scene.required_sources:
        raise ValueError('Scene refers to a source outside the live color table')
    expected = ((pairs, 3), (pairs, 6), (nk, 3), None, (nk,), (nk,))
    for t, shape in zip(tensors, expected):
        if shape is not None and tuple(t.shape) != shape:
            raise ValueError(f'Expected {shape}, got {tuple(t.shape)}')
    if base.numel() != 3:
        raise ValueError('Expected RGB base color')
    return _Splat.apply(*(t.contiguous() for t in tensors), scene,
                        inner_k, sigma, distance_eps, power, runtime_counts)
