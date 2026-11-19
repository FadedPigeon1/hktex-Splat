"""Expose the public HKS renderer and field types through lazy imports.

Rendering and training API; importing this namespace starts no GPU work.
Main API: DirectTileRenderer, TileSurface, LiveHeatField, __getattr__."""
from importlib import import_module

__version__ = '0.1.0'
_EXPORTS = {
    'DirectTileRenderer': ('hktex_splatting.rendering.heat_kernel_splatting', 'DirectTileRenderer'),
    'TileSurface': ('hktex_splatting.rendering.heat_kernel_splatting', 'TileSurface'),
    'LiveHeatField': ('hktex_splatting.modules.spectral_heat_field', 'LiveHeatField'),
    'SourceState': ('hktex_splatting.modules.spectral_heat_field', 'SourceState'),
    'PreparedScene': ('hktex_splatting.rendering.splat_autograd', 'PreparedScene'),
    'splat': ('hktex_splatting.rendering.splat_autograd', 'splat'),
    'ResidentSupportGeometry': ('hktex_splatting.knn_heat.triangle_kernel_support', 'ResidentSupportGeometry'),
    'ResidentDispatchGeometry': ('hktex_splatting.rendering.resident', 'ResidentDispatchGeometry'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS: raise AttributeError(name)
    module, symbol = _EXPORTS[name]
    value = getattr(import_module(module), symbol)
    globals()[name] = value
    return value
