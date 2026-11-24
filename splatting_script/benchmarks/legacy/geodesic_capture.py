"""Capture failed geodesic inputs without changing the original tracer's results.

Optional diagnostic training instrumentation. TraceDiagnostics observes the raw
digeo result before HKTex repairs negative face IDs, records bounded CPU cases,
and restores all wrapped callables. Its timings are not performance evidence.
"""
from contextlib import contextmanager, ExitStack
from functools import wraps
import json
from pathlib import Path


def cpu(value):
    return None if value is None else value.detach().cpu().clone()


class TraceDiagnostics:
    """Count trace attempts and save the first eight failing calls (up to 16 rays each)."""
    def __init__(self, torch, trainer, directory, max_cases=8):
        self.torch, self.trainer, self.directory = torch, trainer, Path(directory)
        self.max_cases = max_cases
        self.calls = self.rays = self.failed_events = 0
        self.events, self.cases, self.geometry = [], [], None
        self.context = None

    def observe(self, mesh, starts, direction, result, options):
        context = self.context
        if context is None: return
        self.calls += 1
        self.rays += len(starts.faces)
        endpoints, info = result
        failed = endpoints.faces < 0
        count = int(failed.sum())
        if not count: return
        self.failed_events += count
        event = dict(call=self.calls, step=int(self.trainer.step_index),
            phase='training' if getattr(self.trainer, '_step_active', False) else 'setup_or_validation',
            caller=context['caller'], rays=len(starts.faces), failed_sample_events=count,
            torch_cuda_stream=context['torch_cuda_stream'])
        self.events.append(event)
        if len(self.cases) >= self.max_cases: return
        indices = failed.nonzero(as_tuple=False).flatten()[:16]
        if self.geometry is None:
            self.geometry = {name: cpu(getattr(mesh, name)) for name in
                ('positions', 'triangles', 'adjacencies', 'triangle_normals', 'vertex_normals', 'v2t')}
        case = dict(event, ops_kwargs=dict(options), trace_batch_indices=cpu(indices),
            coords_before=cpu(context['coords'][indices]),
            input_face_ids=cpu(context['face_ids'][indices]),
            start_faces=cpu(starts.faces[indices]), start_uvs=cpu(starts.uvs[indices]),
            direction=cpu(direction[indices]),
            raw_end_faces=cpu(endpoints.faces[indices]), raw_end_uvs=cpu(endpoints.uvs[indices]),
            rotation=cpu(info.rotation_tensor[indices]) if info.rotation_tensor is not None else None,
            momentum_before=cpu(context['transport'][indices]) if context['transport'] is not None else None)
        context['case'], context['indices'] = case, indices
        self.cases.append(case)

    def summary(self):
        return dict(format='hktex_geodesic_diagnostics_v1', calls=self.calls,
            attempted_sample_events=self.rays, failed_sample_events=self.failed_events,
            failed_call_count=len(self.events), saved_cases=len(self.cases),
            failed_event_fraction=self.failed_events/self.rays if self.rays else None,
            events=self.events,
            scope='Diagnostic CUDA synchronization/copy overhead included. Counts include optimizer and '
                  'out-of-place density/setup calls; indices are local to each trace batch, not persistent kernel IDs. '
                  'Only negative raw end-face IDs are counted, matching the original warning condition.')

    def save(self):
        summary = self.summary()
        if self.cases:
            with (self.directory/'geodesic_failures.pt').open('xb') as stream:
                self.torch.save(dict(format='hktex_geodesic_failure_cases_v1', geometry=self.geometry,
                                     cases=self.cases, summary=summary), stream)
        with (self.directory/'geodesic_diagnostics.json').open('x') as stream:
            json.dump(summary, stream, indent=2, allow_nan=False); stream.write('\n')

    @contextmanager
    def install(self):
        """Observe original ops/trace calls and return the exact original objects."""
        import digeo.ops
        from unittest.mock import patch
        tracer = self.trainer.tracer
        original_ops, original_trace = digeo.ops.trace_geodesics, tracer.trace

        @wraps(original_ops)
        def ops(mesh, starts, direction, *args, **kwargs):
            result = original_ops(mesh, starts, direction, *args, **kwargs)
            self.observe(mesh, starts, direction, result, kwargs)
            return result

        @wraps(original_trace)
        def trace(coords, face_ids, tangent, *args, **kwargs):
            previous = self.context
            context = dict(coords=coords, face_ids=face_ids, transport=kwargs.get('transport_vector'),
                caller='optimizer_in_place' if kwargs.get('out_coords') is coords else 'out_of_place',
                torch_cuda_stream=int(self.torch.cuda.current_stream(coords.device).cuda_stream) if coords.is_cuda else None,
                case=None)
            self.context = context
            try:
                result = original_trace(coords, face_ids, tangent, *args, **kwargs)
                if context['case'] is not None:
                    case, indices = context['case'], context['indices']
                    case['coords_after_fallback'] = cpu(result[0][indices])
                    case['faces_after_fallback'] = cpu(result[1][indices])
                    if len(result) > 2: case['momentum_after'] = cpu(result[2][indices])
                return result
            finally:
                self.context = previous

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(digeo.ops, 'trace_geodesics', ops))
                stack.enter_context(patch.object(tracer, 'trace', trace))
                yield self
        finally:
            self.save()
