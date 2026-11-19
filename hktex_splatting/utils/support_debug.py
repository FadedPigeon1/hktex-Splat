"""Check live support rows/CSR and save failure evidence without repairing it.

Opt-in rendering/training diagnostics; runtime safety checks remain active.
Main API: support_errors, SupportDebugger."""
import json
from pathlib import Path
import uuid

import numpy as np


def support_errors(rows, visible, counts, offsets, kernels, faces, source_count):
    """Check candidate rows and their exact CSR packing using host arrays."""
    errors = []
    if rows.ndim != 2 or len(rows) != len(visible) or rows.shape[1] < 1:
        return ['candidate row shape']
    if counts.ndim != 1 or offsets.shape != (len(counts)+1,) or faces.shape != kernels.shape:
        return ['CSR array shapes']
    row_counts = rows[:, 0].astype(np.int64)
    if np.any((row_counts < 0) | (row_counts > rows.shape[1]-1)):
        return ['candidate row count outside row capacity']
    expected_counts = np.zeros(len(counts), dtype=np.int64)
    expected_counts[visible] = row_counts
    if not np.array_equal(counts, expected_counts):
        errors.append('CSR counts differ from candidate rows')
    expected_offsets = np.r_[0, expected_counts.cumsum()]
    if not np.array_equal(offsets, expected_offsets):
        errors.append('CSR offsets differ from candidate counts')
    valid = np.arange(rows.shape[1]-1)[None, :] < row_counts[:, None]
    expected_kernels = rows[:, 1:][valid]
    if np.any((expected_kernels < 0) | (expected_kernels >= source_count)):
        errors.append('candidate kernel ID outside source table')
    paired = np.arange(max(0, rows.shape[1]-2))[None, :] < (row_counts-1)[:, None]
    if np.any(paired & (rows[:, 2:] <= rows[:, 1:-1])):
        errors.append('duplicate or unordered candidate row')
    if not np.array_equal(kernels, expected_kernels):
        errors.append('CSR kernels differ from candidate rows')
    expected_faces = np.repeat(visible, row_counts)
    if not np.array_equal(faces, expected_faces):
        errors.append('CSR faces differ from candidate rows')
    if np.any((kernels < 0) | (kernels >= source_count)):
        errors.append('CSR kernel ID outside source table')
    if len(kernels) > 1 and np.any(~((faces[1:] > faces[:-1]) |
            ((faces[1:] == faces[:-1]) & (kernels[1:] > kernels[:-1])))):
        errors.append('duplicate or unordered kernel/triangle support')
    return errors


class SupportDebugger:
    """One forward's diagnostic ownership; normal rendering never creates this."""
    def __init__(self, directory, field, source_state, camera):
        self.directory, self.field, self.state, self.camera = Path(directory), field, source_state, camera
        self.rows = None

    def check(self, rows, support, stage='packed_support'):
        import torch
        if rows is not None:
            self.rows = rows
        tensors = dict(rows=self.rows, counts=support.counts, offsets=support.offsets,
                       kernels=support.kernels, faces=support.entry_faces)
        host = {name:value.detach().cpu().clone() for name,value in tensors.items()}
        arrays = {name:value.numpy() for name,value in host.items()}
        errors = support_errors(visible=support.geometry.visible, source_count=support.source_count, **arrays)
        if not errors:
            return
        # Capture inputs as well as malformed outputs; no eigensystem load or
        # training replay is needed to inspect rows/scans/packing on the CPU.
        payload = dict(format='hktex_support_failure_v1', stage=stage, errors=errors,
            source_count=support.source_count, maximum=support.maximum,
            tasks=support.tasks, fragments=support.fragments, camera=self.camera,
            visible=torch.as_tensor(support.geometry.visible.copy()), **host,
            sources=self.state.embedding.detach().cpu().clone(),
            geometry=[dict(begin=begin, center=center.detach().cpu().clone(), radius=radius.detach().cpu().clone())
                      for begin,center,radius in support.geometry.chunks],
            neighbors=support.geometry.neighbors,
            model={name:value.detach().cpu().clone() for name,value in self.field.model.state_dict().items()})
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory/('support_failure_'+uuid.uuid4().hex+'.pt')
        with path.open('xb') as stream:
            torch.save(payload, stream)
        path.with_suffix('.json').write_text(json.dumps(dict(stage=stage, errors=errors, snapshot=str(path)), indent=2)+'\n')
        raise RuntimeError('Invalid hybrid support at '+stage+': '+ '; '.join(errors)+'. Snapshot: '+str(path))
