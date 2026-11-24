"""Bounded, synchronized stage timing for original/HKS surface training.

SurfaceStageProfiler wraps calls temporarily without replacing the original loop.
Measurements include CUDA drains and are diagnostic, never speed-run timings.
"""
from contextlib import contextmanager
from functools import wraps
import time
import numpy as np


class SurfaceStageProfiler:
    def __init__(self, torch, trainer, start, steps):
        self.torch, self.trainer = torch, trainer
        self.start, self.steps = start, steps
        self.rows, self.reuse = {}, []

    def active(self):
        return (getattr(self.trainer, '_initial_recorded', False) and
                self.start <= self.trainer.step_index < self.start+self.steps)

    @contextmanager
    def stage(self, name):
        if not self.active():
            yield
            return
        self.torch.cuda.synchronize(self.trainer.device)
        begin = time.perf_counter()
        try:
            yield
        finally:
            self.torch.cuda.synchronize(self.trainer.device)
            row = self.rows.setdefault(name, dict(calls=0, wall_seconds=0.))
            row['calls'] += 1
            row['wall_seconds'] += time.perf_counter()-begin

    def samples(self, ids):
        if self.active():
            if isinstance(ids,self.torch.Tensor):
                # Profiling-only host diagnostics; production geometry stays GPU-resident.
                ids=ids.detach().cpu().numpy()
            _, counts = np.unique(ids[ids >= 0], return_counts=True)
            self.reuse.append(dict(step=self.trainer.step_index, samples=int(counts.sum()),
                unique_triangles=len(counts), singleton_triangles=int((counts == 1).sum()),
                samples_per_triangle=float(counts.mean()) if len(counts) else 0.))

    @contextmanager
    def install(self):
        saved = []
        def wrap(obj, name, label):
            own = name in vars(obj)
            original = getattr(obj, name)
            @wraps(original)
            def measured(*args, **kwargs):
                with self.stage(label): return original(*args, **kwargs)
            saved.append((obj, name, own, original))
            setattr(obj, name, measured)
        self.trainer.surface_profiler = self
        try:
            for name in ('prepare_knn', 'prepare_batch', 'forward_knn', 'compute_loss', 'reset_knn'):
                wrap(self.trainer, name, name)
            wrap(self.torch.autograd, 'backward', 'backward')
            for i, optimizer in enumerate(self.trainer.optimizers):
                wrap(optimizer, 'step', 'optimizer/'+str(i))
            for i, controller in enumerate(self.trainer.density_controllers):
                for name in ('pre_backward_step', 'post_backward_step'):
                    wrap(controller, name, 'density/'+str(i)+'/'+name)
            dataset=self.trainer.datamodule.train_dataset
            if hasattr(dataset,'visibility_buffers'):
                wrap(dataset,'visibility_buffers','camera_rasterization')
                if hasattr(dataset,'shade_target'):
                    wrap(dataset,'shade_target','camera_target_lighting')
                sampler=getattr(dataset,'target_sampler',None)
                if sampler is not None:
                    wrap(sampler,'sample_device' if dataset.raster_backend=='nvdiffrast' else 'sample','camera_target_lookup')
            yield self
        finally:
            for obj, name, own, original in reversed(saved):
                if own: setattr(obj, name, original)
                else: delattr(obj, name)
            del self.trainer.surface_profiler

    def summary(self):
        return dict(start=self.start, requested_steps=self.steps, stages=self.rows,
            sample_reuse=self.reuse,
            scope='Synchronized inclusive wall times; nested stages overlap. Do not sum rows. '
                  'First prepare_knn receipt/setup is excluded. Sampling and scheduler calls are not separately timed. '
                  'Diagnostic overhead is included in whole-run times; use a separate unprofiled run for performance.')
