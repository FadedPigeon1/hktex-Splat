"""Reuse the original UV trainer loop with observable native and HKS adapters.

BaseTrainer.optimise, Adam, GeodesicOpt, samplers and density operations remain
the original implementations. HKS overrides source/batch/forward preparation;
it rebuilds sample support every batch, including after density changes.
This module is imported only by explicit CUDA execution, never CLI previews.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch

from hktex.trainers.uv_texture import UvTextureTrainer
from hktex.modules.utils import KernelInfo
from hktex_splatting.training.surface_adapter import (
    prepare_sample_grid, surface_forward, ValidDensityContributions, ContributionGroups,
)
from splatting_script.benchmarks.training_research import capture_rng, preserve_rng


# Re-export historical helper names; trainer/checkpoint class paths stay stable.
from hktex_splatting.training.training_checks import (
    tree_hash,
    cpu_tree,
    initial_state_difference,
    share_initial_kernel_placement,
    sampler_state,
    batch_identity,
    preserve_feedback_indices,
    native_forward,
    check_sample_gradients,
)


class VerificationComplete(Exception):
    """Stop the inherited loop before its first training forward/optimizer update."""


class RecordedUvTrainer(UvTextureTrainer):
    """Original UV trainer with receipts; optimise itself is inherited unchanged."""
    backend = 'native'

    def plot_model_histograms(self):
        if getattr(self, 'surface_options', {}).get('training_plots', True): return super().plot_model_histograms()

    def plot_errors(self, errors):
        if getattr(self, 'surface_options', {}).get('training_plots', True): return super().plot_errors(errors)

    def plot_gradient_norms(self, *args, **kwargs):
        if getattr(self, 'surface_options', {}).get('training_plots', True): return super().plot_gradient_norms(*args, **kwargs)

    def attach_run(self, directory, options, expected_initial=None):
        self.run_dir, self.surface_options = Path(directory), options
        self.expected_initial = expected_initial
        self.step_index, self._initial_recorded, self._step_active = 0, False, False
        self.records = []

    def _initial_receipt(self):
        if self._initial_recorded: return
        self.initial_receipt_started_at = time.perf_counter()
        rng = capture_rng(torch, self.device)
        with preserve_rng(torch, self.device):
            # Independent held-out surface batch; no importance-error update.
            from hktex.utils import seed_everything
            seed_everything(self.surface_options['evaluation_seed'])
            evaluation = next(iter(getattr(self.datamodule, '_camera_evaluation_dataset', self.datamodule.train_dataset)))
        snapshot = dict(format='hktex_surface_initial_v1', model=cpu_tree(self.model.state_dict()),
            rng=rng, sampler=cpu_tree(sampler_state(self.datamodule)), evaluation=cpu_tree(evaluation))
        self.initial_identity = {key: tree_hash(snapshot[key]) for key in ('model', 'rng', 'sampler', 'evaluation')}
        self.initialization_receipt = dict(
            policy=('shared-camera-initial-model' if self.surface_options.get('supervision') == 'camera' else 'shared-kernel-placement') if self.surface_options.get('shared_initial_model') else 'independent-replay',
            reference_snapshot=str(self.expected_initial) if self.expected_initial else None,
            model_before_sha256=self.initial_identity['model'], copied_fields=[])
        if self.expected_initial:
            expected = torch.load(self.expected_initial, map_location='cpu', weights_only=False)
            if expected.get('format') != 'hktex_surface_initial_v1':
                raise ValueError('Expected a full-surface initial.pt, not a camera-benchmark snapshot')
            identity = {key: tree_hash(expected[key]) for key in self.initial_identity}
            if self.surface_options.get('shared_initial_model') and all(
                    identity[key] == self.initial_identity[key] for key in ('rng', 'sampler', 'evaluation')):
                camera_initial = self.surface_options.get('supervision') == 'camera'
                fields = ('_kernel_locations', '_kernel_face_ids', '_kernel_colours', '_mean_colour') if camera_initial else ('_kernel_locations', '_kernel_face_ids')
                proposal = {key: snapshot['model'][key] for key in fields
                            if key in snapshot['model']}
                copied = share_initial_kernel_placement(self.model, expected['model'], self.optimizers,
                    include_colours=camera_initial)
                if copied:
                    with (self.run_dir/'initial_kernel_proposal.pt').open('xb') as stream: torch.save(proposal, stream)
                    self.initialization_receipt['copied_fields'] = copied
                    snapshot['model'] = cpu_tree(self.model.state_dict())
                    self.initial_identity['model'] = tree_hash(snapshot['model'])
            if identity != self.initial_identity:
                mismatched = [key for key in identity if identity[key] != self.initial_identity[key]]
                report = dict(actual_identity=self.initial_identity, expected_identity=identity,
                    mismatched_components=mismatched, expected_snapshot=str(self.expected_initial),
                    **initial_state_difference({key: snapshot[key] for key in mismatched},
                                              {key: expected[key] for key in mismatched}))
                with (self.run_dir/'initial_actual.pt').open('xb') as stream: torch.save(snapshot, stream)
                (self.run_dir/'initial_mismatch.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
                raise RuntimeError('Initial model/RNG/sample pool/held-out batch mismatch before any update; '
                    'differing components='+str(mismatched)+'; see initial_mismatch.json and initial_actual.pt')
            self.evaluation_batch = expected['evaluation']
        else:
            self.evaluation_batch = snapshot['evaluation']
            with (self.run_dir/'initial.pt').open('xb') as stream: torch.save(snapshot, stream)
        self.initialization_receipt['model_after_sha256'] = self.initial_identity['model']
        (self.run_dir/'initialization.json').write_text(json.dumps(self.initialization_receipt, indent=2)+'\n')
        (self.run_dir/'initial_identity.json').write_text(json.dumps(self.initial_identity, indent=2)+'\n')
        gate = check_sample_gradients(self, self.evaluation_batch, self.backend,
                                     self.surface_options['verify_samples'])
        self.strict_render_gate_passed = gate['passed']
        self.diagnostic_render_variation = gate.get('diagnostic_accepted', False)
        self.native_approximation_accepted=gate.get('approximation_accepted',False)
        self.gpu_operator_checks_passed=gate.get('gpu_operator_checks_passed',gate['passed'])
        (self.run_dir/'sample_validation.json').write_text(json.dumps(gate, indent=2, allow_nan=False)+'\n')
        if not (gate['passed'] or self.diagnostic_render_variation or self.native_approximation_accepted):
            raise RuntimeError('Validation failed: '+', '.join(gate.get('failed_checks', ['RGB/gradient consistency']))+
                '; see sample_validation.json (native_candidate_replay), kernel_support_failure.pt or render_gate_failure.pt')
        self._initial_recorded = True
        self.initial_receipt_seconds = time.perf_counter()-self.initial_receipt_started_at
        if self.surface_options['verify_only']: raise VerificationComplete()
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.training_start = time.perf_counter()

    def _begin_step(self):
        self._initial_receipt()
        self._step_active = True
        self._step_start = time.perf_counter()
        self._before_count = self.model.N_sources
        self._before_parameters = tuple(id(parameter) for parameter in self.model.parameters())

    def prepare_knn(self, save_barycentric=True):
        self._begin_step()
        return super().prepare_knn(save_barycentric=save_barycentric)

    def prepare_batch(self, data):
        if self._step_active:
            self._batch_identity = batch_identity(data)
        inputs = {k:v for k,v in data.items() if not k.startswith('_camera_')}
        return preserve_feedback_indices(super().prepare_batch(inputs), data)

    def compute_loss(self, colours, gt_colours, data=None):
        result = super().compute_loss(colours, gt_colours, data=data)
        if self._step_active: self._recorded_loss = result[0].detach()
        return result

    def forward_knn(self, data):
        """Original native material field, followed by the optional shared shader."""
        from hktex_splatting.rendering.reference_mesh import shade_batch
        colours, *aux = super().forward_knn(data)
        return shade_batch(colours, data, self.surface_options.get('lighting')), *aux

    def reset_knn(self):
        super().reset_knn()
        if not getattr(self, '_step_active', False): return
        row = dict(step=self.step_index, loss=float(self._recorded_loss),
            kernel_count_before=self._before_count, kernel_count_after=self.model.N_sources,
            parameters_replaced=self._before_parameters != tuple(id(parameter) for parameter in self.model.parameters()),
            batch_sha256=self._batch_identity, host_step_seconds=time.perf_counter()-self._step_start)
        # The original loop already reads loss.item() before this callback.
        if not np.isfinite(row['loss']): raise RuntimeError('Nonfinite original-loop training loss')
        self.records.append(row)
        encoded = json.dumps(row, allow_nan=False)+'\n'
        stream = getattr(self, '_trajectory_stream', None)
        if stream is not None and not stream.closed:
            stream.write(encoded)
            if (self.step_index+1) % 128 == 0 or row['parameters_replaced']:
                stream.flush()
        else:
            with (self.run_dir/'trajectory.jsonl').open('a') as stream: stream.write(encoded)
        self.step_index += 1
        self._step_active = False


class HksUvTrainer(RecordedUvTrainer):
    """Original optimisation lifecycle using HKS on packed surface samples."""
    backend = 'hybrid'

    def configure(self, datamodule, **kwargs):
        super().configure(datamodule, **kwargs)
        from hktex.density_controllers.importance_pruning import ImportancePruningController
        from hktex.density_controllers.error_densification import ErrorDensificationController
        if any(not isinstance(dc, (ImportancePruningController, ErrorDensificationController))
               for dc in self.density_controllers):
            raise ValueError('HKS surface adapter supports the original pruning/error-densification controllers only')
        self.density_controllers = [ValidDensityContributions(dc, self.model) for dc in self.density_controllers]
        for bridge in self.density_controllers:
            if type(bridge.controller) is ImportancePruningController:
                bridge.accumulation_option = 'accumulation_interval'
                bridge.accumulator_names = ('_hit_accumulator', '_contrib_accumulator')
            elif type(bridge.controller) is ErrorDensificationController:
                bridge.accumulation_option = 'error_accumulation_interval'
                bridge.accumulator_names = ('_error_accumulator',)

    def attach_run(self, directory, options, expected_initial=None):
        if (options.get('density_policy', 'original') != 'original' or
                options.get('optimizer_policy', 'original') != 'original'):
            raise ValueError('HKS comparisons require original density and optimizer policies')
        super().attach_run(directory, options, expected_initial)

    def prepare_knn(self, save_barycentric=True):
        self._begin_step()
        vertices = self.mesh.get_face_vertices(self.model.kernel_face_ids)
        bary = self.mesh.cartesian_to_barycentric(self.model.kernel_locations, vertices)
        if save_barycentric: self.model.save_barycentric_locations(bary)
        # Consumed once by this update's renderer, never retained after a step.
        self._hks_prepared_source_geometry = (vertices, bary, self.model.kernel_locations,
            self.model.kernel_locations._version, self.model.kernel_face_ids,
            self.model.kernel_face_ids._version, torch.is_grad_enabled())
        return KernelInfo(vert_idx=vertices, barycentric_coords=bary, albo_evecs=None, mass=None)

    def prepare_batch(self, data):
        self._batch_identity = batch_identity(data)
        optimized = self.surface_options.get('surface_adapter') in ('optimized', 'aggressive', 'direct')
        if optimized:
            if not hasattr(self, '_contribution_groups'): self._contribution_groups = ContributionGroups()
            self._contribution_groups.clear()
            for controller in self.density_controllers:
                controller.group_cache = self._contribution_groups
                controller.skip_inactive = self.surface_options.get('surface_adapter') in ('aggressive', 'direct')
        if '_camera_ids' in data:
            from hktex_splatting.training.camera_supervision import cpu_array
            prepared = self._move_to_device({k:v for k,v in data.items() if not k.startswith('_camera_')})
            device_camera=data.get('_camera_device',False)
            prepared['_camera_ids'] = data['_camera_ids'] if device_camera else cpu_array(data['_camera_ids'])
            prepared['_camera_bary'] = data['_camera_bary'] if device_camera else cpu_array(data['_camera_bary'])
            if device_camera:
                prepared['_camera_device']=True
                if '_camera_pixels' in data:prepared['_camera_pixels']=data['_camera_pixels']
            if '_camera_frame' in data:prepared['_camera_frame']=data['_camera_frame']
            if '_camera_key' in data: prepared['_camera_key'] = data['_camera_key']
            return prepared
        if self.surface_options.get('surface_adapter') == 'direct':
            # Original sampler IDs are already on the CPU; barycentrics go to
            # the GPU once with the batch, without a NumPy/grid round trip.
            host_ids = data['face_id'].detach().cpu().numpy().copy()
            prepared = self._move_to_device(data)
            prepared['_hks_host_ids'] = host_ids
            return preserve_feedback_indices(prepared, data)
        grid, inverse = prepare_sample_grid(data['face_id'].detach().cpu().numpy(),
            data['bary'].detach().cpu().numpy(), grouped=optimized)
        prepared = self._move_to_device(data)
        prepared['_hks_grid'] = grid
        prepared['_hks_inverse'] = inverse
        return preserve_feedback_indices(prepared, data)

    def forward_knn(self, data):
        inactive_original = all(not controller.needs_contributions(self.step_index)
                                for controller in self.density_controllers)
        if inactive_original:
            return surface_forward(self, data), None, None, None
        rgb, tape = surface_forward(self, data, trace=True)
        # The original callback contract is [k,P,1]. Ignore tape padding beyond
        # the configured inner width; retain sentinels until the density bridge.
        width = self.model.cfg.knn_inner_k or self.model.N_sources
        indices = tape['top'][:, :width].T[..., None].detach()
        weights = tape['weights'][:, :width].T[..., None].detach()
        return rgb, None, indices, weights
