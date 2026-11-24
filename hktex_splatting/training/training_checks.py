"""Initial-state receipts and strict RGB/VJP gates for native/HKS training.

These checks support reproducibility and bounded failure diagnosis. They are
separate from the optimization loop; no gate is disabled by this module split.
"""
import hashlib
import numpy as np
import torch
from hktex_splatting.rendering.reference_mesh import shade_batch
from hktex.trainers.uv_texture import UvTextureTrainer
from hktex_splatting.training.surface_adapter import surface_forward
from splatting_script.benchmarks.training_research import preserve_rng

def tree_hash(value):
    """Fingerprint nested state by tensor contents, dtype and shape, not pickle IDs."""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype)).encode()); digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str((item.shape, item.dtype)).encode()); digest.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str): visit(key); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item: visit(child)
        else:
            digest.update(repr(item).encode())
        digest.update(b'\0')
    visit(value)
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value


def initial_state_difference(actual, expected, limit=64):
    """Describe exact receipt differences without accepting or replacing state.

    Called only after an initialization mismatch. Detail is bounded; full states
    remain available in initial_actual.pt and the original initial.pt.
    """
    details = []
    truncated = False

    def visit(left, right, path):
        nonlocal truncated
        if tree_hash(left) == tree_hash(right): return
        if len(details) >= limit:
            truncated = True
            return
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                if key not in left or key not in right:
                    if len(details) < limit:
                        details.append(dict(path=path+'/'+str(key),
                            reason='missing from actual' if key not in left else 'missing from expected'))
                    else: truncated = True
                else: visit(left[key], right[key], path+'/'+str(key))
            return
        if isinstance(left, (list, tuple)) and type(left) is type(right) and len(left) == len(right):
            for i, (a, b) in enumerate(zip(left, right)): visit(a, b, path+'/'+str(i))
            return
        row = dict(path=path, actual_sha256=tree_hash(left), expected_sha256=tree_hash(right))
        arrays = []
        for value in (left, right):
            arrays.append(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
        a, b = arrays
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            row.update(actual_shape=list(a.shape), expected_shape=list(b.shape),
                actual_dtype=str(a.dtype), expected_dtype=str(b.dtype))
            if a.shape == b.shape and a.dtype == b.dtype and a.dtype.kind in 'biuf':
                unequal = a != b
                row['unequal_elements'] = int(np.count_nonzero(unequal))
                row['first_unequal_flat_indices'] = np.flatnonzero(unequal.reshape(-1))[:8].tolist()
                row['actual_nonfinite'] = int(np.count_nonzero(~np.isfinite(a)))
                row['expected_nonfinite'] = int(np.count_nonzero(~np.isfinite(b)))
                if not row['actual_nonfinite'] and not row['expected_nonfinite']:
                    row['max_abs'] = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.0
        else:
            row.update(actual_value=repr(left)[:200], expected_value=repr(right)[:200])
        details.append(row)

    visit(actual, expected, '')
    return dict(details=details, details_truncated=truncated, detail_limit=limit)


def share_initial_kernel_placement(model, expected, optimizers, *, include_colours=False):
    """Share explicit initial placement, optionally its dependent camera colors, before updates.

    This explicit comparison policy uses the first original initialization for
    every worker. Camera mode may also share initial colors/mean; all other fields
    must already match exactly. Surface-mode sharing remains placement-only.
    """
    actual = model.state_dict()
    placement = {'_kernel_locations', '_kernel_face_ids'}
    if include_colours: placement |= {'_kernel_colours', '_mean_colour'}
    if set(actual) != set(expected): return []
    changed = [key for key in actual if tree_hash(actual[key]) != tree_hash(expected[key])]
    if not changed or not set(changed) <= placement: return []
    for key in placement:
        if actual[key].shape != expected[key].shape or actual[key].dtype != expected[key].dtype:
            return []
    if any(parameter.grad is not None for parameter in model.parameters()) or any(
            optimizer.state for optimizer in optimizers):
        raise RuntimeError('Shared placement requires fresh parameters and empty optimizer state')
    if hasattr(model._kernel_locations, 'bary_coords'):
        raise RuntimeError('Shared placement must precede cached barycentric/kernel preparation')
    with torch.no_grad():
        for key in placement: actual[key].copy_(expected[key])
    return sorted(changed)


def sampler_state(datamodule):
    sampler = getattr(datamodule, '_is_wrapper_dataset', None)
    if sampler is None: return None
    return {key: getattr(sampler, key) for key in
            ('pool', 'errors', 'pool_size', 'step', 'current_sampling_method')}


def batch_identity(data):
    if data.get('_camera_device',False):
        # Full per-step pixel hashing would download device visibility and targets.
        # This records the camera sequence only; pixel identity is NOT implied.
        return 'camera-metadata:'+tree_hash(data['_camera_frame'])
    return tree_hash({key: data[key] for key in ('face_id', 'bary', 'colour', 'pool_indices') if key in data})


def preserve_feedback_indices(prepared, original):
    # ImportanceSamplerDataset owns a CPU error pool. Pool IDs are sampler
    # feedback metadata, not CUDA inputs to either model's colour calculation.
    if 'pool_indices' in original:
        prepared['pool_indices'] = original['pool_indices'].detach().cpu()
    return prepared


def native_forward(trainer, data):
    """Direct calls to the original model/UV preparation for independent evaluation."""
    trainer.model.prepare_kernels(trainer.mesh, trainer.eigalbo_interp, save_barycentric=True)
    prepared = UvTextureTrainer.prepare_batch(trainer, data)
    return shade_batch(UvTextureTrainer.forward_knn(trainer, prepared)[0], data,
                       trainer.surface_options.get('lighting'))


def compare_selected_contributions(left, right):
    """Compare source/weight pairs independently of harmless rank permutations.

    Membership, per-source weights and sentinel padding remain strict. RGB and
    VJPs must be checked separately because reduction order can change rounding.
    """
    width=min(left['top'].shape[1],right['top'].shape[1])
    li,lo=left['top'][:,:width].sort(dim=1)
    ri,ro=right['top'][:,:width].sort(dim=1)
    lw=left['weights'][:,:width].gather(1,lo)
    rw=right['weights'][:,:width].gather(1,ro)
    padding=all(bool((t['top'][:,width:]<0).all()&(t['weights'][:,width:]==0).all()) for t in (left,right))
    membership=bool(torch.equal(li,ri))
    weights=membership and bool(torch.allclose(lw,rw,atol=3e-5,rtol=3e-4))
    return dict(ordered_ids_equal=bool(torch.equal(left['top'][:,:width],right['top'][:,:width])),
        membership_equal=membership,weights_by_source_close=weights,padding_valid=padding,
        rows_with_different_membership=int((li!=ri).any(1).sum()),
        max_weight_error_same_membership=None if not membership else float((lw-rw).detach().abs().max()),
        passed=membership and weights and padding)


def native_candidate_replay(trainer, data):
    """Validation oracle: native point candidates through the existing HKS pipeline.

    Use one original query per face to avoid introducing extra candidates via
    triangle unions. Work tiles preserve its barycentric position; this is not
    a camera benchmark or a production support-search replacement.
    """
    face_ids=data['face_id'].long()
    unique,inverse=torch.unique(face_ids,sorted=True,return_inverse=True)
    first=torch.full((len(unique),),len(face_ids),device=face_ids.device,dtype=torch.long)
    first.scatter_reduce_(0,inverse,torch.arange(len(face_ids),device=face_ids.device),reduce='amin')
    rows=first.sort().values
    subset={k:v[rows.to(v.device)] for k,v in data.items()
        if not k.startswith('_') and isinstance(v,torch.Tensor)}
    parameters=tuple(trainer.model.parameters())
    with preserve_rng(torch,trainer.device):
        trainer.model.reset(trainer.eigalbo_interp)
        trainer.model.prepare_kernels(trainer.mesh,trainer.eigalbo_interp,save_barycentric=True)
        prepared=UvTextureTrainer.prepare_batch(trainer,subset)
        native=UvTextureTrainer.forward_knn(trainer,prepared)
        native=(shade_batch(native[0], subset, trainer.surface_options.get('lighting')), *native[1:])
        points=prepared['points_info']
        ids=points['indices'].detach().clone()
        loss=UvTextureTrainer.compute_loss(trainer,native[0],subset['colour'].to(trainer.device))[0]
        native_grads=torch.autograd.grad(loss,parameters,allow_unused=True)
    # Raw native heat already includes its intrinsic-distance attenuation.
    with torch.no_grad():
        raw,normalized=trainer.eigalbo_interp.diffuse_heat(
            points['albo_evals'],points['albo_evecs'],ids,weights=points['weights'])
        native_response=(raw if normalized is None else normalized).detach()
    values=[];gradients=[];tapes=[]
    candidates=(subset['face_id'].to(ids.device),ids)
    for eager in (False,True):
        with preserve_rng(torch,trainer.device):
            rgb,tape=surface_forward(trainer,subset,trace=True,diagnostic_full_trace=True,
                coefficient_backend='eager' if eager else 'triton',splat_backend='torch' if eager else 'triton',
                reference_adapter=eager,validation_candidates=candidates)
            loss=UvTextureTrainer.compute_loss(trainer,rgb,subset['colour'].to(trainer.device))[0]
            gradients.append(torch.autograd.grad(loss,parameters,allow_unused=True))
        values.append(rgb.detach());tapes.append(tape)
    def compare_grads(a,b):
        return [dict(parameter=name,passed=(x is y if x is None or y is None else
            bool(torch.isfinite(x).all()&torch.isfinite(y).all()) and bool(torch.allclose(x,y,atol=3e-5,rtol=3e-4))),
            max_abs=None if x is None or y is None else float((x-y).abs().max()))
            for (name,_),x,y in zip(trainer.model.named_parameters(),a,b)]
    native_checks=compare_grads(gradients[0],native_grads)
    eager_checks=compare_grads(gradients[0],gradients[1])
    ordered,order=ids.sort(dim=1)
    same_ids=torch.equal(tapes[0]['outer'],ordered)
    response_error=None if not same_ids else float((tapes[0]['response'].detach()-native_response.gather(1,order)).abs().max())
    common=min(t['top'].shape[1] for t in tapes)
    selected=compare_selected_contributions(tapes[0],tapes[1])
    operator_ids=bool(torch.equal(tapes[0]['top'][:,:common],tapes[1]['top'][:,:common]))
    operator_weights=bool(torch.allclose(tapes[0]['weights'][:,:common],tapes[1]['weights'][:,:common],atol=3e-5,rtol=3e-4))
    padding_ok=all(bool((t['top'][:,common:]<0).all()&(t['weights'][:,common:]==0).all()) for t in tapes)
    width=min(native[2].shape[0],tapes[0]['top'].shape[1])
    native_top=native[2][...,0].T[:,:width].sort(dim=1).values
    replay_top=tapes[0]['top'][:,:width].sort(dim=1).values
    return dict(scope='Validation only: first original query per face, native outer candidates, HKS work tiles',
        samples=len(rows),candidate_count=ids.shape[1],candidate_ids_equal=same_ids,
        native_rgb_close=bool(torch.allclose(values[0],native[0].detach(),atol=3e-5,rtol=3e-4)),
        native_rgb_max_abs=float((values[0]-native[0].detach()).abs().max()),
        native_response_max_abs=response_error,native_top_ids_equal=bool(torch.equal(native_top,replay_top)),
        native_parameters=native_checks,
        operator_top_ids_equal=operator_ids,operator_weights_close=operator_weights,padding_valid=padding_ok,
        operator_contributions=selected,operator_rgb_max_abs=float((values[0]-values[1]).abs().max()),
        operator_passed=same_ids and selected['passed'] and
            bool(torch.allclose(values[0],values[1],atol=3e-5,rtol=3e-4)) and all(c['passed'] for c in eager_checks),
        operator_parameters=eager_checks,replay_not_a_training_backend=True)


def check_sample_gradients(trainer, data, backend, samples=32):
    """Finite native VJPs; HKS RGB/VJPs versus existing eager spectral/splat control."""
    from hktex_splatting.training.camera_supervision import validation_subset
    data = validation_subset(data, samples)
    parameters = tuple(trainer.model.parameters())
    def run(reference=False):
        with preserve_rng(torch, trainer.device):
            rgb = (native_forward(trainer, data) if backend == 'native' else surface_forward(
                trainer, data, coefficient_backend='eager' if reference else 'triton',
                splat_backend='torch' if reference else 'triton', reference_adapter=reference))
            loss = UvTextureTrainer.compute_loss(trainer, rgb, data['colour'].to(trainer.device))[0]
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        return rgb.detach(), tuple(g.detach() if g is not None else None for g in grads)
    actual, gradients = run()
    finite = bool(torch.isfinite(actual).all()) and all(
        g is None or bool(torch.isfinite(g).all()) for g in gradients)
    report = dict(passed=finite, samples=len(data['face_id']), finite=finite,
        reference='original pointwise forward finite-gradient health' if backend == 'native'
                  else 'eager coefficients/splat with the same support policy, plus independent native field RGB/VJPs',
        selector_independently_validated=False, absolute_tolerance=3e-5, relative_tolerance=3e-4)
    if backend == 'hybrid':
        expected, reference_gradients = run(True)
        report['rgb_max_abs'] = float((actual-expected).abs().max())
        checks = []
        for (name, _), grad, ref in zip(trainer.model.named_parameters(), gradients, reference_gradients):
            checks.append(dict(parameter=name, passed=(grad is ref if grad is None or ref is None else
                bool(torch.isfinite(grad).all() & torch.isfinite(ref).all()) and
                bool(torch.allclose(grad, ref, atol=3e-5, rtol=3e-4))),
                diagnostic_passed=(grad is ref if grad is None or ref is None else
                    bool(torch.isfinite(grad).all() & torch.isfinite(ref).all()) and
                    bool(torch.allclose(grad, ref, atol=1e-4, rtol=3e-4))),
                max_abs=None if grad is None or ref is None else float((grad-ref).abs().max())))
        report['parameters'] = checks
        report['passed'] = finite and bool(torch.allclose(actual, expected, atol=3e-5, rtol=3e-4)) and all(
            row['passed'] for row in checks)
        if trainer.surface_options.get('surface_adapter') in ('optimized', 'aggressive', 'direct'):
            with preserve_rng(torch, trainer.device), torch.no_grad():
                compact_rgb, compact = surface_forward(trainer, data, trace=True)
                full_rgb, full = surface_forward(trainer, data, trace=True, reference_adapter=True)
            report['compact_training_output'] = dict(
                ids_equal=bool(torch.equal(compact['top'], full['top'])),
                weights_close=bool(torch.allclose(compact['weights'], full['weights'], atol=3e-5, rtol=3e-4)),
                rgb_close=bool(torch.allclose(compact_rgb, full_rgb, atol=3e-5, rtol=3e-4)))
            report['passed'] = report['passed'] and all(report['compact_training_output'].values())
        if not report['passed'] and not trainer.surface_options.get('occupancy_threshold',0):
            # Failure-only bounded evidence; do not waive strict RGB/VJP gates.
            with preserve_rng(torch, trainer.device), torch.no_grad():
                fast_rgb, fast = surface_forward(trainer, data, trace=True, diagnostic_full_trace=True)
                eager_rgb, eager = surface_forward(trainer, data, trace=True, reference_adapter=True,
                    coefficient_backend='eager', splat_backend='torch', diagnostic_full_trace=True)
            difference = (fast_rgb-eager_rgb).abs().amax(1)
            rows = torch.where(difference > report['absolute_tolerance'])[0]
            selected = rows[:16]
            width = min(fast['top'].shape[1], eager['top'].shape[1])
            report['failure_stages'] = dict(
                mismatched_rgb_pixels=int(len(rows)),
                selected_id_rows_different=int((fast['top'][:,:width] != eager['top'][:,:width]).any(1).sum()),
                coefficient_order_equal=bool(torch.equal(fast['mapping_kernels'],eager['mapping_kernels']) &
                                              torch.equal(fast['entry_faces'],eager['entry_faces'])),
                first_pixels=selected.cpu().tolist(),
                rgb_actual=fast_rgb[selected].cpu().tolist(), rgb_reference=eager_rgb[selected].cpu().tolist())
            if report['failure_stages']['coefficient_order_equal']:
                report['failure_stages']['corner_heat_max_abs'] = float((fast['corner_heat']-eager['corner_heat']).abs().max()) if fast['corner_heat'].numel() else 0.
                report['failure_stages']['gram_max_abs'] = float((fast['gram']-eager['gram']).abs().max()) if fast['gram'].numel() else 0.
            keys = ('top','weights','outer','response','corner_heat','gram','mapping_kernels','entry_faces','selected_entries')
            snapshot = dict(data=cpu_tree(data), actual_rgb=fast_rgb.cpu(), reference_rgb=eager_rgb.cpu(),
                actual={k:cpu_tree(fast[k]) for k in keys if k in fast},
                reference={k:cpu_tree(eager[k]) for k in keys if k in eager})
            torch.save(snapshot, trainer.run_dir/'render_gate_failure.pt')
    if backend=='hybrid':
        report['operator_checks_passed']=report['passed']
        # Native KNN field is an additional oracle, not the old HKS renderer.
        # Reset native preparation before creating its independent VJP graph.
        with preserve_rng(torch,trainer.device):
            trainer.model.reset(trainer.eigalbo_interp)
            trainer.model.prepare_kernels(trainer.mesh,trainer.eigalbo_interp,save_barycentric=True)
            prepared=UvTextureTrainer.prepare_batch(trainer,data)
            native=UvTextureTrainer.forward_knn(trainer,prepared)
            native_rgb=shade_batch(native[0], data, trainer.surface_options.get('lighting'))
            loss=UvTextureTrainer.compute_loss(trainer,native_rgb,data['colour'].to(trainer.device))[0]
            native_gradients=torch.autograd.grad(loss,parameters,allow_unused=True)
        rgb_ok=bool(torch.allclose(actual,native_rgb.detach(),atol=3e-5,rtol=3e-4))
        gradient_checks=[]
        for (name,_),grad,ref in zip(trainer.model.named_parameters(),gradients,native_gradients):
            gradient_checks.append(dict(parameter=name,passed=(grad is ref if grad is None or ref is None
                else bool(torch.allclose(grad,ref,atol=3e-5,rtol=3e-4))),
                max_abs=None if grad is None or ref is None else float((grad-ref).abs().max())))
        with torch.no_grad():
            _,tape=surface_forward(trainer,data,trace=True,diagnostic_full_trace=True)
            native_ids=native[2][...,0].T;native_weights=native[3][...,0].T
            width=min(native_ids.shape[1],tape['top'].shape[1])
            ni,order=native_ids[:,:width].sort(1)
            hi,horder=tape['top'][:,:width].sort(1)
            contributions=dict(width=width,ids_equal=bool(torch.equal(ni,hi)),
                weights_close=bool(torch.allclose(native_weights[:,:width].gather(1,order),
                    tape['weights'][:,:width].gather(1,horder),atol=3e-5,rtol=3e-4)))
            coverage={}
            if 'outer' in tape:
                # Bound scratch to pixels x support width instead of materializing
                # pixels x native-top-k x support width. Coverage uses full support,
                # not just the HKS-selected top-30, to isolate graph omissions.
                present=torch.stack([(tape['outer']==native_ids[:,i,None]).any(1)
                    for i in range(native_ids.shape[1])],1)
                positive=native_weights>0
                missing=positive&~present
                total=native_weights.clamp_min(0).sum()
                coverage=dict(positive_native_contributions=int(positive.sum()),
                    missing_positive_native_contributions=int(missing.sum()),
                    pixels_with_missing_native_contributions=int(missing.any(1).sum()),
                    missing_native_weight_fraction=float((native_weights.clamp_min(0)*missing).sum()/total.clamp_min(1e-20)),
                    support=tape.get('support'),
                    scope='Native selected positive weights absent from emitted support; not a bound on RGB error')
        report['native_field']=dict(rgb_close=rgb_ok,rgb_max_abs=float((actual-native_rgb.detach()).abs().max()),
            parameters=gradient_checks,top_contributions=contributions,support_coverage=coverage,
            mse_to_native=float((actual-native_rgb.detach()).square().mean()),
            actual_target_mse=float((actual-data['colour'].to(trainer.device)).square().mean()),
            native_target_mse=float((native_rgb.detach()-data['colour'].to(trainer.device)).square().mean()),
            approximate=trainer.surface_options.get('support_domain','graph')!='spectral',
            support_domain=trainer.surface_options.get('support_domain','graph'),
            support_hops=trainer.surface_options.get('surface_support_hops',8))
        report['native_candidate_replay']=native_candidate_replay(trainer,data)
        report['native_field']['equivalence_passed']=rgb_ok and all(c['passed'] for c in gradient_checks)
        report['passed']=report['passed'] and rgb_ok and all(c['passed'] for c in gradient_checks)
        report['passed']=report['passed'] and report['native_candidate_replay']['operator_passed']
        report['failed_checks']=([name for name,passed in (
            ('hks_operator_consistency',report['operator_checks_passed']),
            ('native_field_equivalence',report['native_field']['equivalence_passed']),
            ('replay_operator_consistency',report['native_candidate_replay']['operator_passed'])) if not passed])
        implementation_ok=report['operator_checks_passed'] and report['native_candidate_replay']['operator_passed']
        native_finite=bool(torch.isfinite(native_rgb).all()) and all(g is None or bool(torch.isfinite(g).all()) for g in native_gradients)
        field=report['native_field']
        quality_drop=10*float(np.log10(max(field['actual_target_mse'],1e-20)/max(field['native_target_mse'],1e-20)))
        bound=trainer.surface_options.get('max_initial_psnr_drop_db',.5)
        report['approximation_quality']=dict(initial_psnr_drop_db=quality_drop,maximum_drop_db=bound,
            scope='Initial bounded validation samples only; final held-out quality requires paired training')
        report['approximation_accepted']=bool(trainer.surface_options.get('allow_native_approximation',False) and
            implementation_ok and report['finite'] and native_finite and np.isfinite(quality_drop) and quality_drop<=bound)
        report['gpu_operator_checks_passed']=bool(implementation_ok)
        report['diagnostic_accepted']=False
        if not report['passed']:
            torch.save(dict(data=cpu_tree(data),actual_rgb=actual.cpu(),native_rgb=native_rgb.detach().cpu(),report=report),
                trainer.run_dir/'kernel_support_failure.pt')
    trainer.model.reset(trainer.eigalbo_interp)
    return report


