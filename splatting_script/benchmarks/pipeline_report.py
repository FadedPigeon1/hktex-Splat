"""Build failure-aware timing, memory, quality and scaling reports from saved runs.

Benchmark reporting; no rendering or GPU job is started by readers.
Main API: collect, report, markdown, plots, main."""
import argparse
import csv
import json
import math
from pathlib import Path

from splatting_script.benchmarks.pipeline_trace import distribution, STAGES
from splatting_script.benchmarks.training_research import backend_label, TRAIN_VIEWS


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def number(value):
    return 'N/A' if value is None else f'{value:.4g}' if isinstance(value,(int,float)) else str(value)


def workload(job):
    return (job.get('kernels'),job.get('resolution'),job.get('backend'))


def controls_match(left, right, *, speed_pair=False, scaling_axis=None):
    keys=('initial_model_sha256','assets','config','source_tree_sha256','seed','kernel_count','resolution',
          'cameras','training_views','heldout_views','optimizers','model_config','precision','software','parameters')
    # Across kernel counts initialization/parameter shapes necessarily differ.
    # Across resolutions only camera metadata/image dimensions may differ.
    if scaling_axis=='kernels':
        keys=tuple(k for k in keys if k not in ('initial_model_sha256','kernel_count','parameters'))
    elif scaling_axis=='resolution':
        keys=tuple(k for k in keys if k not in ('resolution','cameras'))
    missing=[k for k in keys if k not in left or k not in right]
    different=[k for k in keys if left.get(k)!=right.get(k)]
    if scaling_axis is None:
        # New records carry exact initialization-file and rendered-input receipts.
        # Historical records remain readable, without claiming these newer checks.
        for key in ('initialization_input','geometry_identity'):
            if key in left or key in right:
                a,b=left.get(key),right.get(key)
                if not speed_pair and key=='geometry_identity':
                    a={n:a[n] for n in TRAIN_VIEWS if a and n in a}
                    b={n:b[n] for n in TRAIN_VIEWS if b and n in b}
                if not a or not b: missing.append(key)
                if a!=b: different.append(key)
    settings=('seed','warmup','visibility','support_cap','footprint_cutoff','coefficient_backend',
                'runtime_counts','selection','splat_backend','max_contributions','max_buffer_mib','support_selector')
    if speed_pair or scaling_axis:
        settings+=('iterations','eval_every','checkpoint_every','progress_every','check_every')
    for key in settings:
        if key not in left.get('arguments',{}) or key not in right.get('arguments',{}):
            missing.append('arguments.'+key)
        if left.get('arguments',{}).get(key)!=right.get('arguments',{}).get(key): different.append('arguments.'+key)
    if left.get('hardware',{}).get('gpu_uuid') in (None,'','unavailable') or right.get('hardware',{}).get('gpu_uuid') in (None,'','unavailable'):
        missing.append('gpu_uuid')
    if left.get('hardware',{}).get('gpu_uuid') != right.get('hardware',{}).get('gpu_uuid'):
        different.append('gpu_uuid')
    return dict(matched=not missing and not different,missing=missing,different=different)


def collect(index):
    workflow=read(Path(index))
    if workflow is None or workflow.get('format')!='hktex_pipeline_workflow_v1':
        raise ValueError('Expected a workflow.json from compare_training workflow')
    records,profiles,manifests,quality_rows=[],{}, {},[]
    failures=[]
    for job in workflow['jobs']:
        if job['role']=='initialization':
            if job.get('status')!='complete': failures.append(job)
            continue
        path=Path(job['run']) if job.get('run') else None
        summary=read(path/'report.json',{}) if path else {}
        manifest=read(path/'research_manifest.json',{}) if path else {}
        status=job.get('status','not_executed')
        if status!='complete': failures.append(job)
        if job['role']=='profile':
            profile=read(path/'pipeline_profile.json') if path else None
            profiles[workload(job)]=dict(profile=profile,manifest=manifest,status=status,run=str(path) if path else None)
            profiles[workload(job)]['validation']=read(path/'profile_validation.json') if path else None
            continue
        if job['role']!='speed': continue
        rows=read(path/'iterations.json',[]) if path else []
        if not rows and path and (path/'iterations.jsonl').exists():
            for line in (path/'iterations.jsonl').read_text().splitlines():
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass  # interrupted final line; failure remains explicit
        history=read(path/'quality.json',[]) if path else []
        if status=='complete' and (not summary or summary.get('status')!='COMPLETE' or not rows or not history):
            status='incomplete_artifacts'
            failures.append(dict(job,status=status))
        diagnostic=bool(summary.get('measurement_diagnostic') or manifest.get('arguments',{}).get('profile')
                        or manifest.get('arguments',{}).get('pipeline_profile') or manifest.get('arguments',{}).get('debug_support'))
        if diagnostic: raise ValueError('A speed record contains diagnostic instrumentation: '+str(path))
        times=[r['seconds']*1000 for r in rows if 'seconds' in r]
        final=history[-1] if history else {}
        splits=final.get('aggregate',{}).get('native_field',{})
        deployed=final.get('aggregate',{}).get('trained_backend',{})
        completion=read(path/'completion.json',{}) if path else {}
        record=dict(kernels=job['kernels'],resolution=job['resolution'],backend=job['backend'],status=status,
            implementation_label=backend_label(job['backend']),
            validation_status='diagnostic_unvalidated' if workflow.get('settings',{}).get('allow_native_variation') else 'strict',
            run=str(path) if path else None, completed_updates=summary.get('completed_updates',len(rows)),
            total_training_seconds=summary.get('measured_training_seconds',sum(times)/1000 if times else None),update_ms=distribution(times),
            cpu_enqueue_ms=distribution(r.get('host_enqueue_seconds',r.get('seconds'))*1000 for r in rows),
            process_cpu_ms=distribution(r.get('process_cpu_seconds')*1000 if r.get('process_cpu_seconds') is not None else None for r in rows),
            loss=distribution(r.get('loss') for r in rows),
            final_training_loss=summary.get('final_training_loss',rows[-1].get('loss') if rows else None),
            process_wall_seconds=job.get('process_wall_seconds'),
            experiment_wall_seconds=completion.get('experiment_wall_seconds'),
            peak_allocated_bytes=summary.get('peak_allocated_bytes'),peak_reserved_bytes=summary.get('peak_reserved_bytes'),
            training_psnr_db=splits.get('train',{}).get('psnr_db'),heldout_psnr_db=splits.get('heldout',{}).get('psnr_db'),
            deployed_training_psnr_db=deployed.get('train',{}).get('psnr_db'),
            deployed_heldout_psnr_db=deployed.get('heldout',{}).get('psnr_db'),
            quality_step=final.get('step'),health=summary.get('final_parameter_validity'),
            selector_diagnostics={name:entry.get('selector',{}) for name,entry in final.get('cameras',{}).items()
                                  if entry.get('selector')},
            geodesic_failed_sample_events=job.get('geodesic_failed_sample_events'),
            geodesic_scope=job.get('geodesic_scope'),jit_cache=summary.get('jit_cache'),
            initial_model_sha256=manifest.get('initial_model_sha256'),source_tree_sha256=manifest.get('source_tree_sha256'))
        records.append(record);manifests[workload(job)]=manifest
        for point in history:
            for family,split_values in point.get('aggregate',{}).items():
                for split,values in split_values.items():
                    if values is not None:
                        quality_rows.append(dict(kernels=job['kernels'],resolution=job['resolution'],backend=job['backend'],
                            implementation_label=backend_label(job['backend']),family=family,
                            step=point['step'],split=split,psnr_db=values['psnr_db'],training_seconds=point['training_seconds']))
    for row in records:
        key=(row['kernels'],row['resolution'],row['backend'])
        profile=profiles.get(key,{})
        row['profile_status']=profile.get('status','not_available')
        row['profile_run']=profile.get('run')
        row['profile_validation']=profile.get('validation')
        match=controls_match(manifests[key],profile.get('manifest',{}))
        row['profile_controls']=match
        row['profile']=profile.get('profile') if match['matched'] else None
    pairs=[]
    for n,res in sorted({(r['kernels'],r['resolution']) for r in records}):
        pair={r['backend']:r for r in records if (r['kernels'],r['resolution'])==(n,res)}
        a,b=pair.get('native'),pair.get('hybrid')
        match=controls_match(manifests.get((n,res,'native'),{}),manifests.get((n,res,'hybrid'),{}),speed_pair=True)
        valid=a and b and a['status']==b['status']=='complete' and match['matched']
        health_ok=bool(a and b and (a.get('health') or {}).get('passed') and (b.get('health') or {}).get('passed'))
        budget_ok=bool(a and b and a['completed_updates']==b['completed_updates']==a['quality_step']==b['quality_step']
            ==manifests.get((n,res,'native'),{}).get('arguments',{}).get('iterations'))
        gates=[j for j in workflow['jobs'] if j['role']=='paired_gate' and (j['kernels'],j['resolution'])==(n,res)]
        gate_required=workflow.get('correctness_protocol')=='paired_preflight_v1'
        gate_ok=bool(len(gates)==1 and gates[0].get('status')=='complete' and
                     gates[0].get('checks') and all(gates[0]['checks'].values()))
        variation_allowed=bool(workflow.get('settings',{}).get('allow_native_variation'))
        valid=valid and health_ok and budget_ok and (gate_ok or not gate_required) and not variation_allowed
        ratio=(a['total_training_seconds']/b['total_training_seconds']
               if valid and a['total_training_seconds'] is not None and b['total_training_seconds'] else None)
        pairs.append(dict(kernels=n,resolution=res,controls=match,health_passed=health_ok,equal_completed_budget=budget_ok,
                          paired_preflight=('diagnostic_unvalidated' if variation_allowed else 'passed' if gate_ok else 'failed_or_missing' if gate_required else 'not_recorded_historical'),
                          timed_update_ratio_native_over_hybrid=ratio,
                          **paired_metrics(a,b,valid),
                          scope='Measured synchronized updates only; excludes setup/evaluation and compares different candidate policies'))
    scaling=[]
    for backend in ('native','hybrid'):
        family=sorted([r for r in records if r['backend']==backend and r['status']=='complete'],key=lambda r:(r['kernels'],r['resolution']))
        for axis,other in (('resolution','kernels'),('kernels','resolution')):
            for fixed in sorted({r[other] for r in family}):
                ordered=sorted([r for r in family if r[other]==fixed],key=lambda r:r[axis])
                for a,b in zip(ordered,ordered[1:]):
                    controls=controls_match(manifests[workload(a)],manifests[workload(b)],scaling_axis=axis)
                    if controls['matched'] and a['update_ms'] and b['update_ms']:
                        scaling.append(dict(backend=backend,axis=axis,fixed_dimension=other,fixed_value=fixed,
                            from_value=a[axis],to_value=b[axis],measured_median_update_ratio=b['update_ms']['median']/a['update_ms']['median']))
    return dict(format='hktex_comprehensive_benchmark_v1',runs=records,pairs=pairs,failures=failures,
        validation_status='diagnostic_unvalidated' if workflow.get('settings',{}).get('allow_native_variation') else 'strict',
        implementations={b:backend_label(b) for b in ('native','hybrid')},
        provenance=[dict(backend=key[2],kernels=key[0],resolution=key[1],manifest=value) for key,value in sorted(manifests.items())],
        workflow_settings=workflow.get('settings',{}),
        quality=quality_rows,scaling=sorted(scaling,key=lambda r:-r['measured_median_update_ratio']),
        estimates=[],hypotheses_scope='Recommendations name measured expensive stages; causes or speedups require separate experiments.',
        units=dict(update_time='milliseconds',total_training_time='seconds',memory='bytes',quality='dB'),
        scope='Speed runs and diagnostic profiles are separate processes. N/A denotes missing/inapplicable measurements. '
              'CPU/CUDA event/CUPTI timings overlap and must not be summed. Tensor storage categories can alias. '
              'Profiler allocator memory is PyTorch memory, not total device/library memory.')


def paired_metrics(original, ours, valid=True):
    """Ratios and signed Ours-minus-Original differences; unavailable is null."""
    def delta(key):
        a,b=(row.get(key) if row else None for row in (original,ours))
        return b-a if valid and a is not None and b is not None and math.isfinite(a) and math.isfinite(b) else None
    a,b=(row.get('peak_allocated_bytes') if row else None for row in (original,ours))
    reduction=100*(a-b)/a if valid and a is not None and a>0 and b is not None else None
    return dict(gpu_memory_reduction_percent=reduction,
        training_psnr_difference_db=delta('training_psnr_db'),heldout_psnr_difference_db=delta('heldout_psnr_db'),
        deployed_training_psnr_difference_db=delta('deployed_training_psnr_db'),
        deployed_heldout_psnr_difference_db=delta('deployed_heldout_psnr_db'))


def write_csv(path, rows, fields):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


PIPELINE_PHASES = (
    ('zero_grad','Clear gradients'), ('visibility.rasterization_and_preparation','Refresh visibility'),
    ('visibility.reuse','Reuse fixed visibility'), ('forward','Forward propagation'),
    ('loss','Calculate loss'), ('backward','Backward propagation'), ('optimizer','Optimizer updates'),
    ('optimizer.geodesic','Move kernels along the mesh'), ('optimizer.other','Other parameter optimizers'),
    ('optimizer.scheduler_and_clamp','Schedulers and parameter constraints'),
)

STAGE_DESCRIPTIONS = {
    'native.source_state':'Prepare learned pointwise kernels', 'native.source_heat':'Cache source heat and normalization',
    'native.faiss_build':'Build source FAISS index', 'native.knn_search':'Find pointwise kernel neighbors',
    'native.query_preparation':'Prepare query points', 'native.distances':'Calculate neighbor distances',
    'native.spectral_gather':'Gather query spectral data', 'native.spectral_heat':'Evaluate pointwise spectral heat',
    'native.distance_weighting':'Weight heat by intrinsic distance', 'native.filter':'Filter kernel contributions',
    'native.top30':'Select strongest 30 contributions', 'native.rgb':'Weighted RGB reconstruction',
    'native.shading':'Pointwise shading (parent)', 'native.output':'Apply model RGB output transform',
    'source.grid_selection':'Select anisotropy/orientation grid', 'source.spectral_gather':'Gather source eigenvectors',
    'hybrid.source_state':'Prepare learned spectral sources', 'support.csr':'Build kernel–triangle support (parent)',
    'support.distances':'Triangle spectral centroid distances', 'support.selector_dispatch':'Dispatch candidate selection',
    'support.legacy_candidates':'Legacy candidate selection', 'support.fused_selection':'Fused candidate selection',
    'support.tiled_selection':'Merge source tiles for candidate selection',
    'support.tiled_threshold_output':'Apply triangle radius/threshold and order support',
    'support.threshold':'Candidate distance threshold', 'support.topk':'Candidate nearest selection',
    'support.sort':'Order selected support', 'support.scans':'Scan support counts',
    'support.packing':'Pack kernel–triangle pairs', 'support.validation':'Check support validity',
    'hybrid.coefficient_topology':'Prepare triangle vertices for coefficients',
    'hybrid.corner_coefficients':'Prepare corner coefficients (parent)',
    'spectral.coefficient_forward':'Evaluate three-corner heat and distance coefficients',
    'hybrid.queue':'Schedule kernel–tile work (parent)', 'hybrid.queue_grouping':'Group kernel–tile queue on GPU',
    'hybrid.splat':'Splat and reconstruct (parent)',
    'hybrid.emission':'Fused barycentric interpolation, distance weighting, filtering and emission',
    'hybrid.top30':'Select strongest 30 contributions', 'hybrid.rgb':'Weighted RGB reconstruction',
    'backward.loss':'Differentiate loss', 'backward.rgb':'Differentiate RGB/color reconstruction',
    'backward.weights':'Differentiate interpolation, weights and filters',
    'spectral.coefficient_backward':'Differentiate spectral corner coefficients',
    'backward.source_state':'Differentiate learned source state',
    'backward.native_spectral':'Differentiate pointwise spectral/source operations',
    'backward.native_filter':'Differentiate pointwise filters',
    'backward.native_rgb':'Differentiate pointwise RGB/top-30 operations',
    'backward.native_distance':'Differentiate pointwise distances/weighting',
    'optimizer.geodesic_trace':'Trace kernel movement on fixed mesh',
    'support.triangle_embedding_setup':'Prepare fixed triangle spectral bounds',
    'visibility.tile_geometry':'Prepare fixed visible pixel/tile geometry',
    'visibility.rasterization':'Rasterize visible triangles',
}
STAGE_DESCRIPTIONS.update(dict(PIPELINE_PHASES))


def inclusive_gpu_ms(profile, stage):
    """Read new and historical profiles using the recorded update denominator."""
    value=profile.get('stages',{}).get(stage,{})
    if 'inclusive_gpu_ms_per_update' in value: return value['inclusive_gpu_ms_per_update']
    activity=value.get('inclusive_cuda_activity') or {}
    total=activity.get('kernel_duration_sum_ms')
    count=profile.get('profiled_updates')
    return total/count if total is not None and count else None


def pipeline_markdown(rows):
    """Keep complete measured pipeline diagnostics in the single retained report."""
    lines=['### Pipeline calculations','',
        'These timings come from separate instrumented profile runs, not the speed runs above. '
        'All times below are milliseconds per profiled update unless marked as setup totals. '
        'GPU time is correlated CUPTI kernel-duration sum, not CPU enqueue time or GPU utilization.','',
        '| Phase | Original host ms | Original GPU ms | HKS host ms | HKS GPU ms |',
        '|---|---:|---:|---:|---:|']
    for stage,label in PIPELINE_PHASES:
        values=[]
        for backend in ('native','hybrid'):
            profile=rows.get(backend,{}).get('profile') or {}
            record=profile.get('stages',{}).get(stage,{})
            values.extend([number((record.get('host_wall_ms') or {}).get('mean')),
                           number(inclusive_gpu_ms(profile,stage))])
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','Parent and child stages overlap: do not add forward to its substeps, or optimizer '
        'to geodesic/other updates. GPU durations can overlap across streams. '
        'N/A means absent, unavailable, unmatched or not measured; it is not zero.','']
    for backend in ('native','hybrid'):
        row=rows.get(backend,{})
        profile=row.get('profile')
        gate=row.get('profile_validation')
        validation=('passed' if gate and gate.get('passed') else
                    'failed' if gate else 'not recorded; observer parity unverified')
        lines+=['#### '+backend_label(backend),'',
            'Profile status: '+str(row.get('profile_status','not_available'))+
            '; matched controls: '+str(row.get('profile_controls',{}).get('matched',False))+
            '; observer RGB/VJP gate: '+validation+'.','']
        if not profile:
            lines+=['Detailed calculations: N/A (missing or unmatched profile).','']
            continue
        lines+=['Profiled updates: '+str(profile.get('profiled_updates','N/A'))+
            '; synchronized diagnostic update mean: '+number((profile.get('synchronized_update_ms') or {}).get('mean'))+
            ' ms; final synchronization wait mean: '+number((profile.get('final_sync_wait_ms') or {}).get('mean'))+' ms.',
            'Unattributed GPU events: '+str(profile.get('unattributed_gpu_events','N/A'))+
            '. Residual work stays in its parent stage; it is not invented as a separately timed calculation.','',
            '| Calculation / recorded scope | Status | Calls/update | Host ms (inclusive) | CPU self wall ms | GPU ms (inclusive) | GPU ms (leaf) |',
            '|---|---|---:|---:|---:|---:|---:|']
        stages=profile.get('stages',{})
        names=list(dict.fromkeys([name for name in STAGES if STAGES[name] in ('both',backend)]+list(stages)))
        for name in names:
            stage=stages.get(name,{})
            if STAGES.get(name,'both') not in ('both',backend) and stage.get('status')!='measured': continue
            if name.startswith('diagnostic.'): continue
            label=STAGE_DESCRIPTIONS.get(name,name.replace('_',' '))
            values=[label+' (`'+name+'`)',str(stage.get('status','N/A')),
                number((stage.get('calls_per_update') or {}).get('mean')),
                number((stage.get('host_wall_ms') or {}).get('mean')),
                number(stage.get('cpu_self_wall_ms_per_update')),
                number(inclusive_gpu_ms(profile,name)), number(stage.get('exclusive_gpu_ms_per_update'))]
            lines.append('| '+' | '.join(values)+' |')
        lines+=['','Host/call values require an explicit observer scope; trace-inferred operations may only '
            'have CPU self wall and GPU timing. CPU self wall includes waits. Leaf GPU time assigns each '
            'correlated kernel to its innermost scope. Native backward substeps require matching autograd '
            'sequence/thread provenance; unclassified gradients remain in total backward.','',
            'HKS interpolation, distance weighting and contribution filtering run together in the emission '
            'kernel; their independent durations cannot be measured from this fused kernel.','']
        setup=profile.get('setup_stages',{})
        if setup:
            lines+=['One-time setup in the diagnostic process (excluded from recurring update time; nested totals overlap):','',
                '| Setup scope | Calls | Host total ms | CUDA event interval total ms |','|---|---:|---:|---:|']
            for name,stage in setup.items():
                lines.append('| '+name+' | '+number(stage.get('calls'))+' | '+number(stage.get('host_wall_total_ms'))+
                    ' | '+number(stage.get('cuda_event_interval_total_ms'))+' |')
            lines+=['','Setup CUDA events include host gaps; they are not isolated kernel durations.','']
        counts={}
        for update in profile.get('work',[]):
            for record in update.get('records',[]):
                for key in ('visible_triangles','kernel_triangle_pairs','expanded_triangle_tile_incidences',
                            'visible_pixels','kernel_tile_items','emitted_pixel_contributions',
                            'selected_contributions','discarded_contributions'):
                    if record.get(key) is not None: counts.setdefault(key,[]).append(record[key])
        if counts:
            lines+=['| Measured work | Mean per sampled forward |','|---|---:|']
            for name,values in counts.items():
                lines.append('| '+name+' | '+number(distribution(values)['mean'])+' |')
            lines+=['']
    return lines


def compact_markdown(data, workflow):
    """Self-contained statistics/failure summary with no links to scratch files."""
    lines=['# HKTex (Original) vs HKTex Splat (Ours)','',
           'Times cover synchronized training updates; process wall includes setup/evaluation. '
           'GPU memory is the PyTorch allocator peak, excluding some library/driver memory. '
           'PSNR differences are Ours minus Original. N/A means unavailable or blocked.','']
    settings=workflow.get('settings',{})
    if settings.get('allow_native_variation'):
        lines+=['**DIAGNOSTIC / UNVALIDATED:** native repeatability was allowed to vary. '
                'Raw statistics are reported; validated speedup and reduction claims are withheld. '
                'Native observer-parity profiling gate is skipped in this mode.','']
    lines+=['Seed: '+str(settings.get('seed','N/A'))+'; requested updates per renderer: '+
            str(settings.get('iterations','N/A'))+'.','']
    metrics=[('Status','status',None),('Training updates','completed_updates',None),
        ('Measured training seconds','total_training_seconds',None),('Process wall seconds','process_wall_seconds',None),
        ('Median update ms','update_ms','median'),('P95 update ms','update_ms','p95'),
        ('Peak allocated GPU bytes','peak_allocated_bytes',None),('Peak reserved GPU bytes','peak_reserved_bytes',None),
        ('Deployed training PSNR dB','deployed_training_psnr_db',None),('Deployed held-out PSNR dB','deployed_heldout_psnr_db',None),
        ('Common native-field training PSNR dB','training_psnr_db',None),('Common native-field held-out PSNR dB','heldout_psnr_db',None),
        ('Final update loss','final_training_loss',None)]
    for pair in data['pairs']:
        n,res=pair['kernels'],pair['resolution']
        rows={r['backend']:r for r in data['runs'] if (r['kernels'],r['resolution'])==(n,res)}
        lines += [f'## {n} kernels, {res} × {res}','',
                  '| Metric | HKTex (Original) | HKTex Splat (Ours) |','|---|---:|---:|']
        for title,key,sub in metrics:
            values=[]
            for backend in ('native','hybrid'):
                value=rows.get(backend,{}).get(key)
                if sub: value=(value or {}).get(sub)
                values.append(number(value))
            lines.append('| '+title+' | '+' | '.join(values)+' |')
        selectors=rows.get('hybrid',{}).get('selector_diagnostics',{})
        if selectors:
            algorithms=sorted({s.get('selection_algorithm','not_recorded') for s in selectors.values()})
            ties=sorted({s.get('selection_tie_policy','not_recorded') for s in selectors.values()})
            fallbacks=sorted({str(s[k]) for s in selectors.values() for k in
                ('selector_fallback','native_distance_fallback','native_topk_fallback') if s.get(k)})
            lines+=['', 'HKS selector: '+', '.join(algorithms)+'; tie policy: '+', '.join(ties)+
                    '; reported fallbacks: '+(', '.join(fallbacks) if fallbacks else 'none')+'.']
        lines+=['', 'Training speedup (Original/Ours): '+number(pair['timed_update_ratio_native_over_hybrid'])+'×; '
                'allocated-memory reduction: '+number(pair['gpu_memory_reduction_percent'])+'%.',
                'Deployed PSNR difference (train / held-out): '+number(pair['deployed_training_psnr_difference_db'])+
                ' / '+number(pair['deployed_heldout_psnr_difference_db'])+' dB.',
                'Common native-field PSNR difference (train / held-out): '+number(pair['training_psnr_difference_db'])+
                ' / '+number(pair['heldout_psnr_difference_db'])+' dB.',
                'Matched controls: '+str(pair['controls']['matched'])+'; paired preflight: '+pair['paired_preflight']+'.','']
        if pair['controls']['missing'] or pair['controls']['different']:
            lines+=['Missing controls: '+str(pair['controls']['missing'])+'; different: '+str(pair['controls']['different'])+'.','']
        for backend,row in rows.items():
            lines+=['- '+backend_label(backend)+': profile '+row['profile_status']+
                    '; initial model SHA256 `'+str(row['initial_model_sha256'])+
                    '`; source SHA256 `'+str(row['source_tree_sha256'])+'`.']
            profile=row.get('profile')
            if profile:
                lines+=['  Top measured GPU stages: '+', '.join(stage['stage'] for stage in profile.get('gpu_bottlenecks',[])[:3])+'.']
        lines+=['']
        lines+=pipeline_markdown(rows)
    lines+=['## Job status','', '| Stage | Status |','|---|---|']
    for job in workflow['jobs']:
        status=job.get('status','not_executed')
        if job.get('diagnostic_allowed'): status+=' (diagnostic; strict check failed)'
        lines.append('| '+job['id']+' | '+status+' |')
    failed=[j for j in workflow['jobs'] if j.get('diagnostic_allowed') or
            (j.get('status') not in (None,'complete') and not j.get('status','').startswith('skipped'))]
    if failed: lines+=['','## Failures','']
    for job in failed:
        lines+=['### '+job['id'],'']
        tail=job.get('error_tail') or job.get('error')
        if tail: lines+=['```text',*str(tail).splitlines()[-8:],'```','']
        if job.get('run'):
            path=Path(job['run'])
            replay=read(path/'preflight_replay.json',{})
            for row in replay.get('tensor_comparison',{}).get('tensors',[]):
                if not row['passed']:
                    lines+=['- '+row['name']+': '+str(row.get('mismatched_elements','N/A'))+
                            ' mismatches; maximum absolute difference '+number(row.get('max_abs'))+'.']
            loss=read(path/'preflight_diagnostics.json',{}).get('training_step',{})
            if loss and not loss.get('passed'):
                lines+=['- Training loss '+number(loss.get('loss'))+'; reference '+number(loss.get('reference_loss'))+
                        '; absolute difference '+number(loss.get('abs_difference'))+'.']
        if not tail: lines+=['Checks: '+str(job.get('checks',{}))+'.']
        lines+=['']
    lines+=['Loss is channel-summed foreground RGB squared error averaged over pixels. '
            'PSNR uses equal-camera mean foreground MSE. Deployed PSNR evaluates each renderer; '
            'common native-field PSNR uses pointwise HKTex for both learned states.','',
            'Only this Markdown report is retained. New initialization, snapshots, logs, images, '
            'traces and compiler caches were temporary. Use --keep-artifacts for full research records.','']
    return '\n'.join(lines)


def markdown(data):
    text=['# HKTex (Original) versus HKTex Splat (Ours)','',data['scope'],'',
          'PSNR differences are Ours minus Original. Common native-field PSNR evaluates both learned states '
          'with the original pointwise renderer; deployed PSNR evaluates each actual renderer. '
          'Loss is channel-summed foreground RGB squared error, averaged over pixels. Final loss is the last '
          'update loss before its optimizer step, not an all-camera evaluation loss.','']
    if data.get('validation_status')=='diagnostic_unvalidated':
        text+=['**DIAGNOSTIC / UNVALIDATED:** native variation was allowed; validated ratios/deltas are withheld. '
               'The native observer-parity profile gate is skipped.','']
    for pair in data['pairs']:
        text += [f"## {pair['resolution']}×{pair['resolution']}, {pair['kernels']} kernels",'']
        rows={r['backend']:r for r in data['runs'] if (r['kernels'],r['resolution'])==(pair['kernels'],pair['resolution'])}
        text+=['| Metric | HKTex (Original) | HKTex Splat (Ours) |','|---|---:|---:|']
        for title,key,sub in [('Status','status',None),('Training iterations','completed_updates',None),
            ('Total measured training seconds','total_training_seconds',None),
            ('Complete process wall seconds (includes setup/evaluation)','process_wall_seconds',None),
            ('Update mean (ms)','update_ms','mean'),('Update median (ms)','update_ms','median'),
            ('Update p95 (ms)','update_ms','p95'),('Update maximum (ms)','update_ms','maximum'),
            ('Host enqueue mean (ms)','cpu_enqueue_ms','mean'),('Process CPU mean (ms)','process_cpu_ms','mean'),
            ('Peak allocated bytes','peak_allocated_bytes',None),('Peak reserved bytes','peak_reserved_bytes',None),
            ('Quality evaluated at update','quality_step',None),
            ('Common native-field training PSNR (dB)','training_psnr_db',None),('Common native-field held-out PSNR (dB)','heldout_psnr_db',None),
            ('Deployed training PSNR (dB)','deployed_training_psnr_db',None),('Deployed held-out PSNR (dB)','deployed_heldout_psnr_db',None),
            ('Final training loss','final_training_loss',None)]:
            values=[]
            for backend in ('native','hybrid'):
                value=rows.get(backend,{}).get(key)
                if sub: value=value.get(sub) if value else None
                values.append(number(value))
            text.append('| '+title+' | '+' | '.join(values)+' |')
        text+=['', 'Matched controls: '+str(pair['controls']['matched'])+'. Healthy runs: '+str(pair['health_passed'])+
               '. Equal completed budget: '+str(pair['equal_completed_budget'])+'. Timed-update ratio: '+number(pair['timed_update_ratio_native_over_hybrid'])+'.','']
        text+=['Paired preflight: '+pair['paired_preflight']+'.',
            'Training speedup (Original / Ours): '+number(pair['timed_update_ratio_native_over_hybrid'])+'×. '
            'Peak allocated GPU memory reduction: '+number(pair['gpu_memory_reduction_percent'])+'%.',
            'Common native-field PSNR difference (training / held-out): '+number(pair['training_psnr_difference_db'])+' / '+number(pair['heldout_psnr_difference_db'])+' dB.',
            'Deployed PSNR difference (training / held-out): '+number(pair['deployed_training_psnr_difference_db'])+' / '+number(pair['deployed_heldout_psnr_difference_db'])+' dB.',
            'Missing controls: '+str(pair['controls']['missing'])+'. Differing controls: '+str(pair['controls']['different'])+'.','']
        text+=pipeline_markdown(rows)
        for backend,row in rows.items():
            text += ['### '+backend_label(backend),'', 'Initial model SHA256: `'+str(row['initial_model_sha256'])+'`; source tree SHA256: `'+str(row['source_tree_sha256'])+'`. Full controls/assets/software are in benchmark_report.json and the original manifests.',
                '', 'Profile status: '+row['profile_status']+'. Profile controls matched: '+str(row['profile_controls']['matched'])+'.','']
            gate=row.get('profile_validation')
            gate_status=('passed on '+gate['device'] if gate and gate.get('passed') else
                         'failed: '+str(gate.get('failure_reason','unspecified')) if gate else 'N/A: not verified')
            text+=['Observer correctness gate: '+gate_status+'.','']
            profile=row['profile']
            if not profile:
                text+=['Detailed timings/memory: N/A (missing or unmatched diagnostic profile).',''];continue
            text+=[
                   'Top CPU self-wall bottlenecks (includes waits): '+', '.join(r['stage'] for r in profile['cpu_bottlenecks'])+'.',
                   'Top GPU exclusive-kernel bottlenecks: '+', '.join(r['stage'] for r in profile['gpu_bottlenecks'])+'.','',
                   '| Measured storage component | Bytes |','|---|---:|']
            memory=profile.get('component_memory') or {}
            for name,value in sorted(memory.items(),key=lambda r:-r[1]['storage_bytes']):
                text.append('| '+name+' | '+number(value['storage_bytes'])+' |')
            spikes=sorted(((name,value['memory']['largest_allocation_bytes']) for name,value in profile['stages'].items()
                if value.get('memory')),key=lambda row:-row[1])[:5]
            text+=['','Largest observed allocator requests: '+(', '.join(name+' ('+str(size)+' bytes)' for name,size in spikes) or 'N/A')+'.']
            saved=[r['other_storage_bytes'] for r in profile.get('saved_for_backward',[]) if r]
            text+=['','Largest sampled saved-for-backward storage outside known shared GPU components: '+number(max(saved) if saved else None)+' bytes.']
            gpu=profile.get('cuda_activity') or {}
            text+=['Observed GPU timeline gaps: '+number((profile.get('observed_gpu_gap_ms_per_update') or {}).get('mean'))+
                   ' ms/update (within correlated activity spans; not hardware idle utilization).',
                   'Unattributed GPU events: '+str(profile.get('unattributed_gpu_events','N/A'))+'.',
                   'Full trace activity union: '+number(gpu.get('activity_union_ms'))+' ms; kernel-duration sum: '+number(gpu.get('kernel_duration_sum_ms'))+' ms.']
            text+=['','Storage categories may alias; they are not additional allocation bytes. Per-stage allocator events and saved-for-backward storage are in JSON/CSV.','']
            targets=[r['stage'] for r in profile['gpu_bottlenecks'][:2]]
            text+=['Measured optimization targets: '+(', '.join(targets) if targets else 'N/A: no correlated GPU activity')+
                   '. This ranks observed costs, not predicted speedups.','']
    text+=['## Scaling','']
    for row in data['scaling'][:5]:
        text.append(f"- {backend_label(row['backend'])}: {row['axis']} {row['from_value']} → {row['to_value']}, measured median-update ratio {number(row['measured_median_update_ratio'])}.")
    if not data['scaling']: text+=['N/A: fewer than two measured comparable sizes.']
    text+=['','## Failures and skipped configurations','']
    for job in data['failures']: text.append('- '+job['id']+': '+job.get('status','not_executed')+'; log '+str(job.get('log','N/A')))
    if not data['failures']: text.append('None recorded.')
    return '\n'.join(text)+'\n'


def plots(data, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    created=[]
    for metric,axis in (('update_ms','resolution'),('update_ms','kernels'),('peak_allocated_bytes','resolution'),
                        ('peak_allocated_bytes','kernels')):
        fig,ax=plt.subplots(figsize=(7,4));other='kernels' if axis=='resolution' else 'resolution'
        for backend,fixed in sorted({(r['backend'],r[other]) for r in data['runs']}):
            rows=sorted([r for r in data['runs'] if r['backend']==backend and r[other]==fixed and r['status']=='complete' and r.get(metric) is not None],key=lambda r:r[axis])
            if rows: ax.plot([r[axis] for r in rows],[r[metric]['median'] if metric=='update_ms' else r[metric]/1024**3 for r in rows],marker='o',label=f'{backend_label(backend)}, {other}={fixed}')
        ax.set(xlabel=axis,ylabel='median update ms' if metric=='update_ms' else 'peak allocated GiB')
        if ax.lines: ax.legend(fontsize=7)
        else: ax.text(.5,.5,'N/A: no completed measurements',ha='center',transform=ax.transAxes)
        fig.tight_layout();name=f'{metric}_vs_{axis}.png';fig.savefig(directory/name,dpi=160);plt.close(fig);created.append(name)
    for family,filename in (('native_field','quality_vs_updates.png'),('trained_backend','deployed_quality_vs_updates.png')):
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax,split in zip(axes,('train','heldout')):
            for n,res,backend in sorted({(r['kernels'],r['resolution'],r['backend']) for r in data['quality']}):
                rows=[r for r in data['quality'] if (r['kernels'],r['resolution'],r['backend'],r['split'],r['family'])==(n,res,backend,split,family)]
                if rows: ax.plot([r['step'] for r in rows],[r['psnr_db'] for r in rows],marker='.',label=f'{backend_label(backend)} {n}/{res}')
            ax.set(xlabel='updates',ylabel=('Common native-field' if family=='native_field' else 'Deployed renderer')+' PSNR dB',title=split)
            if ax.lines: ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(directory/filename,dpi=160);plt.close(fig);created.append(filename)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,metric,title,scale in zip(axes,('total_training_seconds','peak_allocated_bytes'),
                                   ('Total measured training seconds','Peak allocated GPU GiB'),(1,1024**3)):
        for i,backend in enumerate(('native','hybrid')):
            workloads=sorted({(r['kernels'],r['resolution']) for r in data['runs']})
            rows={(r['kernels'],r['resolution']):r for r in data['runs'] if r['backend']==backend and r['status']=='complete'}
            values=[rows.get(key,{}).get(metric) for key in workloads]
            ax.bar([j+(i-.5)*.36 for j in range(len(workloads))],
                   [v/scale if v is not None else float('nan') for v in values],width=.36,label=backend_label(backend))
        ax.set_xticks(range(len(workloads)),[f'{n} / {res}px' for n,res in workloads],rotation=20)
        ax.set_ylabel(title);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(directory/'training_time_and_memory.png',dpi=160);plt.close(fig);created.append('training_time_and_memory.png')
    for name,stages in (('forward_backward_optimizer',('forward','backward','optimizer')),
                        ('candidate_and_spectral',('support.distances','support.topk','support.fused_selection','spectral.coefficient_forward','spectral.coefficient_backward','native.spectral_heat'))):
        fig,ax=plt.subplots(figsize=(10,4));labels=[];series={stage:[] for stage in stages}
        for row in data['runs']:
            if not row['profile']: continue
            labels.append(f"{backend_label(row['backend'])} {row['kernels']}/{row['resolution']}")
            for stage in stages:
                value=row['profile']['stages'].get(stage,{})
                value=((value.get('cuda_event_interval_ms') or {}).get('mean') if name=='forward_backward_optimizer'
                       else value.get('exclusive_gpu_ms_per_update'))
                series[stage].append(float('nan') if value is None else value)
        for i,(stage,values) in enumerate(series.items()):
            if labels: ax.bar([x+i*.8/len(stages) for x in range(len(labels))],values,width=.8/len(stages),label=stage)
        ax.set_xticks(range(len(labels)),labels,rotation=20,ha='right')
        ax.set_ylabel('CUDA event interval ms (diagnostic)' if name=='forward_backward_optimizer' else 'exclusive CUPTI kernel ms/update (diagnostic)')
        if labels: ax.legend(fontsize=7)
        else: ax.text(.5,.5,'N/A: no matched profiles',ha='center',transform=ax.transAxes)
        fig.tight_layout();fig.savefig(directory/(name+'.png'),dpi=160);plt.close(fig);created.append(name+'.png')
    return created


def report(index, output, make_plots=True):
    data=collect(index);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    stage_rows=[];memory_rows=[];work_rows=[]
    for run in data['runs']:
        profile=run['profile']
        if not profile: continue
        prefix={k:run[k] for k in ('kernels','resolution','backend')}
        for name,value in profile['stages'].items():
            timing=dict(prefix,stage=name,status=value['status'],
                host_wall_mean_ms=(value.get('host_wall_ms') or {}).get('mean'),
                calls_per_update=(value.get('calls_per_update') or {}).get('mean'),
                cpu_self_wall_ms_per_update=value.get('cpu_self_wall_ms_per_update'),
                inclusive_gpu_ms_per_update=inclusive_gpu_ms(profile,name),
                exclusive_gpu_ms_per_update=value.get('exclusive_gpu_ms_per_update'),
                gpu_percent_update_wall=value.get('exclusive_gpu_sum_percent_of_update_wall'))
            for metric in ('host_wall_ms','process_cpu_ms','cuda_event_interval_ms','cpu_self_wall_ms','exclusive_gpu_kernel_ms'):
                for stat in ('mean','median','p95','maximum'):
                    timing[metric+'_'+stat]=(value.get(metric) or {}).get(stat)
            stage_rows.append(timing)
            if value.get('memory'): memory_rows.append(dict(prefix,stage=name,kind='allocator_events',**value['memory']))
        for name,value in (profile.get('component_memory') or {}).items():
            memory_rows.append(dict(prefix,stage=name,kind='measured_tensor_storage',**value))
        for index,value in enumerate(profile.get('saved_for_backward',[])):
            if value: memory_rows.append(dict(prefix,stage='forward/loss_boundary',profile_sample=index,
                kind='measured_saved_tensor_storage',**value))
        for update in profile.get('work',[]):
            for record in update['records']: work_rows.append(dict(prefix,iteration=update['iteration'],camera=update['camera'],**record))
    def flat_csv(name,rows):
        labelled=[dict(row,implementation_label=backend_label(row['backend'])) if row.get('backend') else row for row in rows]
        flat=[{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in labelled]
        fields=sorted({k for row in flat for k in row}) or ['status']
        write_csv(output/name,flat,fields)
    flat_csv('overall.csv',[{k:v for k,v in row.items() if k!='profile'} for row in data['runs']])
    flat_csv('pairs.csv',data['pairs'])
    flat_csv('component_timing.csv',stage_rows);flat_csv('memory.csv',memory_rows);flat_csv('work_counts.csv',work_rows)
    flat_csv('quality.csv',data['quality']);flat_csv('scaling.csv',data['scaling']);flat_csv('failures.csv',data['failures'])
    data['plots']=plots(data,output) if make_plots else []
    (output/'benchmark_report.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    (output/'report.md').write_text(markdown(data))
    return data


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('workflow',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--no-plots',action='store_true')
    args=p.parse_args(argv);report(args.workflow,args.output,not args.no_plots)


if __name__=='__main__': main()
