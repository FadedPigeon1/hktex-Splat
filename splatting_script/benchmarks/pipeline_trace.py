"""Analyze CPU/CUPTI traces using correlated intervals and explicit overlap rules.

Benchmark profiling analysis using the standard library.
Main API: analyze_trace, classify, duration, distribution."""
import math
import statistics
from bisect import bisect_right


STAGES = {
    'update':'both', 'zero_grad':'both', 'backward.loss':'both',
    'forward': 'both', 'loss':'both', 'backward':'both', 'optimizer':'both',
    'optimizer.geodesic':'both', 'optimizer.geodesic_trace':'both',
    'optimizer.other':'both', 'optimizer.scheduler_and_clamp':'both',
    'native.source_state':'native', 'native.faiss_build':'native', 'native.knn_search':'native',
    'native.distances':'native', 'native.spectral_gather':'native', 'native.spectral_heat':'native',
    'native.query_preparation':'native', 'native.distance_weighting':'native',
    'native.source_heat':'native', 'native.shading':'native', 'native.output':'native',
    'backward.native_spectral':'native', 'backward.native_filter':'native',
    'backward.native_rgb':'native', 'backward.native_distance':'native',
    'native.filter':'native', 'native.top30':'native', 'native.rgb':'native',
    'hybrid.source_state':'hybrid', 'source.grid_selection':'both', 'source.spectral_gather':'both',
    'visibility.rasterization':'both', 'visibility.reuse':'both',
    'visibility.rasterization_and_preparation':'both', 'visibility.tile_geometry':'hybrid',
    'support.triangle_embedding_setup':'hybrid', 'support.distances':'hybrid',
    'support.legacy_candidates':'hybrid', 'support.selector_dispatch':'hybrid',
    'support.threshold':'hybrid', 'support.topk':'hybrid', 'support.fused_selection':'hybrid',
    'support.tiled_selection':'hybrid', 'support.tiled_threshold_output':'hybrid',
    'support.sort':'hybrid', 'support.scans':'hybrid', 'support.packing':'hybrid',
    'support.csr':'hybrid', 'support.validation':'hybrid',
    'spectral.coefficient_forward':'hybrid', 'spectral.coefficient_backward':'hybrid',
    'hybrid.queue':'hybrid', 'hybrid.queue_grouping':'hybrid', 'hybrid.emission':'hybrid',
    'hybrid.top30':'hybrid', 'hybrid.rgb':'hybrid', 'backward.rgb':'hybrid', 'backward.weights':'hybrid',
    'hybrid.corner_coefficients':'hybrid', 'hybrid.coefficient_topology':'hybrid',
    'hybrid.splat':'hybrid', 'backward.source_state':'hybrid',
}


def distribution(values):
    values = sorted(float(v) for v in values if v is not None)
    if not values: return None
    return dict(samples=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=values[max(0,math.ceil(.95*len(values))-1)], maximum=max(values))


def union(intervals):
    result = []
    for a,b in sorted(intervals):
        if b <= a: continue
        if result and a <= result[-1][1]: result[-1] = (result[-1][0],max(b,result[-1][1]))
        else: result.append((a,b))
    return result


def duration(intervals):
    return sum(b-a for a,b in union(intervals))


def subtract(interval, children):
    a,b = interval
    cursor = a
    result = []
    for lo,hi in union(children):
        lo,hi = max(a,lo),min(b,hi)
        if lo > cursor: result.append((cursor,lo))
        cursor = max(cursor,hi)
    if cursor < b: result.append((cursor,b))
    return result


def classify(labels, operators):
    names = [x.removeprefix('bench::') for x in labels if x.startswith('bench::')]
    if not names: return [],None
    names = ['native.source_state' if n=='native.source_graph' else n for n in names]
    leaf = names[-1]
    if 'support.legacy_candidates' in names:
        if any('cdist' in op for op in operators): leaf='support.distances'
        elif any('kthvalue' in op for op in operators): leaf='support.threshold'
        elif any('topk' in op for op in operators): leaf='support.topk'
        elif any(op in ('aten::sort','aten::argsort') for op in operators): leaf='support.sort'
    elif 'support.csr' in names and any('cumsum' in op for op in operators): leaf='support.scans'
    if 'hybrid.queue' in names and any('_assert_async' in op for op in operators): leaf='support.validation'
    if 'native.shading' in names:
        if 'kernel_filter' in labels: leaf='native.filter'
        elif any('topk' in op for op in operators): leaf='native.top30'
        elif leaf=='native.shading': leaf='native.rgb'
    return list(dict.fromkeys(names+[leaf])),leaf


def analyze_trace(events, observed):
    backend = observed['backend']
    iterations = len(observed.get('updates',[]))
    cpu_categories = {'cpu_op','user_annotation','cuda_runtime','cuda_driver'}
    complete = [e for e in events if e.get('ph')=='X' and 'dur' in e]
    threads = {}
    for e in complete:
        if e.get('cat') in cpu_categories:
            threads.setdefault((e.get('pid'),e.get('tid')),[]).append(e)
    correlations, external, cpu_self, host_ranges, syncs, allocation_ops = {},{},{},{},[],[]
    nodes_by_thread = {}
    update_counter=0
    forward_sequences={}
    cpu_self_updates={}
    for thread,rows in threads.items():
        stack,nodes = [],[]
        for e in sorted(rows,key=lambda e:(e['ts'],-e['dur'])):
            while stack and e['ts'] >= stack[-1]['end']: stack.pop()
            labels=[n['event']['name'] for n in stack if n['event'].get('cat')=='user_annotation']
            operators=[n['event']['name'] for n in stack if n['event'].get('cat')=='cpu_op']
            if e['cat']=='user_annotation': labels.append(e['name'])
            if e['cat']=='cpu_op': operators.append(e['name'])
            names,leaf=classify(labels,operators)
            args=e.get('args',{})
            sequence=args.get('Sequence number')
            if sequence is not None and ('forward' in names or 'loss' in names) and leaf:
                forward_sequences[(e.get('pid'),e.get('tid'),sequence)]=leaf
            if leaf=='backward' and sequence is not None:
                origin=forward_sequences.get((e.get('pid'),args.get('Fwd thread id',e.get('tid')),sequence))
                kind=('spectral' if origin and (origin.startswith('source.') or 'spectral' in origin or origin=='native.source_state') else
                      'filter' if origin=='native.filter' else 'rgb' if origin in ('native.rgb','native.top30','native.output') else
                      'distance' if origin in ('native.distances','native.distance_weighting') else None)
                if origin=='loss': leaf='backward.loss'
                elif kind and backend=='native': leaf='backward.native_'+kind
                elif backend=='hybrid' and origin and (origin.startswith('source.') or origin=='hybrid.source_state'):
                    leaf='backward.source_state'
                if leaf!='backward': names.append(leaf)
            if leaf=='backward' and stack and (stack[-1]['leaf'] or '').startswith('backward.'):
                leaf=stack[-1]['leaf'];names.append(leaf)
            update=stack[-1]['update'] if stack else None
            if e['name']=='bench::update':
                update=update_counter;update_counter+=1
            node=dict(event=e,end=e['ts']+e['dur'],children=[],names=names,leaf=leaf,update=update,
                      parent=stack[-1]['index'] if stack else None,index=len(nodes))
            if stack: stack[-1]['children'].append((e['ts'],e['ts']+e['dur']))
            nodes.append(node)
            args=e.get('args',{})
            if 'External id' in args: external[args['External id']]=(names,leaf,update)
            if e['cat'] in ('cuda_runtime','cuda_driver'):
                if 'correlation' in args: correlations[args['correlation']]=(names,leaf,update)
                if 'synchronize' in e['name'].lower(): syncs.append(e)
            if e['cat']=='cpu_op' and e['name'] in ('aten::empty','aten::empty_like','aten::empty_strided') and names:
                allocation_ops.append(e)
            if e['name'].startswith('bench::'):
                host_ranges.setdefault(e['name'][7:],[]).append((e['ts'],e['ts']+e['dur']))
            stack.append(node)
        for node in nodes:
            if node['leaf']:
                intervals=subtract((node['event']['ts'],node['end']),node['children'])
                cpu_self.setdefault(node['leaf'],[]).extend(intervals)
                if node['update'] is not None:
                    cpu_self_updates.setdefault(node['leaf'],{}).setdefault(node['update'],[]).extend(intervals)
        nodes_by_thread[thread]=(nodes,[n['event']['ts'] for n in nodes])
    grouped,exclusive,all_gpu,unattributed = {},{},[],[]
    per_update,inclusive_updates,exclusive_updates={},{},{}
    for e in complete:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'): continue
        args=e.get('args',{})
        names,leaf,update=correlations.get(args.get('correlation'),external.get(args.get('External id'),([],None,None)))
        all_gpu.append(e)
        if not leaf: unattributed.append(e)
        for name in names: grouped.setdefault(name,[]).append(e)
        if leaf: exclusive.setdefault(leaf,[]).append(e)
        if update is not None:
            per_update.setdefault(update,[]).append(e)
            for name in names: inclusive_updates.setdefault(name,{}).setdefault(update,[]).append(e)
            if leaf: exclusive_updates.setdefault(leaf,{}).setdefault(update,[]).append(e)

    def gpu_stats(rows):
        if not rows: return None
        kernels=[e for e in rows if e['cat']=='kernel']
        copies=[e for e in rows if e['cat']=='gpu_memcpy']
        spans=[(e['ts'],e['ts']+e['dur']) for e in rows]
        byte_values=[e.get('args',{}).get('Bytes',e.get('args',{}).get('bytes')) for e in copies]
        return dict(kernel_launches=len(kernels), kernel_duration_sum_ms=sum(e['dur'] for e in kernels)/1000,
            activity_union_ms=duration(spans)/1000,
            timeline_span_ms=(max(b for a,b in spans)-min(a for a,b in spans))/1000,
            memcpy_calls=len(copies), memcpy_bytes=sum(int(v) for v in byte_values) if all(v is not None for v in byte_values) else None,
            memcpy_duration_sum_ms=sum(e['dur'] for e in copies)/1000,
            memset_calls=sum(e['cat']=='gpu_memset' for e in rows))

    memory = {}
    for e in events:
        if e.get('name') != '[memory]': continue
        args=e.get('args',{})
        if args.get('Device Type') not in (1,'CUDA','cuda'): continue
        nodes,starts=nodes_by_thread.get((e.get('pid'),e.get('tid')),([],[]))
        at=bisect_right(starts,e['ts'])-1
        owner=None
        while at >= 0:
            node=nodes[at]
            if e['ts']<=node['end'] and node['leaf']:
                owner=node;break
            at=node['parent'] if node['parent'] is not None else -1
        if owner is None: continue
        row=memory.setdefault(owner['leaf'],dict(allocation_events=0,allocation_bytes=0,free_bytes=0,
                    largest_allocation_bytes=0,observed_allocator_highwater_bytes=None))
        size=int(args.get('Bytes',0));row['allocation_events']+=1
        row['allocation_bytes']+=max(0,size);row['free_bytes']+=max(0,-size)
        row['largest_allocation_bytes']=max(row['largest_allocation_bytes'],size)
        total=args.get('Total Allocated')
        if total is not None: row['observed_allocator_highwater_bytes']=max(row['observed_allocator_highwater_bytes'] or 0,int(total))

    sampled={}
    for update in observed.get('updates',[]):
        per_stage={}
        for stage in update['stages']:
            per_stage.setdefault(stage['name'],[]).append(stage)
        for name,rows in per_stage.items():
            dst=sampled.setdefault(name,dict(host=[],process=[],cuda=[],calls=[]))
            dst['calls'].append(len(rows))
            dst['host'].append(sum(r['host_wall_ms'] for r in rows))
            dst['process'].append(sum(r['process_cpu_ms'] for r in rows))
            values=[r['cuda_event_interval_ms'] for r in rows if r['cuda_event_interval_ms'] is not None]
            if values: dst['cuda'].append(sum(values))
    update_times=[r['synchronized_update_ms'] for r in observed.get('updates',[])]
    mean_update=statistics.mean(update_times) if update_times else None
    stages={}
    for name in sorted(set(STAGES)|set(sampled)|set(grouped)|set(cpu_self)):
        present=name in sampled or name in grouped or name in cpu_self
        applicable=STAGES.get(name,'both') in ('both',backend)
        rows=sampled.get(name,{})
        own_cpu=duration(cpu_self.get(name,[]))/1000
        leaf_gpu=gpu_stats(exclusive.get(name,[]))
        stages[name]=dict(status='measured' if present else 'N/A',
            backend=STAGES.get(name,'both'),
            reason=None if present else ('Stage absent in this backend' if not applicable else 'Not executed in the sampled window; cached setup or unavailable instrumentation'),
            host_wall_ms=distribution(rows.get('host',[])+[0]*(iterations-len(rows['host']))) if rows else None,
            process_cpu_ms=distribution(rows.get('process',[])+[0]*(iterations-len(rows['process']))) if rows else None,
            cuda_event_interval_ms=distribution(rows['cuda']+[0]*(iterations-len(rows['cuda']))) if rows.get('cuda') else None,
            calls_per_update=distribution(rows.get('calls',[])+[0]*(iterations-len(rows.get('calls',[])))) if rows else None,
            cpu_self_wall_ms_per_update=own_cpu/iterations if present and iterations else None,
            cpu_self_wall_ms=distribution(duration(v)/1000 for v in cpu_self_updates.get(name,{}).values()),
            inclusive_cuda_activity=gpu_stats(grouped.get(name,[])), exclusive_cuda_activity=leaf_gpu,
            inclusive_gpu_ms_per_update=(gpu_stats(grouped[name])['kernel_duration_sum_ms']/iterations
                if name in grouped and iterations else None),
            exclusive_gpu_kernel_ms=distribution(gpu_stats(v)['kernel_duration_sum_ms'] for v in exclusive_updates.get(name,{}).values()),
            inclusive_gpu_activity_union_ms=distribution(gpu_stats(v)['activity_union_ms'] for v in inclusive_updates.get(name,{}).values()),
            exclusive_gpu_ms_per_update=leaf_gpu['kernel_duration_sum_ms']/iterations if leaf_gpu and iterations else None,
            exclusive_gpu_sum_percent_of_update_wall=(leaf_gpu['kernel_duration_sum_ms']/iterations/mean_update*100
                if leaf_gpu and iterations and mean_update else None),
            memory=memory.get(name))
    gpu_total=gpu_stats(all_gpu)
    if gpu_total:
        gpu_total['observed_gaps_ms']=gpu_total['timeline_span_ms']-gpu_total['activity_union_ms']
    cpu_rank=sorted((dict(stage=n,cpu_self_wall_ms_per_update=v['cpu_self_wall_ms_per_update']) for n,v in stages.items()
        if n not in ('update','forward','backward','optimizer') and v['cpu_self_wall_ms_per_update'] is not None
        and not n.startswith('diagnostic.')),key=lambda r:-r['cpu_self_wall_ms_per_update'])[:5]
    gpu_rank=sorted((dict(stage=n,exclusive_gpu_ms_per_update=v['exclusive_gpu_ms_per_update']) for n,v in stages.items()
        if n not in ('update','forward','backward','optimizer') and v['exclusive_gpu_ms_per_update'] is not None
        and not n.startswith('diagnostic.')),key=lambda r:-r['exclusive_gpu_ms_per_update'])[:5]
    setup_stages={}
    for setup in observed.get('setup',[]):
        for stage in setup.get('stages',[]):
            row=setup_stages.setdefault(stage['name'],dict(calls=0,host_wall_total_ms=0.,
                process_cpu_total_ms=0.,cuda_event_interval_total_ms=None))
            row['calls']+=1
            row['host_wall_total_ms']+=stage['host_wall_ms']
            row['process_cpu_total_ms']+=stage['process_cpu_ms']
            value=stage.get('cuda_event_interval_ms')
            if value is not None: row['cuda_event_interval_total_ms']=(row['cuda_event_interval_total_ms'] or 0.)+value
    return dict(format='hktex_pipeline_profile_v1', backend=backend, profiled_updates=iterations,
        initial_model_sha256=observed.get('initial_model_sha256'), precision=observed.get('precision'),
        controls=observed.get('controls'), health=observed.get('health'), stages=stages,
        synchronized_update_ms=distribution(update_times),
        final_sync_wait_ms=distribution(r['final_sync_wait_ms'] for r in observed.get('updates',[])),
        cuda_activity=gpu_total, unattributed_gpu_events=len(unattributed),
        cuda_kernel_ms_per_update=distribution(gpu_stats(v)['kernel_duration_sum_ms'] for v in per_update.values()),
        observed_gpu_gap_ms_per_update=distribution(gpu_stats(v)['timeline_span_ms']-gpu_stats(v)['activity_union_ms'] for v in per_update.values()),
        synchronization=dict(calls=len(syncs),api_wall_sum_ms=sum(e['dur'] for e in syncs)/1000),
        allocation_operators=dict(calls=len(allocation_ops),cpu_wall_union_ms=duration([(e['ts'],e['ts']+e['dur']) for e in allocation_ops])/1000),
        cpu_bottlenecks=cpu_rank, gpu_bottlenecks=gpu_rank,
        component_memory=observed.get('component_memory'),
        memory_bottlenecks=dict(
            known_tensor_storage=sorted((dict(component=name,**value) for name,value in
                (observed.get('component_memory') or {}).items()),key=lambda row:-row.get('cuda_storage_bytes',row['storage_bytes']))[:5],
            allocation_requests=sorted((dict(stage=name,**value) for name,value in memory.items()),
                key=lambda row:-row['largest_allocation_bytes'])[:5],
            scope='Measured storage snapshots and separate largest allocator requests; categories may alias. Not whole-device ownership.'),
        setup=observed.get('setup',[]),
        setup_stages=setup_stages,
        saved_for_backward=[r.get('saved_for_backward') for r in observed.get('updates',[])],
        work=[dict(iteration=r['iteration'],camera=r['camera'],records=r.get('work',[])) for r in observed.get('updates',[])],
        selector_events=observed.get('selector_events',[]),
        units=dict(time='milliseconds',memory='bytes',trace_input_time='microseconds'),
        scopes=dict(cpu='Self wall time excludes all nested CPU ranges; it includes host waits, not just CPU execution.',
            gpu='CUPTI duration sums may overlap across streams; activity union is separate. Inclusive parent stages are not additive.',
            events='Current-stream CUDA event intervals include host-induced gaps and are not isolated GPU execution.',
            gaps='Whole trace span minus observed activity union; includes between-update/observer gaps, not hardware utilization.',
            memory='Allocator event highwaters are global live bytes observed within a stage, not exclusive ownership or sums.',
            native_rgb='Exclusive residual of native shading after heat/filter/topk; includes weighted-color preparation and reconstruction.',
            native_gradients='Attributed only when CPU autograd sequence/Fwd-thread IDs match a measured forward stage; otherwise retained in total backward.',
            missing='N/A is never converted to a measured zero. Uncorrelated GPU events remain unattributed.'))
