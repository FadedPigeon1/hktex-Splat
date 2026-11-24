"""Compare exact HKS distance layouts without launching training.

CPU interpreter checks and offline SM80 compilation cannot predict GPU speed.
Explicit --execute times matched distance tables and support cutoffs on CUDA.
Only a Markdown report is written. Legacy gather reductions remain numerical
and timing references for the compact, operation-order-preserving reduction.
"""
import argparse
import os
import tempfile
from pathlib import Path

LAYOUTS = (('reference_source8',1,8), ('reference_face4_source16',4,16), ('source8', 1, 8), ('source32', 1, 32), ('source64', 1, 64),
           ('face2_source16', 2, 16), ('face4_source8', 4, 8), ('face4_source16', 4, 16))


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--compile-sm80', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reverse', action='store_true', help='Reverse layout order for a repeat timing run')
    parser.add_argument('--pruning',action='store_true',help='Also compare opt-in conservative prefix pruning, including its seed/bound overhead')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if sum((args.cpu, args.compile_sm80, args.execute)) > 1:
        parser.error('Select exactly one execution mode')
    if not any((args.cpu, args.compile_sm80, args.execute)):
        print('Preview: two legacy references plus six compact exact layouts; --cpu checks CPU interpreter, --compile-sm80 compiles offline, --execute benchmarks CUDA. No training.')
        return 0
    if args.cpu: os.environ['TRITON_INTERPRET'] = '1'
    elif os.environ.get('TRITON_INTERPRET') == '1': parser.error('Unset TRITON_INTERPRET for compilation/CUDA')
    if args.output and args.output.exists(): parser.error('Use a fresh report path')
    layouts = LAYOUTS[::-1] if args.reverse else LAYOUTS
    os.environ.setdefault('TRITON_CACHE_DIR', str(Path(tempfile.gettempdir())/
                          ('hktex_distance_study_'+str(os.getuid()))/'triton'))
    import torch
    import triton
    from hktex_splatting.triton import resident_gpu as gpu
    lines = ['# Exact spectral-distance layout study', '',
             'Legacy versus compact exact reductions. CPU timings are not GPU speed estimates.', '']
    if args.compile_sm80:
        from triton.compiler import ASTSource
        from triton.backends.compiler import GPUTarget
        lines += ['| Layout | Cubin bytes | Shared bytes |', '|---|---:|---:|']
        variants=(*layouts,('prefix_face4_source16',4,16)) if args.pruning else layouts
        for name, bf, bs in variants:
            fn = gpu.support_distances if bf == 1 else gpu.support_distances_blocked
            signature = dict(CENTER='*fp64', SOURCE='*fp32', DISTANCE='*fp64', N='i32',
                             CS0='i32', CS1='i32', SS0='i32', SS1='i32')
            constants = dict(DIM=64, BD=64, CUDA_SQRT=True,COMPACT=not name.startswith('reference_'))
            if bf == 1: constants['PAIRS'] = bs
            else:
                signature['M'] = 'i32'; constants.update(BF=bf, BS=bs)
                constants.update(PRUNE=name.startswith('prefix'),PREFIX=16)
                if name.startswith('prefix'):signature['UPPER']='*fp64'
                else:constants['UPPER']=None
            compiled = triton.compile(ASTSource(fn, signature=signature, constexprs=constants),
                target=GPUTarget('cuda', 80, 32), options=dict(num_warps=4, enable_fp_fusion=False))
            row = f'| {name} | {len(compiled.asm["cubin"])} | {compiled.metadata.shared} |'
            print(row, flush=True); lines.append(row)
        lines += ['', 'Compilation/resource counts do not establish speed or actual occupancy.']
    else:
        if args.execute:
            if not torch.cuda.is_available(): raise RuntimeError('An allocated CUDA GPU is required')
            device = 'cuda'
            lines += ['GPU: '+torch.cuda.get_device_name(), '', '| Faces | Sources | Layout | Median ms | Exact distance/support |', '|---:|---:|---|---:|---|']
        else:
            device = 'cpu'; torch.set_num_threads(1)
        def launch(c, s, out, bf, bs, prune=False,compact=True):
            m, n, dim = len(c), len(s), c.shape[1]
            common = dict(num_warps=4, enable_fp_fusion=False)
            bd = max(32, triton.next_power_of_2(dim))
            if bf == 1:
                gpu.support_distances[(triton.cdiv(n, bs), m)](c, s, out, n,
                    *c.stride(), *s.stride(), dim, bd, bs, args.execute, COMPACT=compact, **common)
            else:
                upper=None
                if prune:
                    rank=min(10,n)
                    seed=torch.empty((m,rank),device=device,dtype=torch.float64)
                    upper=torch.empty(m,device=device,dtype=torch.float64)
                    gpu.support_distances[(triton.cdiv(rank,8),m)](c,s,seed,rank,*c.stride(),*s.stride(),
                        dim,bd,8,args.execute,**common)
                    gpu.support_seed_upper[(m,)](seed,torch.full((m,),.1,device=device,dtype=torch.float64),
                        upper,rank,1e-5,triton.next_power_of_2(rank),**common)
                gpu.support_distances_blocked[(triton.cdiv(n, bs), triton.cdiv(m, bf))](
                    c, s, out, m, n, *c.stride(), *s.stride(), dim, bd, bf, bs,args.execute,
                    UPPER=upper,PRUNE=prune,PREFIX=min(16,dim),COMPACT=compact,**common)
        cases = [(128, 5000, 64), (512, 5000, 64)] if args.execute else [(7, 35, d) for d in (2, 31, 32, 64, 65, 256)]
        for m, n, dim in cases:
            gen = torch.Generator(device=device).manual_seed(23+dim)
            # Strided views and row/source tails exercise real indexing/masks.
            c = torch.randn(m, dim*2, generator=gen, device=device, dtype=torch.float64)[:, ::2]
            s = torch.randn(n, dim*2, generator=gen, device=device)[:, ::2]
            if args.execute: c, s = c.contiguous(), s.contiguous()
            reference = torch.empty(m, n, device=device, dtype=torch.float64)
            launch(c, s, reference, 1, 8,compact=False)
            radius = torch.full((m,), .1, device=device, dtype=torch.float64)
            cap, rank = min(16, n-1), min(10, n)
            def support(distance):
                rows = torch.empty(m, cap+1, device=device, dtype=torch.int32)
                gpu.select_support_row_radix[(m,)](distance, radius, rows, n, rank, cap, 0, cap+1,
                    1e-5, triton.next_power_of_2(n), num_warps=8, enable_fp_fusion=False)
                return rows
            expected_rows = support(reference)
            variants=(*layouts,('prefix_face4_source16',4,16)) if args.pruning else layouts
            for name, bf, bs in variants:
                prune=name.startswith('prefix')
                output = torch.empty_like(reference)
                launch(c, s, output, bf, bs,prune,compact=not name.startswith('reference_'))
                if prune:
                    kept=output.isfinite()
                    torch.testing.assert_close(output[kept],reference[kept],atol=0,rtol=0,equal_nan=True)
                else:torch.testing.assert_close(output, reference, atol=0, rtol=0, equal_nan=True)
                assert torch.equal(support(output), expected_rows), 'Support differs: '+name
                if args.execute:
                    # Warm JIT before CUDA-event timing; independent launch batches
                    # give distribution without including compilation or validation.
                    samples = []
                    for _ in range(7):
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(10): launch(c, s, output, bf, bs,prune,compact=not name.startswith('reference_'))
                        stop.record(); stop.synchronize(); samples.append(start.elapsed_time(stop)/10)
                    median = sorted(samples)[len(samples)//2]
                    row = f'| {m} | {n} | {name} | {median:.6f} | pass |'
                    print(row, flush=True); lines.append(row)
                else: print(f'PASS exact distances and support: dimension={dim}, layout={name}', flush=True)
        if args.cpu: lines += ['CPU interpreter: '+str(len(layouts)+int(args.pruning))+' layouts × six dimensions passed retained-distance/support checks, strided inputs and tails.']
        else: lines += ['', 'Timing scope: distance computation and output stores; pruning also includes seed/bound launches and buffer allocation. Pruned slots may become infinity; retained distances and final support IDs must agree exactly. Full training and gradients require separate validation. Rerun with rotated layout order before selecting a default.']
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream: stream.write('\n'.join(lines)+'\n')
        print('DISTANCE_REPORT='+str(args.output))
    return 0


if __name__ == '__main__': raise SystemExit(main())
