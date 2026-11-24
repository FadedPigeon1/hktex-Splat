"""Compare HKTex (Original, native) and HKTex Splat (Ours, HKS).

Rendering/training entry; HKS is default and preview is GPU-free.
The historical CLI keyword 'hybrid' selects Heat-Kernel Splatting; 'native'
selects original pointwise HKTex. These identifiers remain compatible.
Main API: BACKENDS, main."""
import argparse

BACKENDS = {'hybrid': 'direct', 'native': 'native', 'original': 'native', '2dgs': 'direct'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--backend', choices=tuple(BACKENDS), metavar='{native,hybrid}', default='hybrid')
    args, rest = parser.parse_known_args(argv)
    if '--mode' in rest or any(a.startswith('--mode=') for a in rest):
        parser.error('Select --backend hybrid or native instead of --mode')
    if any(flag in rest for flag in ('-h', '--help')):
        from splatting_script.benchmarks.renderer_comparison import parser as harness_parser
        help_parser = harness_parser()
        help_parser.description = __doc__
        help_parser.add_argument('--backend', choices=tuple(BACKENDS), metavar='{native,hybrid}', default='hybrid',
                                help='native: HKTex (Original); hybrid: HKTex Splat (Ours, default)')
        for action in help_parser._actions:
            if action.dest == 'mode':
                action.help = argparse.SUPPRESS
        help_parser.epilog = 'Compatibility aliases original/2dgs select these same two implementations.'
        help_parser.print_help()
        return
    from splatting_script.benchmarks.renderer_comparison import main as run
    return run(['--mode', BACKENDS[args.backend], *rest])


if __name__ == '__main__':
    main()
