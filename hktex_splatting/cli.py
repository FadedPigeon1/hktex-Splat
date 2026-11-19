"""Route Heat-Kernel Splatting and original HKTex rendering and training commands.

The matched harness is GPU-free until --execute. Original HKTex stays in hktex.
Main API: main.
"""
def main(argv=None):
    from splatting_script.experiments.compare_backends import main as run
    return run(argv)
