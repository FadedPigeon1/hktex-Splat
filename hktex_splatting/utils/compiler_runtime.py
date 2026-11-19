"""Compare compiler policies recorded in historical benchmark manifests.

Required by compare_training for existing result compatibility; no cache setup runs here.
Main API: matched_policy.
"""
def matched_policy(record):
    """Physical paths/preexisting flags vary by worker; policy must match."""
    return dict(compiler_environment=record['compiler_environment'],
                effective_compile_threads=record['effective_compile_threads'],
                effective_worker_start=record['effective_worker_start'],
                cache_mode=record['cache']['mode'],
                cache_identity_sha256=record['cache'].get('identity_sha256'))
