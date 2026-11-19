"""Provide the recorded fixed training and held-out camera definitions.

Shared rendering/training/benchmark geometry inputs.
Main API: camera, cameras."""
import copy
from hktex_splatting.utils.experiment_io import CAMERAS, VIEWS

def camera(view, resolution):
    # The original camera helper restricts resolution to 512/1024. Reuse its
    # established basis at 512, then change only pixel dimensions for the smoke test.
    from hktex_splatting.utils.reference_assets import camera as original_camera
    c = original_camera(view, 512, CAMERAS)
    return dict(c, width=resolution, height=resolution)

def cameras():
    import numpy as np
    out={v:camera(v,500) for v in VIEWS}
    # Fixed BEFORE quality evaluation, unrelated to support construction.
    # Both methods use these same independent views; no tuning on them.
    for name,base,angle in [('heldout_left','side',27),('heldout_right','three_quarter',-31)]:
        c=copy.deepcopy(out[base]);p=np.array(c['position'])-c['target']
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        c['position']=(rot@p+np.array(c['target'])).tolist();out[name]=c
    return out
