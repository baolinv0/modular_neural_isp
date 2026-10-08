"""Seeded local CPU kernel witness; no private input data required."""
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.special import ndtr
from scipy.integrate import quad
import rawpy
import tifffile

def run():
    rng=np.random.default_rng(19)
    photons=int(rng.poisson(12.5,size=256).sum())
    normal=float(ndtr(0))
    integral,error=quad(lambda x: x*x,0,1)
    assert photons==3146
    assert normal==0.5
    assert abs(integral-1/3)<1e-13
    return {"python":platform.python_version(),
            "versions":{"numpy":np.__version__,"scipy":scipy.__version__,
                        "rawpy":rawpy.__version__,"tifffile":tifffile.__version__},
            "witness":{"seed":19,"poisson_sum":photons,
                       "normal_cdf_zero":normal,"integral_x2":integral,
                       "quadrature_error":error},"status":"PASS"}

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    result=run()
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
