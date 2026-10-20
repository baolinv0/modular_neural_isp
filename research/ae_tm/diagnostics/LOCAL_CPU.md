# Local CPU diagnostic environment

Python 3.12.14 was used. These diagnostics use NumPy/SciPy, not the Torch training pipeline. The installed package versions are locked in requirements.txt.

Create an isolated environment on a machine with Python 3.12:
~~~powershell
python -m venv .venv
$CapturePython = '.\.venv\Scripts\python.exe'
& $CapturePython -m pip install numpy==2.3.5
& $CapturePython -m pip install -r research/ae_tm/diagnostics/requirements.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $CapturePython -X utf8 research/ae_tm/diagnostics/environment_witness.py --output work/local-cpu-witness.json
& $CapturePython -X utf8 -m pytest research/ae_tm/tests/test_dng_audit.py -q
~~~

Set CapturePython to an existing documented isolated Python if reusing a warm environment. Expected witness: status PASS, seed19 Poisson sum3146, normal CDF0=0.5, integral x²=1/3. An independent agent must follow the actual local invocation before environment acceptance.

Audit user-owned data:
~~~powershell
& $CapturePython -X utf8 research/ae_tm/diagnostics/dng_audit.py --data-root '<user-data-directory>' --protocol research/ae_tm/diagnostics/DNG_PROTOCOL.json --output '<local-report-directory>' --cache '<local-pixel-cache-directory>'
~~~

Paths are CLI parameters. Do not commit raw files, photos, contact sheets, pixel caches or absolute private data paths. The script reads one DNG buffer at a time and emits 8 green64×64 patches per source. The published audit contains only source hashes, relative names, format metadata, approximate strided source statistics and patch positions.

The declared data contains 34 paired iPhone/S25 sources from one capture session, grouped into11 conservative background groups. The two cameras within a pair are not independent. Group assignment precedes all crop/noise generation. One-session diagnostic intervals are descriptive; source count is not independent scene count. Existing GT JPG and linear PNG files are checked for presence but not consumed as clean or expert ground truth.

Both cameras' inspected RAW IFDs are processed three-channel Linear RAW. iPhone may store10-bit nonlinear codes with a linearization table; LibRaw exposes16-bit linear values. S25 stores12-bit linear values. Do not read only the root JPEG preview IFD. Native Bayer/clean HDR/calibrated recapture are outside this diagnostic protocol.

DNG_PROTOCOL.json is a new named protocol. The original capture_tm target, action catalog, timing budget, simulator and formal S24 test are preserved. Fixed analytic target for this diagnostic is sqrt(x/(x+0.5)) from already bounded/noisy green DNG source. It is a real-content relative surrogate, not new physical information or an expert-rendering target.
