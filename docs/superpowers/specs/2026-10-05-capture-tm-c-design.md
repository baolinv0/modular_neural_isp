# C: capture exposure and tone mapping

User-authorized implementation, 2026-10-05. Base: main e0f82e9932b560a53a63d5eaf1a85ceefe4c91d3. Branch: feature/capture-tm-c-20261005.

Build a runnable research system around the existing frozen PhotofinishingModule. Separate actual capture time/analog/digital gain from render intent and virtual gain. Implement Apple US9432647B2 capture-bias compensation and Samsung US12243201B2 virtual-gain-informed tone construction as engineering interpretations, not complete patent reproductions. Borrow AdaptiveAE's image/history-histogram/stage observation; start with C's supervised offline oracle ranker, not a claim of reproducing the paper's A3C.

## Shared interfaces

Package `capture_tm`. PyTorch 2.5+, numpy, Pillow, YAML; pytest for verification. Original module dependencies remain untouched. All image tensors float32, channels-first; render output is display RGB [0,1]. No silent input-domain guessing. Working-directory-independent paths.

`types.py`:
- `CaptureAction(exposure_s: float, analog_gain: float = 1., digital_gain: float = 1.)`; positive finite values; `scale(reference: CaptureAction) -> float` is t*ga*gd ratio, only a brightness scale, not photon quality.
- `SensorProfile(full_well_e: float = 10000., read_noise_e: float = 3., adc_noise_dn: float = .5, bit_depth: int = 12, black_level_dn: float = 64., reference_exposure_s: float = 1/120, wb: tuple = (1.,1.,1.), ccm: tuple = identity)`; validated. `to_dict()/from_dict()`.
- `Scene(scene_id: str, split: str, frames: Tensor[T,3,H,W], frame_times_s: Tensor[T], subject_mask: Tensor[1,H,W] | None = None, source_kind: str = 'synthetic')`. Frames are nonnegative **sensor-linear RGB relative radiance**: at reference_exposure_s and unit gain, unclipped expected signal equals frames. HDR values >1 legal. Single frame means static; multiple frames strictly increasing times and shutter support must be sufficient. Masks and all variants share a source/scene split.
- `CaptureResult(rgb: Tensor[3,H,W], noise_variance: Tensor[3,H,W], saturation_mask: Tensor[3,H,W], action: CaptureAction, metadata: dict)`. Sensor-normalized RGB observation, not display RGB; WB/CCM handled by renderer.

`simulation.py`: `capture(scene, action, sensor, *, center_s=0., seed=0, noisy=True) -> CaptureResult`; integrate the same latent motion interval at a common exposure midpoint, sample photon noise in electrons, clip full-well, add pre-gain read noise, analog gain, post-gain ADC noise, black offset/quantization/ADC clip, then digital gain. Keep raw saturation provenance. Increasing analog/digital gain does not increase photons. `action_grid(exposures_s, analog_gains, *, digital_gain=1.) -> list[CaptureAction]`.

`data.py`: JSON manifest `{version:1, sensor:{...}, scenes:[{scene_id, split, frames_path, frame_times_s, source_kind, subject_mask_path?}]}` with relative paths resolved at manifest. `.npy/.npz` float frames only. `load_manifest(path) -> (SensorProfile,list[Scene])`; fail on unknown/missing domains, malformed shapes, overlap source identities. `write_demo_dataset(directory, *, seed=0, size=64, scenes=12) -> Path` produces deterministic static and moving HDR fixtures split by scene. At least train/val/test.

`sources.py`: explicit RawGen OETF-encoded XYZ import (inverse sRGB OETF first, camera matrix supplied, no camera-render previews as linear truth), linear RGB/XYZ import, RL-3A linear RGB PNG/DNG-derived relative proxy import, provenance and clipping limitations. DNG decoding optional dependency; never claim a clipped/noisy proxy is clean HDR. RL-3A noise-profile bridge labels fit domain and warns against calibrated-sensor claims.

`tone.py`: `PatentToneMapper(mode='apple', backend=None, *, curve_bins=256, max_render_ev=4.)` is nn.Module. `forward(x: Tensor[B,3,H,W], capture_bias_ev=0., render_intent_ev=0., *, virtual_gain=None, subject_mask=None) -> dict` with `output`, `tone_curve`, `virtual_gain`, `capture_compensation_ev`. Both modes operate before destructive clamp, preserve monotone finite HDR-domain mapping, apply virtual compensation once; capture bias cancellation must not cancel user intent. Target gain/curve policy based only on current observation/explicit reference scale, not privileged clean scene.

`modular.py`: `ModularPhotofinishingBackend(checkpoint: str|Path|None=None, *, device='cpu')`; None loads repository's shipped style-0 checkpoint, fails if absent. Frozen eval weights. Forward accepts bounded linear RGB, returns display RGB. Add a lightweight deterministic `AnalyticBackend` for tests/smoke, clearly distinct from trained modular renderer. No random-weight fallback.

`policy.py`: `ExposurePolicy(num_actions, *, width=24, histogram_bins=32, history_length=3)`; `forward(previews: Tensor[B,T,3,H,W], capture_state: Tensor[B,T,3], candidate_features: Tensor[B,K,3], feasible_mask: BoolTensor[B,K], *, stage=None) -> Tensor[B,K]`. Capture/candidate features are log2(t/reference_t), log2(analog_gain), log2(digital_gain); normalization fixed in one helper. Uses image features, temporal histograms, state/stage and shared candidate scorer. No clean/noise oracle/candidate images fed at deployment. Reject all-infeasible sets; invalid actions cannot win. `ranking_loss(scores, target_costs, feasible_mask) -> Tensor` differentiable teacher distribution + score fitting/ranking.

`control.py`: `ExposureConstraints(min_exposure_s=..., max_exposure_s=..., min_analog_gain=..., max_analog_gain=..., max_total_capture_s=..., readout_s=..., control_delay_frames=..., anti_flicker_hz=None)`; `.feasible_mask(actions)`. Explicit requested/effective frame ledger for simulated delayed control; no camera hardware claim. Same legal action family across all baselines. Return absolute physical action plus capture bias and render compensation.

Root owns `objective.py`, `experiment.py`, `cli.py`, configs, integration tests and documentation. Build offline teacher labels using noisy repeated physical capture -> fixed frontend -> frozen patent/modular TM, compare all candidates to one common clean/reference appearance at the same scene time and intent. Store costs, legal mask, preview observation, actions, renderer identity, source/group split, noise seeds. Train policy on train only, select val only, evaluate test once. Report baseline/oracle/policy regret and actual output quality, saturation, capture time; synthetic smoke is software evidence only. Inference emits next-frame requested action plus actual metadata requirements; no fabrication of real camera execution.

## Acceptance and boundaries

Meaningful tests: gain does not create photons; t changes noise and motion; full-well/ADC clipping retained; same-scene variants never cross splits; intent is not cancelled; monotone curves/no duplicate gain; frozen shipped modular checkpoint runs; infeasible actions masked; checkpoint/reload deterministic; end-to-end dataset -> oracle -> train -> test -> inference succeeds. Independent AE/TM/ISP reviewers must inspect implementation and fixes before delivery.

Real mobile calibration, true motion/RAW training data and a full GPU run remain separate from CPU synthetic verification. Existing nonaligned iPhone/S24 PGT reference is an appearance source only, not pixel-aligned radiance or AE GT. No claim that these mechanisms are novel or that the current policy improves real phone quality.

## Implementation clarifications after independent review

- Current delivery is C1 single future capture selection. Common-midpoint offline labels do not map controller frame ticks to hardware cadence; no closed-loop latency performance claim.
- Scene domain/source_id/provenance are mandatory in manifests. Actual decoded float32 frame/mask hashes, original input SHA, canonical path and original source group prevent duplicate-source split leakage. Timestamp arithmetic is float64.
- Black-subtracted negative read noise and floating digital HDR are preserved by simulation; fixed frontend clamps negative colors explicitly after WB/CCM. The default sensor is assumed_engineering; measured_sensor requires dictionary provenance.
- History shutter/readout windows must be sequential; candidate shutter begins after last observation shutter/readout. Online mask accepts remaining budget and scheduled future midpoint offset. Budget is remaining time for final capture, prior observations excluded.
- After observing k, submission tick k+1 becomes effective at k+1+delay. Requests carry sensor/renderer identities and never substitute for actual effective capture metadata.
- Samsung includes a fixed-anchor fallback for unresolved histogram atoms. Pre-backend black/monotone guarantees do not imply final learned-backend pixel guarantees; both branches share real frozen weights with a minimum input size 12x12.
- Full and image-history ablation share all 53,881 default parameters; external state/stage are zeroed only in ablation. Observation action menus vary actual metadata; stage remains fixed in C1.
- Checkpoint/cache pairing locks actual dataset and renderer digests plus teacher time/noise protocol. No resumable optimizer state is implemented; trained checkpoints reload for evaluation and inference.
