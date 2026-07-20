# Cross-Camera Samsung-Style Domain Adaptation v2 Design

## Goal

Given an iPhone 16-bit linear RGB image that has already undergone black-level correction and device-specific white balance, produce the Samsung rendering behavior encoded by the frozen Samsung-trained photofinishing model.

The learnable target is device-input compensation, not iPhone-native rendering style:

\[
X_i \xrightarrow{C_i} X_c \xrightarrow{A_t} X_s^{equiv}
\xrightarrow{F_s\;frozen} B_t \xrightarrow{R_t\;optional} \hat Y_t.
\]

`F_s` remains the style carrier. `A_t` removes residual input-domain differences. `R_t` is activated only when a stable, supported, luminance-only TM residual remains after Phase 1.

## Frozen Scope

- Source: Samsung paired linear input and GT, plus the existing Samsung-trained TM model.
- Calibration: 50 balanced same-scene but only approximately aligned iPhone/Samsung pairs; the Samsung side includes GT.
- Target: unpaired iPhone 16-bit linear RGB with incomplete HDR metadata.
- Supported residual failures: highlight-safe global midtone underexposure and face midtone underexposure.
- Trainable components: a low-capacity `TargetCameraAdapter`, then optionally a narrower `TMResidualAdapter`.
- Frozen components: Samsung TM backbone, Phase 1 modules while Phase 2 is trained, and all source-style definitions.
- Excluded: re-running black-level correction, full AWB estimation, full color-style transfer, 3A, multi-frame HDR, denoising, deblurring, super-resolution, end-to-end backbone retraining, online learning, adversarial alignment, and RL.

## Data Contracts

### Samsung source data

`SourceSample` contains Samsung linear RGB, Samsung GT, metadata, scene/group IDs, optional face/subject masks, and exact content hashes. It is used to qualify the frozen teacher, build source residual profiles, calibrate gates, and enforce source non-regression.

### Cross-device calibration data

`CalibrationPair` contains iPhone linear RGB, Samsung linear RGB, Samsung GT, both metadata records, alignment quality, shared scene/group ID, optional masks, and hashes. The 40-pair development portion uses group-aware five-fold validation. Ten pairs remain locked until model and loss choices are frozen.

### Target data

`TargetSample` contains iPhone linear RGB, metadata and confidence fields, target split, scene group, optional semantic masks, and hashes. Target train, validation, and locked holdout are disjoint by scene group.

All contracts reject unknown fields, invalid dimensions, non-finite values, unsupported encodings, impossible confidence values, and missing mandatory provenance. A critical unavailable signal never becomes `PASS`.

## Phase 1: Device Input Adaptation

### Device canonicalization

`C_i` applies only metadata-derivable operations, in this order:

1. convert the 16-bit representation to a normalized linear range using the declared white level;
2. undo/align known applied AWB channel gains when source and target gains are comparable;
3. transform to the declared common linear color space when a reliable CCM is available;
4. apply a bounded linear exposure-scale prior based on exposure metadata and same-scene robust statistics;
5. estimate reliable-dark and highlight-valid masks;
6. emit separate white-level, AWB, color, exposure, HDR and overall confidence fields.

The exposure product `time * ISO / aperture^2` is only a prior because phone HDR merge, digital gain and internal normalization can invalidate exact photometric proportionality. Low confidence keeps both canonicalization and learned residuals closer to identity.

### TargetCameraAdapter

The adapter order is fixed:

\[
X_1=D(g)X_c,\qquad
X_2=(I+\Delta M)X_1,\qquad
X_3=Curve_k(X_2).
\]

- residual per-channel gain `D(g)`;
- near-identity 3x3 matrix `I + delta_M`;
- six-point monotonic luminance curve applied by chroma-preserving RGB scaling;
- identity initialization, bounded parameters, and metadata-confidence gating;
- a tiny MLP maps fixed low-dimensional metadata/image statistics to parameters.

With only 50 calibration pairs, high-resolution gain maps, U-Nets and arbitrary local color transforms are forbidden.

### Samsung teacher branch

For a calibration pair:

\[
O_s=F_s(C_s(X_s^c)),\qquad
O_i=F_s(A_t(C_i(X_i^c))).
\]

The student matches `O_s`, because frozen Samsung model behavior is the desired style. Samsung GT qualifies the teacher and measures source residual; it does not replace the teacher output.

Teacher qualification thresholds come from independent Samsung paired validation distributions. A sample is qualified only when every critical metric is at or below its calibrated P75 and has no hard defect. P75-P90 samples are downweighted; any metric above P90 or any hard defect is rejected.

### Non-pixel-aligned distillation

\[
L_{pair}=\lambda_t L_{tone}+\lambda_r L_{ROI}+\lambda_l L_{lowfreq}+\lambda_a L_{adapter}.
\]

- `L_tone`: log-luminance quantiles, headroom, clipping and contrast; always available.
- `L_ROI`: face/subject/background statistics; enabled only for reliable corresponding ROIs.
- `L_lowfreq`: masked low-pass comparison; enabled only when low-frequency registration is valid.
- `L_adapter`: near-identity, bounded matrix/gain, curve and confidence regularization.
- no intermediate feature loss in the MVP, avoiding Samsung sensor-feature leakage.

Alignment quality selects the maximum legal loss: scene-only -> tone statistics; ROI -> ROI statistics; low-frequency -> low-frequency map; pixel-level loss remains disabled for approximate pairs.

### Two-stage parameter fitting

Each pair first obtains `theta_pair*`:

1. weighted least-squares initialization of channel gain and a near-identity matrix over reliable, registered, low-frequency regions;
2. isotonic initialization of the luminance curve;
3. joint refinement through frozen `F_s` using `L_pair`, while gradients update only pair parameters.

The frozen Samsung model must pass a gradient canary: strict checkpoint load, finite output, zero trainable backbone parameters, finite non-zero gradient at its input.

The per-pair parameters train a tiny `z -> theta` predictor. Scene-group five-fold development requires at least four folds with positive median improvement, at least 30/40 improved pairs, and a paired-bootstrap lower confidence bound above zero. The locked ten pairs must preserve the direction of improvement.

## Phase 2: Supported Residual Adaptation

### Activation

Phase 2 remains off unless Phase 1 locked holdout passes, adapter parameters are not saturating bounds, source replay has not regressed, target samples remain in source support, and a stable global/face underexposure residual remains after:

\[
B_t=F_s(A_t(C_i(X_t))).
\]

Stability requires at least 50 eligible target-train images, at least five scene groups, no group above 40%, at least 60% of samples above their conditional source P75, at least four of five groups with the same residual direction, and a bootstrap lower bound above zero after excluding OOD/metadata/boundary failures.

### Source residual anchor

On Samsung paired data:

\[
r_s=Y_s^*-F_s(C_s(X_s)).
\]

The source profile stores conditional global/face log-luminance residuals, dispersion, legal correction intervals, dynamic luminance intervals, highlight headroom and source-support statistics. It is versioned and bound to dataset, model, feature-schema and threshold hashes.

### Frozen feature vector

`psi_t` contains only interpretable scalar features:

- adapted output: global/dynamic-ROI P10/P25/P50/P75/P90 log luminance, MAD/local contrast, highlight headroom, clipping, shadow/midtone/highlight coverage, face quantiles, face/background ratio, ROI area/position;
- adapted linear representation: reliable-dark coverage, highlight-valid coverage, effective dynamic-range estimate, saturation, unreliable coverage in issue ROIs;
- metadata: completeness and exposure/white-level/WB/HDR confidence;
- Phase 1: parameter-bound margin, canonicalization confidence and calibration-support distance;
- categorical failure and scene conditions.

The estimator uses a calibrated low-capacity nearest-neighbor/binned profile. Missing support yields `UNAVAILABLE`, not extrapolation.

### Dynamic ROI

For face underexposure:

\[
ROI_{issue}=M_{face}^{soft}\cap M_{reliable}\cap\neg M_{highlight}\cap M_{luma-range}.
\]

For global underexposure:

\[
ROI_{issue}=M_{reliable}\cap\neg M_{highlight}\cap\{l_{low}\le L_{B_t}\le l_{high}\}.
\]

The luminance interval comes from the conditional source profile. A face box is Gaussian-feathered with sigma proportional to face size. Insufficient reliable area makes the gate unavailable.

### DirectionAlignment

The estimator returns global/face vectors:

\[
(\hat r_t,I_t,\sigma_t,q_t,[l_{low},l_{high}]).
\]

Candidate correction for region `j` is the median log-luminance change relative to baseline. Baseline correction is zero. Standardized distance is:

\[
D(r,\hat r)=\sum_j w_j\frac{|r_j-\hat r_j|}{\max(\sigma_j,\sigma_{min})}.
\]

PASS requires every component inside its interval, matching sign, improving over identity by calibrated `epsilon_dir`, a supported global/face correction ratio, and bounded non-target correction.

### Teachers

- L1 searches global monotonic curve parameters within the source-predicted interval. It emits parameter supervision.
- L2 adds genuinely distinct face/subject versus background correction using Gaussian-feathered ROIs and a smooth low-frequency gain field. It emits range or projected supervision.
- L3 is optional and local-only. Its raw generated image is only an appearance proposal; color, texture, geometry and generated detail are discarded by mandatory TM-space projection. A raw L3 image can never enter a pixel manifest.

VLMs may classify failure, provide semantic ROI and highlight-risk hints. They cannot determine numerical correction strength, approve candidates, compensate hard gates or select the pixel route.

### TMResidualAdapter

The adapter returns a complete corrected image:

\[
R_t(Y;\phi)=Y\frac{L'}{\max(L,\epsilon)},\qquad
L'=clip(C_6(L)G_{lowfreq},0,1).
\]

- six-point monotonic luminance curve with identity initialization;
- bounded 8x8 log-gain grid, bicubic upsampling and Gaussian smoothing;
- soft-ROI modulation;
- no 3x3 channel mixing, geometry change or high-frequency synthesis;
- 69 trainable scalar parameters in the default global instance.

Phase 2 freezes canonicalization, `A_t` and `F_s`. Source identity loss is `|R_t(O_s)-O_s|`. Source GT non-regression penalizes only increases beyond the calibrated tolerance.

## Certification and Routing

Critical non-compensable gates are:

1. Phase1Valid
2. SourceSupport
3. Eligibility
4. InputSupport
5. DirectionAlignment
6. IssueImprovement on the dynamic ROI
7. HighlightSafety using headroom/saturation metadata
8. Geometry
9. HighFrequency
10. ColorPreservation
11. NonTargetRegression
12. TMFeasibility
13. BoundaryArtifact

Every critical gate must be `PASS`. `UNAVAILABLE` never means `PASS`. Real-mode thresholds must come from a non-synthetic calibration profile; without one, real pixel routing is disabled. Overall IQA or VLM preference never decides acceptance.

Projection maps a proposal to the legal global monotonic curve plus smooth low-frequency gain space. Every projected target receives a fresh full certification. The raw-generated artifact cannot be used as a pixel target, and artifact hashes bind canonical tensor bytes rather than a container encoding.

Structured uncertainty remains separate: source support, residual interval width, diagnosis reliability, teacher agreement, projection retention, arbiter availability/order stability and metadata completeness. Routing uses a necessary-condition table:

- OOD/unreliable/unsupported -> diagnostic only;
- hard failure or wrong direction -> reject;
- stable L1 -> parameter;
- bounded but non-unique L2 -> range constraint;
- safe but perceptually ambiguous -> preference;
- source-grounded, fully projected/re-certified and calibrated -> pixel.

If parameter/range-only training matches or beats pixel training, the production MVP omits pixel supervision.

## Manifests and Lineage

Every output records input/model/config/profile hashes, parents, transformations, gate evidence, supervision type, synthetic/real status and route reasons. The manifest rejects unknown fields and prohibits raw generated pixels from the pixel route. Synthetic fixtures and deterministic model doubles are labeled explicitly.

## Mechanical Acceptance

- existing capture tests remain green;
- strict contracts and config reject malformed/unknown fields;
- Phase 1 identity, AWB alignment, ordering, monotonicity, loss gating and two-stage solver tests pass;
- real Samsung checkpoint gradient canary passes on CPU when the included files are present;
- Phase 2 support/ROI/direction/teacher/projection/certification/router tests pass;
- raw L3 direct injection, unavailable critical gates, boundary artifacts and uncalibrated real pixel routing fail closed;
- deterministic synthetic Phase 1 -> Phase 2 CLI run repeats exactly;
- reports clearly distinguish mechanical evidence from unavailable real-data effectiveness.

Real effectiveness remains unverified until the 50 calibration pairs and independent iPhone/Samsung holdouts are executed.
