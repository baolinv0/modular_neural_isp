# TM PGT IQA V2 Design

## Goal

Turn one front-camera source image plus semantic masks into a deterministic,
auditable 16-candidate TM pool and select a high-confidence pseudo-GT. The
system evaluates only exposure, tone, dynamic-range allocation, local face
exposure, and face/background tone relationship. It rejects clear non-TM
edits, but does not score texture, sharpness, noise, bokeh, or beautification.

## Pipeline

`source + masks -> candidate pool -> objective qualification -> family-balanced
Top-K -> Qwen semantic review + pairwise tournament -> PGT class, confidence,
reports`.

Candidate generation is deterministic except for the optional image-edit
adapter. Retinex is the ordered primary search axis (eight EV variants), local
face TM supplies three masked lifts, tone shape supplies two exposure-preserving
curves, simple gain supplies one baseline, and the optional edit adapter can
provide `qwen_normal` and `qwen_strong`. Missing edit-service configuration must
not prevent the deterministic fourteen candidates from being generated.

## Core contracts

- Candidate manifests are authoritative; candidate family and parameters are
  never inferred from a filename.
- Label maps use 0=background, 1=face/non-skin, 2=skin, 3=human/body. Soft
  masks, if supplied, are preferred and derived into face core/rings/background.
- Objective scoring remains continuous and config-driven. Hard guards apply only
  severe clip/underexposure, clear source-preservation failure, or semantic
  TM-only failure; halo is a warning.
- Pool outlier is family-balanced and affects only `pool_confidence`; it cannot
  by itself reject or downgrade image quality.
- Qwen receives source, candidate, overlay, and numerical evidence. It creates
  per-source scene intent once, candidate naturalness/TM-only judgements, and
  pairwise preferences over a deterministic family-balanced Top-K.
- `pgt_class` and `selection_confidence` are independent. Certified labels have
  weight 1.0 even when the winner is only weakly separated.

## Outputs and non-goals

The batch CLI emits per-scene JSON, aggregate report JSON/CSV, selected image,
candidate manifests, semantic JSON, and candidate/ranking/failure grids. It
also exposes objective-only execution. It does not claim live Qwen conformance
or human-ranking validation: both remain explicitly `NOT COMPLETE` until run on
the deployment server and annotated scenes.

