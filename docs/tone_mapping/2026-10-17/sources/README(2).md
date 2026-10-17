# Multi-Model Pseudo-GT Single-Scene Probe v1.1

This package contains the frozen engineering handoff for a single-scene Tone Mapping pseudo-GT feasibility probe.

- `CONTRACT.md`: authoritative scope, model/mode contract, hard gates, and decisions.
- `PLAN.md`: implementation phases, modules, CLI, tests, and outputs.
- `IMPLEMENTATION_CHECKLIST.md`: engineering and acceptance checklist.

Important boundary: this experiment does not train a Tone Mapping model and cannot establish generalization or production readiness. Boogu is included as a no-reference single-image editing extension; native same-scene visual-reference support must not be assumed.
