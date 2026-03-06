---
name: connector-vision-auto-teaching
description: Domain knowledge for Connector Vision Auto Teaching on OQC Unloader. Use when working on connector pin/mold position detection, auto-teaching workflows, product model adaptation, or OQC unloader integration.
---

# Connector Vision Auto Teaching

## Project Goal

Automate pre-teaching of **connector pin and mold positions** per product model. Product models have different connector sizes/shapes; this program replaces manual teaching with automated detection for the OQC unloader machine.

## Key Concepts

- **Pin**: Connector pin location to detect.
- **Mold**: Mold boundary/position to detect.
- **Teaching**: Calibration step where the system learns correct positions for a given product model.
- **Auto Teaching**: Automate this calibration so new product models require minimal manual setup.

## Architecture (Fork from Edge)

This project forks the Edge (OLED FCB edge detection) system. Reuse:

- Vision pipeline: pre-processing, edge extraction, post-filters.
- Auto-optimization: parameter search, scoring, adaptive refinement.
- ROI-based evaluation and caching.

## Documentation

- **FORMAL_DOCUMENTATION.md**: Requirements, architecture, definitions. Update when changing scope or algorithm.
- **DEVELOPMENT_NOTES.md**: History, rationale, trial-and-error. Update when making design decisions.
- **docs/TROUBLESHOOTING.md**: EXE/Pin error cases, diagnosis, resolution. Use when EXE build/training fails.

## Conventions

- Follow FORMAL_DOCUMENTATION.md and DEVELOPMENT_NOTES.md style.
- English for formal docs; Korean allowed in README/operator guides.
- Python 3.8+, NumPy, Pillow. Optional CuPy for GPU.

## Date Verification

- **Always confirm current date** from user_info or system before writing dates in docs.
- Wrong dates (e.g. 2026-03-07 when actual is 2026-03-06) undermine trust in documentation.

## Reflection Workflow (Before Commit / Handoff)

1. **Pre-action**: Read `.cursor/rules/agent-preflight.mdc` — apply mandatory checks before writing dates, editing CI scripts, or creating handoff docs.
2. **Anti-patterns**: Read `.cursor/rules/agent-anti-patterns.mdc` — avoid date assumption, CI-only failure ignorance, documentation-only handoff, verification skip.
3. **Self-review**: Run `/self-review` (or follow `.cursor/commands/self-review.md`) before committing.
4. **Experience replay**: When a new failure occurs, add to `docs/EXPERIENCE_REPLAY.md` and update agent-preflight or agent-anti-patterns if no rule exists.
