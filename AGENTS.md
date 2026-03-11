# Agent Guide — Connector Vision Auto Teaching for OQC Unloader Machine

This project automates the pre-teaching process for connector pin and mold position detection. Product models have different connector sizes/shapes, so teaching was done manually each time; this program automates that for the OQC unloader machine.

---

## Cursor Setup

### MCP (Model Context Protocol)

Configured in `.cursor/mcp.json`:

| Server | Purpose |
|-------|---------|
| **github** | PRs, issues, repo management. Requires `GITHUB_PERSONAL_ACCESS_TOKEN` (see `.cursor/MCP_SETUP.md`). |
| **fetch** | Fetch web content (docs, APIs). |

See `.cursor/MCP_SETUP.md` for setup. **Required**: `.cursor/MCP_FINAL_STEPS.md` — token + restart.

### Skills (`.cursor/skills/`)

| Skill | When to use |
|-------|-------------|
| **connector-vision-auto-teaching** | Connector pin/mold detection, auto-teaching workflows, product model adaptation, OQC integration. |
| **edge-detection-pipeline** | Modifying `sobel_edge_detection.py`, tuning parameters, boundary/contour extraction. |

### Rules (`.cursor/rules/`)

| Rule | Scope |
|------|-------|
| **project-conventions** | Always apply. Core conventions. |
| **python-numpy** | When editing `**/*.py`. |
| **docs-style** | When editing `FORMAL_DOCUMENTATION.md`, `DEVELOPMENT_NOTES.md`, `docs/**/*.md`. |

---

## Documentation

- **FORMAL_DOCUMENTATION.md**: Requirements, architecture, definitions.
- **DEVELOPMENT_NOTES.md**: History, rationale, trial-and-error.
- **INSIGHTS.md**: 분석·평가·튜닝 결과 통합.
- Follow their style when updating.

---

## Cursor Cloud specific instructions

### Environment overview

This is a self-contained Python desktop application (no database, no web server, no Docker). Core stack: Python 3.12, PyTorch (CPU), Ultralytics YOLO, Tkinter GUI.

### Running tests

- Generate test data first (not committed): `python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_synthetic` and `python tools_scripts/generate_pin_test_data.py --output-dir test_data/pin_large_factory --large-factory --large-factory-n 5`
- Run pytest: `DISPLAY=:99 python -m pytest tests/ -v --tb=short -k "not test_train_validation and not test_edit_roi_validation"`
  - The two excluded tests (`test_train_validation`, `test_edit_roi_validation`) hang in headless environments because `messagebox.showerror()` opens a modal dialog that blocks. They pass on Windows with a real display.
- CI verification scripts (see `.github/workflows/build-pin-exe.yml` for the full list): `python tools_scripts/test_exe_stdout_fix.py`, `python tools_scripts/repro_exe_train_crash.py`, `python tools_scripts/verify_roi_path.py`, `python tools_scripts/test_gui_auto_fill.py`

### Running the application

- **GUI** (requires `DISPLAY`): start Xvfb if headless (`Xvfb :99 -screen 0 1024x768x24 &`), then `DISPLAY=:99 python run_pin_gui.py`
- **CLI train**: `python -m pin_detection.cli train --unmasked-dir <dir> --masked-dir <dir> --output-dir <dir> --epochs N`
- **CLI inference**: `python -m pin_detection.cli inference --model <.pt> --image <img> --conf 0.25`

### Gotchas

- `python3-tk` system package is required for Tkinter GUI and many tests. The update script installs it.
- No linter is configured in the project (`pyproject.toml` has no ruff/flake8/pylint settings).
- YOLO training on CPU is slow (~3-8s/epoch for small synthetic data). Use `--epochs 2` for quick validation.
