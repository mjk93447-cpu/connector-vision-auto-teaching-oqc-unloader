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
