# Self-Review (Pre-Commit)

Run this before committing to catch common mistakes.

## Steps

1. **Date check**: If any staged file contains a date (e.g. 2026-03-06), verify it matches `user_info` → `Today's date`. Fix if wrong.

2. **Link check**: If HANDOFF.md or similar references `docs/WEEKLY_HANDOFF_*.md`, confirm that file exists and the date in the filename is correct.

3. **EXE check**: If `run_pin_gui.py`, `pin_detection/gui.py`, or `tools_scripts/test_exe_stdout_fix.py` changed:
   - Run `python tools_scripts/test_exe_stdout_fix.py`
   - Optionally run `test_exe_train.bat` if EXE build is available

4. **Staging check**: Ensure no `__pycache__/`, `pin_models_*/`, or large artifacts are staged unless intended.

5. **Report**: List what was verified and any issues found. Fix issues before committing.
