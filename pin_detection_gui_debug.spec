# Debug build: console=True to see crash traceback in a window.
# Use for Action #25 crash diagnosis. Output: dist/pin_detection_gui_debug.exe
# Run: pin_detection_gui_debug.exe — console window shows stderr.
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['ultralytics', 'PIL._tkinter_finder', 'matplotlib', 'pin_detection.debug_log']
if Path('models/yolo26n.pt').exists():
    datas.append(('models/yolo26n.pt', 'models'))
tmp_ret = collect_all('ultralytics')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('numpy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('PIL')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

a = Analysis(
    ['run_pin_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow', 'tensorflow_core', 'tensorflow_estimator',
        'pytest', 'IPython', 'jupyter', 'notebook', 'sphinx',
        'setuptools',
        'triton', 'onnxruntime', 'onnx',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pin_detection_gui_debug',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,   # DEBUG: show console to capture crash traceback
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=False,
)
