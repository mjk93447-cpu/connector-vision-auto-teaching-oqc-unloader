# -*- mode: python ; coding: utf-8 -*-
# One-dir build — DLL in same folder as exe. Use when one-file has permission issues.
# Output: dist/pin_detection_gui/pin_detection_gui.exe + _internal/
from pathlib import Path
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = ['ultralytics', 'PIL._tkinter_finder', 'matplotlib']
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
    [],
    exclude_binaries=True,
    name='pin_detection_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    console=False,
    uac_admin=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pin_detection_gui',
)
