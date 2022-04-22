# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports_pyqt = collect_submodules("PyQt5")
hiddenimports_image_codecs = collect_submodules('imagecodecs')
hiddenimports_sklearn = collect_submodules("sklearn")

block_cipher = None


a = Analysis(['infer_ctc_data.py'],
             pathex=['/srv/loeffler/Projects/pixel_embed_graph_match/pixel-embed-graph-match'],
             binaries=[],
             datas=[],
             hiddenimports=[*hiddenimports_sklearn, *hiddenimports_pyqt, *hiddenimports_image_codecs, "sklearn.neighbors._typedefs", "sklearn.utils._typedefs"],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,  
          [],
          name='infer_ctc_data',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )
