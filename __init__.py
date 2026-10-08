"""ComfyUI VELA H3 v1.0.0: production execution optimizations for VDN-H3 on MiniMax-H3.

Reference implementation: github.com/OpenVDN/vdn-minimax-h3 (Apache-2.0).
This package ports the released Video Delta Attention onto ComfyUI's native
MiniMax-H3 model as runtime model patches. VELA v1.0.0 does not modify
comfy/ldm/minimax/model.py or any other ComfyUI core source file.
"""

import os, sys
_PKG = os.path.dirname(__file__)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

from vdn_h3_24gb.vela_production_config import apply_vela_production_config
apply_vela_production_config()
from vdn_h3_24gb.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
