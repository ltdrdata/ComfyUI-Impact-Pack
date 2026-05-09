import sys
print(f"DEBUG: conftest.py executing, sys.path={sys.path[:3]}", file=sys.stderr)
"""
Pytest configuration for Impact Pack tests.
Mocks ComfyUI-specific modules for standalone testing.
"""
import sys
import tempfile
import os
from unittest.mock import MagicMock

# Create temp directory for mock file paths
_mock_temp_dir = tempfile.mkdtemp()

# Mock folder_paths with __file__ attribute
mock_folder_paths = MagicMock()
mock_folder_paths.__file__ = os.path.join(_mock_temp_dir, 'folder_paths.py')
mock_folder_paths.wildcards_dir = os.path.join(_mock_temp_dir, 'wildcards')
mock_folder_paths.models_dir = os.path.join(_mock_temp_dir, 'models')
os.makedirs(mock_folder_paths.wildcards_dir, exist_ok=True)
os.makedirs(mock_folder_paths.models_dir, exist_ok=True)
sys.modules['folder_paths'] = mock_folder_paths

# Mock nodes with __file__ attribute
mock_nodes = MagicMock()
mock_nodes.__file__ = os.path.join(_mock_temp_dir, 'nodes.py')
sys.modules['nodes'] = mock_nodes

# Mock impact package with __file__ attribute
mock_impact = MagicMock()
mock_impact.__file__ = os.path.join(_mock_temp_dir, 'impact')
# Set config and utils as attributes for 'from impact import config, utils'
sys.modules['impact'] = mock_impact

# Mock impact.config with __file__ and get_config()
mock_impact_config = MagicMock()
mock_impact_config.__file__ = os.path.join(_mock_temp_dir, 'impact', 'config.py')
mock_impact_config.get_config.return_value = {}
sys.modules['impact.config'] = mock_impact_config
# Also set as attribute on impact module
mock_impact.config = mock_impact_config

# Mock impact.utils with __file__ attribute
mock_impact_utils = MagicMock()
mock_impact_utils.__file__ = os.path.join(_mock_temp_dir, 'impact', 'utils.py')
sys.modules['impact.utils'] = mock_impact_utils
# Also set as attribute on impact module
mock_impact.utils = mock_impact_utils
mock_impact.utils = mock_impact_utils

# Mock comfy module with __file__ attribute (required by root __init__.py)
mock_comfy = MagicMock()
mock_comfy.__file__ = os.path.join(_mock_temp_dir, 'comfy', '__init__.py')
sys.modules['comfy'] = mock_comfy

# Mock comfy.samplers submodule
mock_comfy_samplers = MagicMock()
mock_comfy_samplers.__file__ = os.path.join(_mock_temp_dir, 'comfy', 'samplers.py')
sys.modules['comfy.samplers'] = mock_comfy_samplers
mock_comfy.samplers = mock_comfy_samplers

# Mock comfy.sd submodule
mock_comfy_sd = MagicMock()
mock_comfy_sd.__file__ = os.path.join(_mock_temp_dir, 'comfy', 'sd.py')
sys.modules['comfy.sd'] = mock_comfy_sd
mock_comfy.sd = mock_comfy_sd

# Mock other dependencies required by root __init__.py
mock_torch = MagicMock()
mock_torch.__file__ = os.path.join(_mock_temp_dir, 'torch', '__init__.py')
sys.modules['torch'] = mock_torch

mock_cv2 = MagicMock()
mock_cv2.__file__ = os.path.join(_mock_temp_dir, 'cv2', '__init__.py')
mock_cv2.setNumThreads = MagicMock()
sys.modules['cv2'] = mock_cv2

mock_numpy = MagicMock()
mock_numpy.__file__ = os.path.join(_mock_temp_dir, 'numpy', '__init__.py')
sys.modules['numpy'] = mock_numpy

mock_pil = MagicMock()
mock_pil.__file__ = os.path.join(_mock_temp_dir, 'PIL', '__init__.py')
mock_pil.Image = MagicMock()
mock_pil.ImageFilter = MagicMock()
sys.modules['PIL'] = mock_pil
sys.modules['PIL.Image'] = mock_pil.Image
sys.modules['PIL.ImageFilter'] = mock_pil.ImageFilter

mock_skimage = MagicMock()
mock_skimage.__file__ = os.path.join(_mock_temp_dir, 'skimage', '__init__.py')
mock_skimage.measure = MagicMock()
mock_skimage.measure.label = MagicMock()
mock_skimage.measure.regionprops = MagicMock()
sys.modules['skimage'] = mock_skimage
sys.modules['skimage.measure'] = mock_skimage.measure

mock_piexif = MagicMock()
mock_piexif.__file__ = os.path.join(_mock_temp_dir, 'piexif', '__init__.py')
sys.modules['piexif'] = mock_piexif
