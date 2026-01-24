"""Setup script to add src directory to Python path for Jupyter notebooks.

Import this at the top of your notebooks with:
    from setup_imports import *

Or add the path manually with:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd().parent))
"""

import sys
from pathlib import Path

# Get the project root directory (parent of notebooks/)
project_root = Path(__file__).parent.parent

# Add project root to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Verify imports work
