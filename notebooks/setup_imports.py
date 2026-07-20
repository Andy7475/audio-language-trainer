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
src_root = project_root / "src"

# Add project root and src/ to Python path so both
# `from src.phrases.x import ...` and bare `from phrases.x import ...` work.
for p in (src_root, project_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Verify imports work
