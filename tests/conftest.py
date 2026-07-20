"""pytest configuration: add src/ to sys.path so tests can import project modules."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
