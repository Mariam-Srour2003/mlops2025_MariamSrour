import sys
from pathlib import Path

# Ensure the project root (workspace) is on sys.path so tests can import top-level packages
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also add the `src` directory so packages under `src` (e.g., `ml_project`) are importable
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
