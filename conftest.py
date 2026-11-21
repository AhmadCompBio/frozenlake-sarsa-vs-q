"""Pytest configuration: make the repo root importable so `from src import ...`
works when tests are run from any subdirectory.

pytest's prepend import mode normally adds the *test file's* root directory
(`tests/`) to sys.path — not the repo root — so `from src.fl_env import ...`
would otherwise fail. Inserting the repo root here fixes that without
requiring an editable install.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
