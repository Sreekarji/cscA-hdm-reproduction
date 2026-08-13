"""
pytest configuration — adds all code subdirectories to sys.path so
tests can import project modules without installation.
"""
import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR  = os.path.dirname(_TESTS_DIR)

for _sub in ("", "hdm", "channel", "evaluation", "utils", "experiments", "lam"):
    _p = os.path.join(_CODE_DIR, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
