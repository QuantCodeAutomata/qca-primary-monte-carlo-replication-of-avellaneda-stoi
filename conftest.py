"""
pytest configuration — adds project root to sys.path so that
`src.*` and `exp.*` imports work from any test file.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
