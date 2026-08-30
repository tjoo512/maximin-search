"""Vendored Qwen2.5-Math evaluation code (do not edit the modules in here).

Scraped from https://github.com/QwenLM/Qwen2.5-Math/blob/main/evaluation

`parser.py` does `from utils import *`, which resolves against the top-level
`utils` module (kept at src/utils.py). This package adds its own directory to
sys.path so the vendored modules import cleanly without being modified, then
re-exports the symbols our code uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from grader import math_equal
from parser import strip_string, extract_answer

__all__ = ["math_equal", "strip_string", "extract_answer"]
