"""Vendored Skywork-o1 PRM inference code (do not edit the modules in here).

Scraped from https://github.com/SkyworkAI/skywork-o1-prm-inference

The scraped modules use flat absolute imports (e.g. `from modeling_base import
PreTrainedModelWrapper`). Rather than rewrite the vendored code to use relative
imports, this package puts its own directory on sys.path so those flat imports
keep resolving, then re-exports the symbols our code actually uses.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from prm_model import PRM_MODEL
from io_utils import (
    prepare_input,
    prepare_batch_input_for_model,
    derive_step_rewards,
)

__all__ = [
    "PRM_MODEL",
    "prepare_input",
    "prepare_batch_input_for_model",
    "derive_step_rewards",
]
