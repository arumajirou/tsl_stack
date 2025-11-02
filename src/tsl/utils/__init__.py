# -*- coding: utf-8 -*-
"""
tsl.utils subpackage
"""
from .nf_safe_load import (
    load_neuralforecast,
    register_nf_safe_globals,
)

__all__ = [
    "load_neuralforecast",
    "register_nf_safe_globals",
]
