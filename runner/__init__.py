# src/tsl/runner/__init__.py
from .nf_compat import safe_nf_load, safe_nf_predict, prime_safe_globals

__all__ = [
    "safe_nf_load",
    "safe_nf_predict",
    "prime_safe_globals",
]
