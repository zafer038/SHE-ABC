"""SHE-ABC v4: v2 with two parameter changes.

- restart_window 0.01 -> 0.05: a longer stagnation window before a collapse
  restart.
- p_min 0.05 -> 0.15: a higher floor of the operator probabilities (no operator
  exceeds 0.70).
"""
from dataclasses import dataclass

from .eshaguabc_v2 import ESHAGUABC2


@dataclass
class ESHAGUABC4(ESHAGUABC2):
    name: str = 'eshaguabc_v4'
    p_min: float = .15
    restart_window: float = .05
