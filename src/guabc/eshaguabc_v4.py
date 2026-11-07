from dataclasses import dataclass

from .eshaguabc_v2 import ESHAGUABC2


@dataclass
class ESHAGUABC4(ESHAGUABC2):
    name: str = 'eshaguabc_v4'
    p_min: float = .15
    restart_window: float = .05
