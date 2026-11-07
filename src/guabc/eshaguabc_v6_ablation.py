from dataclasses import dataclass

import numpy as np

from .eshaguabc import OPERATORS
from .eshaguabc_v6 import ESHAGUABC6


@dataclass
class ESHAGUABC6NoPool(ESHAGUABC6):
    name: str = 'eshaguabc_v6_no_pool'

    def _probabilities(self, share):
        p = np.zeros(len(OPERATORS))
        p[OPERATORS.index('guabc')] = 1.
        return p


@dataclass
class ESHAGUABC6NoEigen(ESHAGUABC6):
    name: str = 'eshaguabc_v6_no_eigen'
    p_eig: float = 0.


@dataclass
class ESHAGUABC6NoScout(ESHAGUABC6):
    name: str = 'eshaguabc_v6_no_scout'
    limit_factor: float = 1e12


@dataclass
class ESHAGUABC6NoCredit(ESHAGUABC6):
    name: str = 'eshaguabc_v6_no_credit'
    operator_learning_rate: float = 0.


@dataclass
class ESHAGUABC6NoRestart(ESHAGUABC6):
    name: str = 'eshaguabc_v6_no_restart'
    restart_diversity: float = 0.


ABLATIONS = (ESHAGUABC6NoPool, ESHAGUABC6NoEigen, ESHAGUABC6NoScout, ESHAGUABC6NoCredit, ESHAGUABC6NoRestart)
