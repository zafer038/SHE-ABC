"""Leave-one-out variants of SHE-ABC (v6).

Each variant is v6 with exactly one component switched off; everything else
(code path, parameters, seeds) is unchanged. The two population components are
covered by the version comparison: v5 = v6 without the population reduction,
v4 = v5 without the dimension-scaled colony.

- no_pool    : operator pool removed, only the guabc (ABC) operator is used
- no_eigen   : eigen crossover off (p_eig = 0); the covariance scout still uses the basis
- no_scout   : scout phase off (limit never reached)
- no_credit  : adaptive operator credit off (operator probabilities stay uniform)
- no_restart : collapse restart off
"""
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
