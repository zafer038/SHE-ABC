"""SHE-ABC v3 (a side branch of v2): v2 with one change.

The eigen basis is estimated from max(eig_top_rate x N, eig_min_per_dim x D)
elites instead of eig_top_rate x N. With N = 100 this is 30 elites at D = 10
(identical to v2) and 40 at D = 20.
"""
from dataclasses import dataclass

import numpy as np

from .eshaguabc_v2 import ESHAGUABC2


@dataclass
class ESHAGUABC3(ESHAGUABC2):
    name: str = 'eshaguabc_v3'
    eig_min_per_dim: float = 2.

    def _basis(self, pop, fit):
        """Eigen-decomposition of the elite covariance (ascending eigenvalues)."""
        d = pop.shape[1]
        count = min(len(pop), max(2, int(round(self.eig_top_rate*len(pop))),
                                  int(round(self.eig_min_per_dim*d))))
        top = np.argsort(fit)[:count]
        cov = np.atleast_2d(np.cov(pop[top].T)) + 1e-12*np.eye(d)
        try:
            values, basis = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return np.ones(d), np.eye(d), False
        if not (np.isfinite(values).all() and np.isfinite(basis).all()):
            return np.ones(d), np.eye(d), False
        return np.maximum(values, 0.), basis, True
