from dataclasses import dataclass

import numpy as np

from .eshaguabc_v2 import ESHAGUABC2


@dataclass
class ESHAGUABC3(ESHAGUABC2):
    name: str = 'eshaguabc_v3'
    eig_min_per_dim: float = 2.

    def _basis(self, pop, fit):
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
