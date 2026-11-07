from dataclasses import dataclass

from .eshaguabc_v4 import ESHAGUABC4


@dataclass
class ESHAGUABC5(ESHAGUABC4):
    name: str = 'eshaguabc_v5'
    population_per_dim: float = 5.

    def run(self, evaluator, rng, population_size):
        n = max(int(population_size), int(round(self.population_per_dim*evaluator.dimension)))
        result = super().run(evaluator, rng, n)
        result['population_size_used'] = n
        return result
