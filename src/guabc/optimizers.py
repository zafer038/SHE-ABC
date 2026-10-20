from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .evaluation import BudgetedEvaluator


LOWER, UPPER = -100.0, 100.0


def _initial_population(evaluator, rng, population_size, dimension):
    pop = rng.uniform(LOWER, UPPER, size=(population_size, dimension))
    fit = evaluator.evaluate(pop)
    return pop[: len(fit)], fit


@dataclass
class ArtificialBeeColony:
    name: str = "abc"
    limit_factor: float = 1.0
    gbest_acceleration: float = 0.0
    gaussian_scout: bool = False
    sigma_max: float = 0.10
    sigma_min: float = 0.001
    decay: float = 3.0

    def run(self, evaluator: BudgetedEvaluator, rng, population_size: int):
        d = evaluator.dimension
        pop, fit = _initial_population(evaluator, rng, population_size, d)
        trials = np.zeros(len(pop), dtype=int)
        limit = max(1, int(self.limit_factor * len(pop) * d))
        scout_events = 0

        while not evaluator.done:
            best = pop[int(np.argmin(fit))].copy()
            for phase in ("employed", "onlooker"):
                if evaluator.done:
                    break
                if phase == "employed":
                    selected = np.arange(len(pop))
                else:
                    quality = np.max(fit) - fit + 1e-12
                    selected = rng.choice(len(pop), size=len(pop), p=quality / quality.sum())
                for i in selected:
                    if evaluator.done:
                        break
                    candidates = np.delete(np.arange(len(pop)), i)
                    k = int(rng.choice(candidates))
                    j = int(rng.integers(d))
                    trial = pop[i].copy()
                    phi = rng.uniform(-1.0, 1.0)
                    trial[j] += phi * (pop[i, j] - pop[k, j])
                    if self.gbest_acceleration > 0:
                        psi = rng.uniform(0.0, self.gbest_acceleration)
                        trial[j] += psi * (best[j] - pop[i, j])
                    trial = np.clip(trial, LOWER, UPPER)
                    value = evaluator.evaluate(trial)
                    if len(value) == 0:
                        break
                    if value[0] < fit[i]:
                        pop[i], fit[i], trials[i] = trial, value[0], 0
                    else:
                        trials[i] += 1

            progress = evaluator.evaluations / evaluator.max_fes
            for i in np.flatnonzero(trials >= limit):
                if evaluator.done:
                    break
                if self.gaussian_scout:
                    sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * np.exp(
                        -self.decay * progress
                    )
                    trial = best + rng.normal(0.0, sigma, size=d) * (UPPER - LOWER)
                else:
                    trial = rng.uniform(LOWER, UPPER, size=d)
                trial = np.clip(trial, LOWER, UPPER)
                value = evaluator.evaluate(trial)
                if len(value):
                    pop[i], fit[i], trials[i] = trial, value[0], 0
                    scout_events += 1
        return {
            "best_position": evaluator.best_position,
            "best_value": evaluator.best_value,
            "scout_events": scout_events,
            "scout_limit": limit,
        }


@dataclass
class DifferentialEvolution:
    name: str = "de"
    mutation: float = 0.5
    crossover: float = 0.9

    def run(self, evaluator, rng, population_size):
        d = evaluator.dimension
        n = max(4, population_size)
        pop, fit = _initial_population(evaluator, rng, n, d)
        while not evaluator.done:
            for i in range(n):
                if evaluator.done:
                    break
                pool = np.delete(np.arange(n), i)
                a, b, c = rng.choice(pool, size=3, replace=False)
                mutant = pop[a] + self.mutation * (pop[b] - pop[c])
                mask = rng.random(d) < self.crossover
                mask[int(rng.integers(d))] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, LOWER, UPPER)
                value = evaluator.evaluate(trial)
                if len(value) and value[0] <= fit[i]:
                    pop[i], fit[i] = trial, value[0]
        return {
            "best_position": evaluator.best_position,
            "best_value": evaluator.best_value,
        }


@dataclass
class LinearSuccessHistoryDE:
    """L-SHADE 1.0.1 translated from Tanabe and Fukunaga's fixed source."""

    name: str = "lshade"
    initial_population_multiplier: float = 18.0
    minimum_population_size: int = 4
    memory_size: int = 5
    archive_rate: float = 1.4
    p_best_rate: float = 0.11

    @staticmethod
    def _cauchy_positive(rng, location):
        value = location + 0.1 * rng.standard_cauchy()
        while value <= 0:
            value = location + 0.1 * rng.standard_cauchy()
        return min(1.0, float(value))

    @staticmethod
    def _midpoint_bounds(vector, parent):
        repaired = vector.copy()
        below = repaired < LOWER
        above = repaired > UPPER
        repaired[below] = (LOWER + parent[below]) / 2.0
        repaired[above] = (UPPER + parent[above]) / 2.0
        return repaired

    def run(self, evaluator, rng, population_size):
        d = evaluator.dimension
        initial_size = max(
            self.minimum_population_size,
            int(round(self.initial_population_multiplier * d)),
        )
        initial_size = min(initial_size, evaluator.max_fes)
        pop, fit = _initial_population(evaluator, rng, initial_size, d)
        initial_size = len(pop)
        archive = []
        memory_f = np.full(self.memory_size, 0.5, dtype=float)
        memory_cr = np.full(self.memory_size, 0.5, dtype=float)
        memory_position = 0
        generations = 0
        successful_updates = 0
        maximum_archive_size = 0

        while not evaluator.done and len(pop) >= self.minimum_population_size:
            n = len(pop)
            order = np.argsort(fit)
            p_count = max(2, int(round(n * self.p_best_rate)))
            p_count = min(p_count, n)
            children = np.empty_like(pop)
            sampled_f = np.empty(n, dtype=float)
            sampled_cr = np.empty(n, dtype=float)

            for target in range(n):
                memory_index = int(rng.integers(self.memory_size))
                mu_f, mu_cr = memory_f[memory_index], memory_cr[memory_index]
                sampled_f[target] = self._cauchy_positive(rng, mu_f)
                sampled_cr[target] = (
                    0.0
                    if mu_cr < 0
                    else float(np.clip(rng.normal(mu_cr, 0.1), 0.0, 1.0))
                )
                p_best = int(rng.choice(order[:p_count]))

                r1_candidates = np.delete(np.arange(n), target)
                r1 = int(rng.choice(r1_candidates))
                union_size = n + len(archive)
                while True:
                    r2_union = int(rng.integers(union_size))
                    if r2_union != target and r2_union != r1:
                        break
                r2_vector = pop[r2_union] if r2_union < n else archive[r2_union - n]
                mutant = (
                    pop[target]
                    + sampled_f[target] * (pop[p_best] - pop[target])
                    + sampled_f[target] * (pop[r1] - r2_vector)
                )
                mutant = self._midpoint_bounds(mutant, pop[target])
                mask = rng.random(d) < sampled_cr[target]
                mask[int(rng.integers(d))] = True
                children[target] = np.where(mask, mutant, pop[target])

            child_fit = evaluator.evaluate(children)
            active = len(child_fit)
            if active == 0:
                break
            success_f, success_cr, improvements = [], [], []
            for i in range(active):
                if child_fit[i] <= fit[i]:
                    if child_fit[i] < fit[i]:
                        archive.append(pop[i].copy())
                        success_f.append(sampled_f[i])
                        success_cr.append(sampled_cr[i])
                        improvements.append(abs(fit[i] - child_fit[i]))
                    pop[i], fit[i] = children[i], child_fit[i]

            archive_capacity = int(round(self.archive_rate * len(pop)))
            while len(archive) > archive_capacity:
                del archive[int(rng.integers(len(archive)))]
            maximum_archive_size = max(maximum_archive_size, len(archive))

            if success_f:
                weights = np.asarray(improvements, dtype=float)
                weights /= weights.sum()
                success_f = np.asarray(success_f, dtype=float)
                success_cr = np.asarray(success_cr, dtype=float)
                memory_f[memory_position] = np.sum(weights * success_f**2) / np.sum(
                    weights * success_f
                )
                cr_denominator = np.sum(weights * success_cr)
                memory_cr[memory_position] = (
                    -1.0
                    if cr_denominator == 0
                    else np.sum(weights * success_cr**2) / cr_denominator
                )
                memory_position = (memory_position + 1) % self.memory_size
                successful_updates += 1

            planned_size = int(
                round(
                    (self.minimum_population_size - initial_size)
                    * evaluator.evaluations
                    / evaluator.max_fes
                    + initial_size
                )
            )
            planned_size = max(self.minimum_population_size, planned_size)
            if len(pop) > planned_size:
                keep = np.argsort(fit)[:planned_size]
                pop, fit = pop[keep], fit[keep]
                archive_capacity = int(round(self.archive_rate * len(pop)))
                while len(archive) > archive_capacity:
                    del archive[int(rng.integers(len(archive)))]
            generations += 1
            if active < n:
                break

        return {
            "best_position": evaluator.best_position,
            "best_value": evaluator.best_value,
            "generations": generations,
            "initial_population_size": initial_size,
            "final_population_size": len(pop),
            "successful_memory_updates": successful_updates,
            "maximum_archive_size": maximum_archive_size,
        }


OPTIMIZERS = {
    "abc": ArtificialBeeColony(name="abc"),
    "gabc": ArtificialBeeColony(name="gabc", gbest_acceleration=1.5),
    "de": DifferentialEvolution(),
    "lshade": LinearSuccessHistoryDE(),
}

# SHE-ABC (eshaguabc_v6) and its development versions v1-v5 import LOWER/UPPER/_initial_population
# from this module, so they are registered after they are defined.
from .eshaguabc import ESHAGUABC  # noqa: E402

OPTIMIZERS["eshaguabc_v1"] = ESHAGUABC()

from .eshaguabc_v2 import ESHAGUABC2  # noqa: E402

OPTIMIZERS["eshaguabc_v2"] = ESHAGUABC2()

from .eshaguabc_v3 import ESHAGUABC3  # noqa: E402

OPTIMIZERS["eshaguabc_v3"] = ESHAGUABC3()

from .eshaguabc_v4 import ESHAGUABC4  # noqa: E402

OPTIMIZERS["eshaguabc_v4"] = ESHAGUABC4()

from .eshaguabc_v5 import ESHAGUABC5  # noqa: E402

OPTIMIZERS["eshaguabc_v5"] = ESHAGUABC5()

from .eshaguabc_v6 import ESHAGUABC6  # noqa: E402

OPTIMIZERS["eshaguabc_v6"] = ESHAGUABC6()

from .eshaguabc_v6_ablation import ABLATIONS  # noqa: E402

for _cls in ABLATIONS:
    OPTIMIZERS[_cls.name] = _cls()
