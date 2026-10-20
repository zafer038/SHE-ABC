"""SHE-ABC v1: artificial bee colony with an adaptive operator ensemble.

Employed, onlooker and scout phases as in ABC. Every bee draws one of three
proposal operators from an adaptive pool:

  0 'pbest'  current-to-pbest/1 with external archive (as in SHADE)
  1 'rand'   current-to-rand/1 (rotation-invariant, no crossover)
  2 'guabc'  ABC single-axis step with global-best guidance (the ABC move)

With probability ``p_eig`` a bee works in the eigenvector basis of the
covariance of the best ``eig_top_rate`` share of the colony: operator 0 does
its binomial crossover there, operator 2 steps along one eigen-axis. Scouts
sample an ellipsoid aligned with the same basis. F/CR use SHADE's
success-history memory. Proposals use L-SHADE midpoint repair; scouts clip.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .optimizers import LOWER, UPPER, _initial_population

OPERATORS = ('pbest', 'rand', 'guabc')


@dataclass
class ESHAGUABC:
    name: str = 'eshaguabc_v1'
    limit_factor: float = .05
    sigma_max: float = .10
    sigma_min: float = .001
    decay: float = 3.
    memory_size: int = 5
    p_best_rate: float = .11
    archive_rate: float = 1.4
    gbest_acceleration: float = 1.5
    p_eig: float = .5
    eig_top_rate: float = .30
    p_min: float = .05
    operator_learning_rate: float = .5

    def _basis(self, pop, fit):
        """Eigen-decomposition of the elite covariance (ascending eigenvalues)."""
        d = pop.shape[1]
        top = np.argsort(fit)[:max(2, int(round(self.eig_top_rate*len(pop))))]
        cov = np.atleast_2d(np.cov(pop[top].T)) + 1e-12*np.eye(d)
        try:
            values, basis = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            return np.ones(d), np.eye(d), False
        if not (np.isfinite(values).all() and np.isfinite(basis).all()):
            return np.ones(d), np.eye(d), False
        return np.maximum(values, 0.), basis, True

    @staticmethod
    def _repair(trial, parent):
        low, high = trial < LOWER, trial > UPPER
        trial[low] = (LOWER+parent[low])/2.
        trial[high] = (UPPER+parent[high])/2.
        return trial

    @staticmethod
    def _other(rng, n, *excluded):
        while True:
            k = int(rng.integers(n))
            if k not in excluded:
                return k

    def _sample_f(self, rng, mf, slot):
        f = mf[slot] + .1*rng.standard_cauchy()
        while f <= 0:
            f = mf[slot] + .1*rng.standard_cauchy()
        return min(1., float(f))

    def _probabilities(self, share):
        return self.p_min + (1.-len(OPERATORS)*self.p_min)*share

    def run(self, evaluator, rng, population_size):
        if population_size < 5:
            raise ValueError('SHE-ABC requires at least five individuals')
        if len(OPERATORS)*self.p_min >= 1.:
            raise ValueError('p_min too large for the operator pool')
        d = evaluator.dimension
        pop, fit = _initial_population(evaluator, rng, population_size, d)
        n = len(pop)
        trials = np.zeros(n, dtype=int)
        limit = max(1, int(self.limit_factor*n*d))
        p_count = min(n, max(2, int(round(n*self.p_best_rate))))
        archive_capacity = int(round(self.archive_rate*n))
        mf, mcr = np.full(self.memory_size, .5), np.full(self.memory_size, .5)
        memory_index = 0
        archive = []
        share = np.full(len(OPERATORS), 1./len(OPERATORS))
        probabilities = self._probabilities(share)
        usage = np.zeros(len(OPERATORS), dtype=int)
        wins = np.zeros(len(OPERATORS), dtype=int)
        scouts = updates = archive_draws = eigen_proposals = generations = eig_failures = 0
        trace = deque(maxlen=256)
        while not evaluator.done:
            best = pop[int(np.argmin(fit))].copy()
            eigenvalues, basis, eig_ok = self._basis(pop, fit)
            eig_failures += int(not eig_ok)
            cumulative = np.cumsum(probabilities)
            gains = np.zeros(len(OPERATORS))
            uses = np.zeros(len(OPERATORS))
            success_f, weight_f, success_cr, weight_cr = [], [], [], []
            for phase in ('employed', 'onlooker'):
                if evaluator.done:
                    break
                quality = np.max(fit)-fit+1e-12
                selected = (np.arange(n) if phase == 'employed' else
                            rng.choice(n, size=n, p=quality/quality.sum()))
                for i in selected:
                    if evaluator.done:
                        break
                    op = min(int(np.searchsorted(cumulative, rng.random()*cumulative[-1], side='right')),
                             len(OPERATORS)-1)
                    use_eig = eig_ok and rng.random() < self.p_eig
                    parent = pop[i]
                    f = cr = None
                    if op == 0:
                        slot = int(rng.integers(self.memory_size))
                        f = self._sample_f(rng, mf, slot)
                        cr = 0. if mcr[slot] < 0 else float(np.clip(rng.normal(mcr[slot], .1), 0., 1.))
                        guide = int(rng.choice(np.argsort(fit)[:p_count]))
                        r1 = self._other(rng, n, i)
                        r2 = self._other(rng, n+len(archive), i, r1)
                        donor = pop[r2] if r2 < n else archive[r2-n]
                        archive_draws += int(r2 >= n)
                        mutant = parent + f*(pop[guide]-parent) + f*(pop[r1]-donor)
                        mask = rng.random(d) < cr
                        mask[int(rng.integers(d))] = True
                        if use_eig:
                            # Crossover on eigen-coordinates, then rotate back.
                            rotated = np.where(mask, mutant@basis, parent@basis)
                            trial = rotated@basis.T
                        else:
                            trial = np.where(mask, mutant, parent)
                    elif op == 1:
                        use_eig = False  # current-to-rand/1 is rotation-invariant
                        slot = int(rng.integers(self.memory_size))
                        f = self._sample_f(rng, mf, slot)
                        r1 = self._other(rng, n, i)
                        r2 = self._other(rng, n, i, r1)
                        r3 = self._other(rng, n, i, r1, r2)
                        trial = parent + rng.random()*(pop[r1]-parent) + f*(pop[r2]-pop[r3])
                    else:
                        k = self._other(rng, n, i)
                        j = int(rng.integers(d))
                        phi = rng.uniform(-1., 1.)
                        psi = rng.uniform(0., self.gbest_acceleration)
                        step = phi*(parent-pop[k]) + psi*(best-parent)
                        if use_eig:
                            # Single-axis move along eigenvector j (x' = xB).
                            axis = basis[:, j]
                            trial = parent + float(step@axis)*axis
                        else:
                            trial = parent.copy()
                            trial[j] += step[j]
                    trial = self._repair(np.asarray(trial, dtype=float), parent)
                    value = evaluator.evaluate(trial)
                    if not len(value):
                        break
                    gain = float(fit[i]-value[0])
                    usage[op] += 1
                    uses[op] += 1
                    eigen_proposals += int(use_eig)
                    trace.append({'evaluation':evaluator.evaluations-1, 'phase':phase, 'target':int(i),
                                  'operator':OPERATORS[op], 'eigen':bool(use_eig), 'f':f, 'cr':cr,
                                  'archive_size':len(archive), 'improvement':gain})
                    if gain > 0:
                        wins[op] += 1
                        gains[op] += gain
                        archive.append(parent.copy())
                        if len(archive) > archive_capacity:
                            del archive[int(rng.integers(len(archive)))]
                        if f is not None:
                            success_f.append(f)
                            weight_f.append(gain)
                        if cr is not None:
                            success_cr.append(cr)
                            weight_cr.append(gain)
                        pop[i], fit[i], trials[i] = trial, value[0], 0
                    else:
                        if gain == 0:
                            pop[i] = trial  # neutral move across plateaus
                        trials[i] += 1
            updated = False
            if success_f:
                w, fs = np.asarray(weight_f)/np.sum(weight_f), np.asarray(success_f)
                mf[memory_index] = np.sum(w*fs**2)/np.sum(w*fs)
                updated = True
            if success_cr:
                w, crs = np.asarray(weight_cr)/np.sum(weight_cr), np.asarray(success_cr)
                denom = np.sum(w*crs)
                mcr[memory_index] = -1. if denom == 0 else np.sum(w*crs**2)/denom
                updated = True
            if updated:
                memory_index = (memory_index+1) % self.memory_size
                updates += 1
            rates = gains/np.maximum(uses, 1.)
            if rates.sum() > 0 and np.isfinite(rates).all():
                share = ((1.-self.operator_learning_rate)*share
                         + self.operator_learning_rate*rates/rates.sum())
                share /= share.sum()
                probabilities = self._probabilities(share)
            generations += 1
            exhausted = np.flatnonzero(trials >= limit)
            if len(exhausted) and not evaluator.done:
                progress = evaluator.evaluations/evaluator.max_fes
                sigma = self.sigma_min+(self.sigma_max-self.sigma_min)*np.exp(-self.decay*progress)
                eigenvalues, basis, eig_ok = self._basis(pop, fit)
                mean = eigenvalues.mean()
                # Ellipsoid shape blended 50/50 with isotropic noise; the mean
                # per-axis variance stays 1.
                scale = np.sqrt((eigenvalues/mean+1.)/2.) if eig_ok and mean > 0 else np.ones(d)
                for i in exhausted:
                    if evaluator.done:
                        break
                    elite = int(np.argmin(fit))
                    step = (rng.standard_normal(d)*scale)@basis.T
                    trial = np.clip(pop[elite]+sigma*step*(UPPER-LOWER), LOWER, UPPER)
                    value = evaluator.evaluate(trial)
                    if not len(value):
                        break
                    trials[i] = 0
                    # The colony's current best is only replaced by a better scout.
                    if i != elite or value[0] < fit[i]:
                        pop[i], fit[i] = trial, value[0]
                        scouts += 1
        return {'best_position':evaluator.best_position, 'best_value':evaluator.best_value,
                'generations':generations, 'scout_events':scouts, 'scout_limit':limit,
                'successful_memory_updates':updates, 'archive_draws':archive_draws,
                'eigen_proposals':eigen_proposals, 'eigen_failures':eig_failures,
                'operator_names':list(OPERATORS), 'operator_usage':usage.tolist(),
                'operator_successes':wins.tolist(), 'final_operator_probabilities':probabilities.tolist(),
                'final_memory_f':mf.tolist(), 'final_memory_cr':mcr.tolist(),
                'proposal_trace':list(trace)}
