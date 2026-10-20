"""SHE-ABC v2: v1 with two changes.

A. Operator credit. An operator is rewarded by the rate at which its trials
   enter the p-best elite (value below the current p_count-th best fitness);
   v1 rewards the mean fitness gain per use.
B. Collapse restart. When the colony's mean distance to its centroid falls
   below ``restart_diversity`` x its initial value and the colony best has not
   improved at all (beyond floating-point noise, ``restart_rtol`` x |f|) for
   ``restart_window`` x max_fes evaluations, the
   whole colony is re-sampled uniformly (the incumbent is not re-inserted; the
   evaluator keeps the best-so-far) and memory, archive, trial counters and
   operator probabilities are reset.

Everything else is identical to v1.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

from .eshaguabc import ESHAGUABC, OPERATORS
from .optimizers import LOWER, UPPER, _initial_population


@dataclass
class ESHAGUABC2(ESHAGUABC):
    name: str = 'eshaguabc_v2'
    restart_diversity: float = 1e-3
    restart_window: float = .01
    restart_rtol: float = 1e-12  # improvements below this relative size are noise

    def run(self, evaluator, rng, population_size):
        if population_size < 5:
            raise ValueError('SHE-ABC requires at least five individuals')
        if len(OPERATORS)*self.p_min >= 1.:
            raise ValueError('p_min too large for the operator pool')
        d = evaluator.dimension
        pop, fit = _initial_population(evaluator, rng, population_size, d)
        n = len(pop)
        limit = max(1, int(self.limit_factor*n*d))
        p_count = min(n, max(2, int(round(n*self.p_best_rate))))
        archive_capacity = int(round(self.archive_rate*n))
        initial_diversity = float(np.mean(np.linalg.norm(pop-pop.mean(axis=0), axis=1)))

        def fresh_state():
            share = np.full(len(OPERATORS), 1./len(OPERATORS))
            return (np.zeros(n, dtype=int), np.full(self.memory_size, .5), np.full(self.memory_size, .5),
                    0, [], share, self._probabilities(share))

        trials, mf, mcr, memory_index, archive, share, probabilities = fresh_state()
        window_best, window_start = float(fit.min()), evaluator.evaluations
        usage = np.zeros(len(OPERATORS), dtype=int)
        wins = np.zeros(len(OPERATORS), dtype=int)
        elite_total = np.zeros(len(OPERATORS), dtype=int)
        scouts = updates = archive_draws = eigen_proposals = generations = eig_failures = 0
        restart_fes = []
        trace = deque(maxlen=256)
        while not evaluator.done:
            best = pop[int(np.argmin(fit))].copy()
            eigenvalues, basis, eig_ok = self._basis(pop, fit)
            eig_failures += int(not eig_ok)
            cumulative = np.cumsum(probabilities)
            elite_entries = np.zeros(len(OPERATORS))
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
                            rotated = np.where(mask, mutant@basis, parent@basis)
                            trial = rotated@basis.T
                        else:
                            trial = np.where(mask, mutant, parent)
                    elif op == 1:
                        use_eig = False
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
                            axis = basis[:, j]
                            trial = parent + float(step@axis)*axis
                        else:
                            trial = parent.copy()
                            trial[j] += step[j]
                    trial = self._repair(np.asarray(trial, dtype=float), parent)
                    elite_threshold = np.partition(fit, p_count-1)[p_count-1]
                    value = evaluator.evaluate(trial)
                    if not len(value):
                        break
                    gain = float(fit[i]-value[0])
                    usage[op] += 1
                    uses[op] += 1
                    eigen_proposals += int(use_eig)
                    entered = value[0] < elite_threshold
                    elite_entries[op] += entered
                    elite_total[op] += entered
                    trace.append({'evaluation':evaluator.evaluations-1, 'phase':phase, 'target':int(i),
                                  'operator':OPERATORS[op], 'eigen':bool(use_eig), 'f':f, 'cr':cr,
                                  'archive_size':len(archive), 'improvement':gain, 'elite_entry':bool(entered)})
                    if gain > 0:
                        wins[op] += 1
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
                            pop[i] = trial
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
            # (A) credit = elite-entry rate per use.
            rates = elite_entries/np.maximum(uses, 1.)
            if rates.sum() > 0:
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
                    if i != elite or value[0] < fit[i]:
                        pop[i], fit[i] = trial, value[0]
                        scouts += 1
            # (B) collapse restart.
            # Only a change beyond floating-point level (restart_rtol x |f|) counts as
            # improvement, so runs converging slowly near the optimum are not restarted.
            if fit.min() < window_best-self.restart_rtol*max(1., abs(window_best)):
                window_best, window_start = float(fit.min()), evaluator.evaluations
            diversity = float(np.mean(np.linalg.norm(pop-pop.mean(axis=0), axis=1)))
            if (not evaluator.done and diversity < self.restart_diversity*initial_diversity
                    and evaluator.evaluations-window_start >= self.restart_window*evaluator.max_fes):
                restart_fes.append(evaluator.evaluations)
                new_pop, new_fit = _initial_population(evaluator, rng, n, d)
                if len(new_fit) < n:
                    break  # budget or target reached during re-sampling
                pop, fit = new_pop, new_fit
                trials, mf, mcr, memory_index, archive, share, probabilities = fresh_state()
                window_best, window_start = float(fit.min()), evaluator.evaluations
        return {'best_position':evaluator.best_position, 'best_value':evaluator.best_value,
                'generations':generations, 'scout_events':scouts, 'scout_limit':limit,
                'restarts':len(restart_fes), 'restart_fes':restart_fes,
                'successful_memory_updates':updates, 'archive_draws':archive_draws,
                'eigen_proposals':eigen_proposals, 'eigen_failures':eig_failures,
                'operator_names':list(OPERATORS), 'operator_usage':usage.tolist(),
                'operator_successes':wins.tolist(), 'operator_elite_entries':elite_total.tolist(),
                'final_operator_probabilities':probabilities.tolist(),
                'final_memory_f':mf.tolist(), 'final_memory_cr':mcr.tolist(),
                'proposal_trace':list(trace)}
