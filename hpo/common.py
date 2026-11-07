from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOWER, UPPER = -100.0, 100.0


@dataclass
class HPOEvaluator:
    objective: callable
    dimension: int
    max_fes: int
    evaluations: int = 0
    best_value: float = math.inf
    best_position: np.ndarray | None = None
    best_info: dict | None = None
    terminated_at: int | None = None
    log: list = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.max_fes - self.evaluations

    @property
    def done(self) -> bool:
        return self.remaining <= 0

    def evaluate(self, points) -> np.ndarray:
        x = np.atleast_2d(np.asarray(points, dtype=float))
        if self.done:
            return np.empty(0)
        x = np.clip(x[: self.remaining], LOWER, UPPER)
        values = []
        for point in x:
            started = time.perf_counter()
            value, info = self.objective(point)
            value = float(value) if np.isfinite(value) else 1.0
            self.evaluations += 1
            if value < self.best_value:
                self.best_value, self.best_position, self.best_info = value, point.copy(), info
            self.log.append({"evaluation": self.evaluations, "x": point.tolist(), "value": value,
                             "best_so_far": self.best_value, "seconds": time.perf_counter() - started,
                             **{k: v for k, v in info.items() if k != "per_class_iou"}})
            values.append(value)
        return np.array(values)


def unit(x):
    return (np.clip(np.asarray(x, dtype=float), LOWER, UPPER) - LOWER)/(UPPER - LOWER)


def decode_finetune(x) -> dict:
    u = unit(x)
    return {
        "lr": 10**(-5 + 3*u[0]),
        "weight_decay": 10**(-6 + 4*u[1]),
        "batch_size": (8, 16, 32)[min(2, int(3*u[2]))],
        "dice_weight": float(u[3]),
        "aug_prob": float(u[4]),
        "encoder_lr_scale": 10**(-2 + 2*u[5]),
    }


def decode_postproc(x) -> dict:
    u = unit(x)
    return {"class_bias": (-2 + 4*u[:6]).tolist(),
            "w_hflip": float(u[6]), "w_vflip": float(u[7])}


DIMENSIONS = {"finetune": 6, "postproc": 8}
DECODERS = {"finetune": decode_finetune, "postproc": decode_postproc}


def _size(expr, D, budget):
    return int(round(eval(str(expr), {"__builtins__": {}},
                          {"D": D, "budget": budget, "max": max, "min": min, "round": round})))


def run_algorithm(spec: dict, evaluator: HPOEvaluator, seed: int) -> dict:
    kind, D, budget = spec["kind"], evaluator.dimension, evaluator.max_fes
    rng = np.random.Generator(np.random.MT19937(seed))
    if kind == "random":
        while not evaluator.done:
            evaluator.evaluate(rng.uniform(LOWER, UPPER, size=D))
        return {}
    from guabc.optimizers import ArtificialBeeColony, DifferentialEvolution, LinearSuccessHistoryDE
    from guabc.eshaguabc_v6 import ESHAGUABC6
    params = dict(spec.get("params", {}))
    if kind == "abc":
        algo = ArtificialBeeColony(name=spec["name"], **params)
        pop = _size(spec["population"], D, budget)
    elif kind == "de":
        algo = DifferentialEvolution(name=spec["name"], **params)
        pop = _size(spec["population"], D, budget)
    elif kind == "lshade":
        algo = LinearSuccessHistoryDE(name=spec["name"], **params)
        pop = 0
    elif kind == "eshaguabc_v6":
        n_init = int(min(max(round(budget/10), 10), 5*D))
        n_min = max(5, n_init//2)
        algo = replace(ESHAGUABC6(), population_per_dim=n_init/D, **params)
        pop = n_min
    else:
        raise ValueError(f"unknown algorithm kind {kind}")
    diagnostics = algo.run(evaluator, rng, max(pop, 4 if kind != "eshaguabc_v6" else 5))
    return {k: v for k, v in diagnostics.items() if k not in ("best_position", "best_value", "proposal_trace")}
