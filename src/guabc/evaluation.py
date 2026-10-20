from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cec2022 import official_checkpoints


@dataclass
class BudgetedEvaluator:
    objective: callable
    dimension: int
    max_fes: int
    optimum: float
    target_error: float = 1e-8
    evaluations: int = 0
    best_value: float = np.inf
    best_position: np.ndarray | None = None
    terminated_at: int | None = None
    trace: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.checkpoints = official_checkpoints(self.dimension, self.max_fes)
        self._next_checkpoint = 0

    @property
    def remaining(self) -> int:
        return self.max_fes - self.evaluations

    @property
    def done(self) -> bool:
        return self.remaining <= 0 or self.terminated_at is not None

    @property
    def best_error(self) -> float:
        return max(0.0, float(self.best_value - self.optimum))

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if self.done:
            return np.empty(0, dtype=float)
        x = x[: self.remaining]
        values = np.asarray(self.objective(x), dtype=float)
        if len(values) != len(x):
            raise RuntimeError("Objective returned a different number of values")

        processed = 0
        for point, value in zip(x, values):
            self.evaluations += 1
            processed += 1
            if value < self.best_value:
                self.best_value = float(value)
                self.best_position = point.copy()
            while (
                self._next_checkpoint < len(self.checkpoints)
                and self.evaluations >= self.checkpoints[self._next_checkpoint]
            ):
                self.trace.append(
                    {
                        "fes": self.checkpoints[self._next_checkpoint],
                        "best_error": max(self.best_error, self.target_error),
                    }
                )
                self._next_checkpoint += 1
            if self.best_error < self.target_error and self.terminated_at is None:
                self.terminated_at = self.evaluations
                break
        return values[:processed]

    def finalize_trace(self) -> list[dict]:
        terminal_error = max(self.best_error, self.target_error)
        while self._next_checkpoint < len(self.checkpoints):
            self.trace.append(
                {"fes": self.checkpoints[self._next_checkpoint], "best_error": terminal_error}
            )
            self._next_checkpoint += 1
        return self.trace
