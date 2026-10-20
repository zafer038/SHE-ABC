"""Budgeted evaluator with the CEC 2017 checkpoints (14 fractions of MaxFES)."""
from __future__ import annotations

from dataclasses import dataclass

from .cec2017 import official_checkpoints
from .evaluation import BudgetedEvaluator


@dataclass
class BudgetedEvaluator2017(BudgetedEvaluator):
    def __post_init__(self):
        self.checkpoints = official_checkpoints(self.dimension, self.max_fes)
        self._next_checkpoint = 0
