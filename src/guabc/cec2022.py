from __future__ import annotations

import importlib.util
import math
import os
import threading
from pathlib import Path

import numpy as np


BIAS = {
    1: 300.0,
    2: 400.0,
    3: 600.0,
    4: 800.0,
    5: 900.0,
    6: 1800.0,
    7: 2000.0,
    8: 2200.0,
    9: 2300.0,
    10: 2400.0,
    11: 2600.0,
    12: 2700.0,
}

OFFICIAL_MAX_FES = {10: 200_000, 20: 1_000_000}
_LOCK = threading.Lock()
_MODULE = None


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def official_python_dir() -> Path:
    return repository_root() / "external" / "cec2022-python" / "Python"


def _load_official_module():
    global _MODULE
    if _MODULE is None:
        module_path = official_python_dir() / "CEC2022.py"
        if not module_path.exists():
            raise FileNotFoundError(
                "Official CEC2022.py is missing. Extract Python-CEC2022.zip "
                "from external/cec2022-official first."
            )
        spec = importlib.util.spec_from_file_location("cec2022_official", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE = module
    return _MODULE


def official_checkpoints(dimension: int, max_fes: int) -> list[int]:
    """Return the 16 FE checkpoints defined in the CEC 2022 report."""
    points = [math.floor(dimension ** (k / 5.0 - 3.0) * max_fes) for k in range(16)]
    return [max(1, min(max_fes, value)) for value in points]


def official_run_seed(dimension: int, function_number: int, run: int) -> int:
    """Return the report-prescribed seed (``run`` is zero based)."""
    if dimension not in (10, 20):
        raise ValueError("Official seeds are defined here only for D=10 and D=20")
    runs = 30
    if not 0 <= run < runs:
        raise ValueError("run must be in 0..29")
    seeds_path = official_python_dir() / "input_data" / "Rand_Seeds.txt"
    seeds = np.loadtxt(seeds_path, dtype=float).astype(int)
    raw_index = int(dimension / 10 * function_number * runs + (run + 1)) - runs
    return int(seeds[raw_index % 1000])


class CEC2022Function:
    """Thin adapter around the unmodified official Python implementation."""

    def __init__(self, function_number: int, dimension: int):
        if function_number not in BIAS:
            raise ValueError("function_number must be in 1..12")
        if dimension not in (10, 20):
            raise ValueError("Official CEC 2022 experiments support only D=10 and D=20")
        self.function_number = int(function_number)
        self.dimension = int(dimension)
        self.bias = BIAS[self.function_number]
        self._module = _load_official_module()
        data_dir = official_python_dir() / "input_data"
        self._matrix = np.loadtxt(
            data_dir / f"M_{self.function_number}_D{self.dimension}.txt"
        )
        shift_raw = np.loadtxt(data_dir / f"shift_data_{self.function_number}.txt")
        if self.function_number < 9:
            self._shift = np.asarray(shift_raw[: self.dimension], dtype=float)
        else:
            self._shift = np.asarray(shift_raw[:9, : self.dimension], dtype=float).reshape(-1)
        self._shuffle = None
        if 6 <= self.function_number <= 8:
            self._shuffle = np.loadtxt(
                data_dir
                / f"shuffle_data_{self.function_number}_D{self.dimension}.txt"
            )

    def _evaluate_one(self, point: np.ndarray) -> float:
        m = self._module
        d = self.dimension
        # Several functions in the official translation use these work arrays
        # as module globals. Resetting them reproduces cec22_test_func without
        # re-reading the immutable data files for every FE.
        m.y = [0.0] * d
        m.z = [None] * d
        args = (point, d, self._shift, self._matrix)
        mapping = {
            1: lambda: m.zakharov_func(*args, 1, 1),
            2: lambda: m.rosenbrock_func(*args, 1, 1),
            3: lambda: m.schaffer_F7_func(*args, 1, 1),
            4: lambda: m.step_rastrigin_func(*args, 1, 1),
            5: lambda: m.levy_func(*args, 1, 1),
            6: lambda: m.hf02(point, d, self._shift, self._matrix, self._shuffle, 1, 1),
            7: lambda: m.hf10(point, d, self._shift, self._matrix, self._shuffle, 1, 1),
            8: lambda: m.hf06(point, d, self._shift, self._matrix, self._shuffle, 1, 1),
            9: lambda: m.cf01(*args, 1, 1),
            10: lambda: m.cf02(*args, 1, 1),
            11: lambda: m.cf06(*args, 1, 1),
            12: lambda: m.cf07(*args, 1, 1),
        }
        return float(mapping[self.function_number]()) + self.bias

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (n, {self.dimension}), received {x.shape}")

        # The official functions called by _evaluate_one use module globals, so
        # access is serialized. They read no files (only cec22_test_func does,
        # see legacy_call), so no chdir is needed: chdir on every evaluation made
        # parallel workers contend in the file system and ran ~20x slower.
        with _LOCK:
            raw = np.array([self._evaluate_one(point) for point in x], dtype=float)
        return np.asarray(raw, dtype=float)

    def legacy_call(self, points: np.ndarray) -> np.ndarray:
        """Slow official entry point, retained only for parity tests."""
        x = np.asarray(points, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        with _LOCK:
            previous = Path.cwd()
            try:
                os.chdir(official_python_dir())
                return np.array(
                    [
                        float(
                            self._module.cec22_test_func(
                                point, self.dimension, 1, self.function_number
                            )[0]
                        )
                        for point in x
                    ]
                )
            finally:
                os.chdir(previous)

    def error(self, points: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, self(points) - self.bias)
