from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

FUNCTIONS = tuple(range(1, 31))
DIMENSIONS = (10, 20, 30, 50, 100)
BIAS = {f: 100.0*f for f in FUNCTIONS}
CDIR = Path(__file__).resolve().parents[2] / "external" / "cec2014"
_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        os.chdir(CDIR)
        _LIB = ctypes.CDLL(str(CDIR / "cec14.dll"))
        _LIB.cec14_eval.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _LIB.cec14_eval.restype = None
    return _LIB


def max_fes(dimension: int) -> int:
    return 10_000*int(dimension)


class CEC2014Function:
    def __init__(self, function_number: int, dimension: int):
        if function_number not in BIAS:
            raise ValueError("function_number must be in 1..30")
        if dimension not in DIMENSIONS:
            raise ValueError("Supported dimensions: 10, 20, 30, 50, 100")
        self.function_number, self.dimension = int(function_number), int(dimension)
        self.bias = BIAS[self.function_number]

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(points, dtype=np.float64)))
        if x.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (n, {self.dimension}), received {x.shape}")
        f = np.empty(len(x), dtype=np.float64)
        _lib().cec14_eval(x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                          self.dimension, len(x), self.function_number)
        return f
