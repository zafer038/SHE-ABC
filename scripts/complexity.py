from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from guabc import cec2017, cec2022
from guabc.evaluation import BudgetedEvaluator
from guabc.evaluation_cec2017 import BudgetedEvaluator2017
from guabc.optimizers import OPTIMIZERS

EVALS, REPEATS, ALGORITHM = 200_000, 5, "eshaguabc_v6"


def t0():
    started = time.perf_counter()
    for i in range(1, 1_000_001):
        x = 0.55 + i
        x = x + x; x = x/2; x = x*x; x = math.sqrt(x); x = math.log(x); x = math.exp(x); x = x/(x + 2)
    return time.perf_counter() - started


def measure(suite, function, dimension):
    if suite == "CEC2017":
        objective, evaluator_cls, bias = cec2017.CEC2017Function(function, dimension), BudgetedEvaluator2017, cec2017.BIAS
    else:
        objective, evaluator_cls, bias = cec2022.CEC2022Function(function, dimension), BudgetedEvaluator, cec2022.BIAS
    rng = np.random.default_rng(0)
    points = rng.uniform(-100, 100, (EVALS, dimension))
    started = time.perf_counter()
    for p in points:
        objective(p[None, :])
    t1 = time.perf_counter() - started
    t2 = []
    for r in range(REPEATS):
        evaluator = evaluator_cls(objective=objective, dimension=dimension, max_fes=EVALS,
                                  optimum=bias[function], target_error=-1.)
        started = time.perf_counter()
        OPTIMIZERS[ALGORITHM].run(evaluator, np.random.Generator(np.random.MT19937(r)), 100)
        t2.append(time.perf_counter() - started)
        assert evaluator.evaluations == EVALS
    t2_mean = float(np.mean(t2))
    return {"suite": suite, "function": function, "dimension": dimension, "T1": t1, "T2_runs": t2, "T2_mean": t2_mean}


def main():
    out = ROOT / "results" / f"complexity_{ALGORITHM}.json"
    rows = []
    base = t0()
    print(f"T0 = {base:.3f} s", flush=True)
    plan = [("CEC2017", 18, d) for d in (10, 30, 50, 100)] + \
           [("CEC2022", f, d) for f in (1, 7) for d in (10, 20)]
    for suite, function, dimension in plan:
        row = measure(suite, function, dimension)
        row["ratio"] = (row["T2_mean"] - row["T1"])/base
        rows.append(row)
        print(f"{suite} F{function} D{dimension}: T1 {row['T1']:.2f} s  T^2 {row['T2_mean']:.2f} s  "
              f"(T^2-T1)/T0 {row['ratio']:.1f}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"algorithm": ALGORITHM, "T0": base, "rows": rows, "python": sys.version,
                                   "numpy": np.__version__, "platform": platform.platform(),
                                   "processor": platform.processor()}, indent=1), encoding="utf-8")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
