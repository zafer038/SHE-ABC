"""Algorithm complexity of eshaguabc_v6 by the CEC procedure (T0, T1, T2), single process.

    python scripts/complexity.py            -> results/complexity_eshaguabc_v6.json

T0  : time of the CEC test loop (10^6 iterations of the prescribed arithmetic).
T1  : time of 200 000 evaluations of the test function alone (one point per call,
      the way the algorithm calls it).
T2  : time of eshaguabc_v6 using 200 000 evaluations on the same function (early
      stopping off), mean of 5 runs = T^2.
Reported: T0, T1, T^2 and (T^2 - T1)/T0.
CEC 2017 prescribes F18. For CEC 2022 both F1 and F7 are measured; use the one the
CEC 2022 technical report prescribes. Run on an otherwise idle machine.
"""
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
from guabc import cec2017, cec2022  # noqa: E402
from guabc.evaluation import BudgetedEvaluator  # noqa: E402
from guabc.evaluation_cec2017 import BudgetedEvaluator2017  # noqa: E402
from guabc.optimizers import OPTIMIZERS  # noqa: E402

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
                                  optimum=bias[function], target_error=-1.)  # no early stop
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
