"""Validate the CEC 2017 port.

1. Optimum: F(o) - F* for every function / dimension.
2. Basic functions against the organisers' CEC 2022 Python code (same C lineage),
   fed with CEC 2017 shift/rotation data.
3. Hybrid / composition functions against an independent loop-by-loop
   transcription of the C code built on those official basic functions
   (weierstrass and bi_rastrigin, absent from CEC 2022, transcribed literally).
"""
import importlib.util, math, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "external" / "cec2017" / "input_data"
CEC22 = ROOT / "external" / "cec2022-python" / "Python" / "CEC2022.py"
sys.path.insert(0, str(ROOT / "src"))
from guabc.cec2017 import CEC2017Function, FUNCTIONS, BIAS  # noqa: E402

spec = importlib.util.spec_from_file_location("cec22", CEC22)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
PI, INF = m.PI, m.INF

# ---------- literal transcriptions of the two functions CEC 2022 lacks ----------
def weierstrass_ref(x, nx, Os, Mr, s, r):
    z = m.sr_func(x, nx, Os, Mr, 0.5/100.0, s, r)
    f = 0.0
    for i in range(nx):
        sm = 0.0; sm2 = 0.0
        for j in range(21):
            sm += math.pow(0.5, j)*math.cos(2.0*PI*math.pow(3.0, j)*(z[i] + 0.5))
            sm2 += math.pow(0.5, j)*math.cos(2.0*PI*math.pow(3.0, j)*0.5)
        f += sm
    return f - nx*sm2

def bi_rastrigin_ref(x, nx, Os, Mr, s, r):
    mu0, d = 2.5, 1.0
    sc = 1.0 - 1.0/(2.0*math.pow(nx + 20.0, 0.5) - 8.2)
    mu1 = -math.pow((mu0*mu0 - d)/sc, 0.5)
    xx = list(x)
    if s == 1:
        for i in range(nx): m.y[i] = xx[i] - Os[i]
    else:
        for i in range(nx): m.y[i] = xx[i]
    for i in range(nx): m.y[i] *= 10.0/100.0
    tmpx = [2*m.y[i]*(-1.0 if Os[i] < 0.0 else 1.0) for i in range(nx)]
    zz = list(tmpx)
    tmpx = [t + mu0 for t in tmpx]
    tmp1 = sum((t - mu0)**2 for t in tmpx)
    tmp2 = sum((t - mu1)**2 for t in tmpx)*sc + d*nx
    if r == 1:
        rot = m.rotatefunc(zz, nx, Mr)
        for i in range(nx): m.y[i] = rot[i]
        tmp = sum(math.cos(2.0*PI*m.y[i]) for i in range(nx))
    else:
        tmp = sum(math.cos(2.0*PI*v) for v in zz)
    return min(tmp1, tmp2) + 10.0*(nx - tmp)

REF = {name: getattr(m, name + "_func") for name in
       ("ellips", "bent_cigar", "discus", "rosenbrock", "ackley", "griewank", "rastrigin", "schwefel",
        "grie_rosen", "escaffer6", "happycat", "hgbat", "schaffer_F7", "step_rastrigin", "levy",
        "zakharov", "katsuura")}
REF["weierstrass"] = weierstrass_ref
REF["bi_rastrigin"] = bi_rastrigin_ref


def levy_ref(x, nx, Os, Mr, s, r):
    # CEC 2022 changed w = 1 + (z-1)/4 to 1 + z/4; CEC 2017's C code keeps (z-1).
    z = m.sr_func(x, nx, Os, Mr, 1.0, s, r)
    w = [1.0 + (z[i] - 1.0)/4.0 for i in range(nx)]
    t1 = math.pow(math.sin(PI*w[0]), 2)
    t3 = math.pow(w[nx - 1] - 1, 2)*(1 + math.pow(math.sin(2*PI*w[nx - 1]), 2))
    return t1 + sum(math.pow(w[i] - 1, 2)*(1 + 10*math.pow(math.sin(PI*w[i] + 1), 2)) for i in range(nx - 1)) + t3


REF["levy"] = levy_ref


def optimum(fn):
    """F9's minimum is where z = 1 (C code's w uses z-1), elsewhere at the shift."""
    d = fn.dimension
    if fn.function_number == 9:
        return fn.Os[:d] + np.linalg.solve(fn.M[:d*d].reshape(d, d), np.ones(d))
    return fn.Os[:d]

def hybrid_ref(k, x, nx, Os, Mr2, S, s, r):
    gp, names = CEC2017Function._HYBRID[k]
    cf = len(gp)
    g_nx = []; tmp = 0
    for i in range(cf - 1):
        g_nx.append(int(math.ceil(gp[i]*nx))); tmp += g_nx[-1]
    g_nx.append(nx - tmp)
    g = [0]
    for i in range(1, cf): g.append(g[i - 1] + g_nx[i - 1])
    z = m.sr_func(x, nx, Os, Mr2, 1.0, s, r)
    if not isinstance(m.y, list) or len(m.y) < nx:
        m.y = [0.0]*nx
    m.y = list(m.y)
    for i in range(nx): m.y[i] = z[S[i] - 1]
    f = 0.0
    for i, name in enumerate(names):
        seg = m.y[g[i]:g[i] + g_nx[i]]          # C passes &y[G]; segment read before the call
        f += REF[name](seg, g_nx[i], Os, Mr2, 0, 0)
    return f

def composition_ref(fn, x, nx):
    comps, scales, delta, bias = CEC2017Function._COMPOSITION[fn.function_number - 20]
    fit = []
    for i, comp in enumerate(comps):
        Os_i = fn.Os[i*nx:(i + 1)*nx]
        Mr_i = fn.M[i*nx*nx:(i + 1)*nx*nx].reshape(nx, nx)
        if isinstance(comp, int):
            v = hybrid_ref(comp, x, nx, Os_i, Mr_i, fn.SS[i*nx:(i + 1)*nx], 1, 1)
        else:
            v = REF[comp](x, nx, Os_i, Mr_i, 1, 1)
        if scales[i] is not None:
            v = scales[i][0]*v/scales[i][1]
        fit.append(v)
    return m.cf_cal(x, nx, fn.Os, list(delta), list(bias), fit, len(fit))

def reference(fn, x):
    nx, f = fn.dimension, fn.function_number
    m.y = [0.0]*nx; m.z = [0.0]*nx
    if f <= 10:
        v = REF[fn._BASIC[f]](x, nx, fn.Os, fn.M[:nx*nx].reshape(nx, nx), 1, 1)
    elif f <= 20:
        v = hybrid_ref(f - 10, x, nx, fn.Os, fn.M[:nx*nx].reshape(nx, nx), fn.SS, 1, 1)
    else:
        v = composition_ref(fn, x, nx)
    return float(v) + fn.bias

if __name__ == "__main__":
    rng = np.random.default_rng(12345)
    dims = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10, 30, 50]
    npts = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    worst_opt, worst_rel = 0.0, 0.0
    for d in dims:
        for f in FUNCTIONS:
            fn = CEC2017Function(f, d, DATA)
            o = optimum(fn)
            err_opt = fn(o)[0] - BIAS[f]
            worst_opt = max(worst_opt, abs(err_opt))
            X = rng.uniform(-100, 100, (npts, d))
            X = np.vstack([X, o + rng.normal(0, 1e-3, (3, d)), o + rng.normal(0, 1.0, (3, d))])
            mine = fn(X)
            ref = np.array([reference(fn, x) for x in X])
            rel = np.max(np.abs(mine - ref)/np.maximum(1.0, np.abs(ref - fn.bias)))
            worst_rel = max(worst_rel, rel)
            flag = "" if rel < 1e-9 and abs(err_opt) < 1e-8 else "   <-- CHECK"
            print(f"D{d:3d} F{f:02d}: F(o)-F* = {err_opt: .3e} | max rel diff vs reference = {rel:.2e}{flag}", flush=True)
    print(f"WORST: optimum error {worst_opt:.3e}, relative difference {worst_rel:.3e}")
