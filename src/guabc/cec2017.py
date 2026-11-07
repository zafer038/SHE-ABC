from __future__ import annotations

import math
from pathlib import Path

import numpy as np

INF = 1.0e99
E = 2.7182818284590452353602874713526625
PI = 3.1415926535897932384626433832795029
CF_NUM = 10
FUNCTIONS = tuple(f for f in range(1, 31) if f != 2)
DIMENSIONS = (10, 30, 50, 100)
BIAS = {f: 100.0*f for f in range(1, 31)}
CHECKPOINT_FRACTIONS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "external" / "cec2017" / "input_data"


def official_checkpoints(dimension: int, max_fes: int) -> list[int]:
    return [max(1, min(max_fes, int(round(c*max_fes)))) for c in CHECKPOINT_FRACTIONS]


def max_fes(dimension: int) -> int:
    return 10000*dimension


def official_run_seed(dimension: int, function_number: int, run: int, runs: int = 30) -> int:
    if dimension not in DIMENSIONS:
        raise ValueError("Supported dimensions: 10, 30, 50, 100")
    if not 0 <= run < runs:
        raise ValueError(f"run must be in 0..{runs - 1}")
    path = Path(__file__).resolve().parents[2] / "external" / "cec2022-python" / "Python" / "input_data" / "Rand_Seeds.txt"
    seeds = np.loadtxt(path, dtype=float).astype(int)
    raw_index = int(dimension/10*function_number*runs + (run + 1)) - runs
    return int(seeds[raw_index % 1000])


def _numbers(path: Path, dtype=float) -> np.ndarray:
    return np.array(path.read_text().split(), dtype=dtype)


class CEC2017Function:
    def __init__(self, function_number: int, dimension: int, directory: Path | None = None):
        if function_number not in BIAS:
            raise ValueError("function_number must be in 1..30")
        if dimension not in DIMENSIONS:
            raise ValueError("Supported dimensions: 10, 30, 50, 100")
        self.function_number = f = int(function_number)
        self.dimension = nx = int(dimension)
        self.bias = BIAS[f]
        d = Path(directory) if directory else data_dir()
        m = _numbers(d / f"M_{f}_D{nx}.txt")
        if f < 20:
            self.M = m[:nx*nx].copy()
            self.Os = _numbers(d / f"shift_data_{f}.txt")[:nx].copy()
        else:
            self.M = m[:CF_NUM*nx*nx].copy()
            rows = [line.split() for line in (d / f"shift_data_{f}.txt").read_text().splitlines() if line.strip()]
            self.Os = np.array([float(v) for row in rows[:CF_NUM] for v in row[:nx]])
        self.SS = None
        if 11 <= f <= 20:
            self.SS = _numbers(d / f"shuffle_data_{f}_D{nx}.txt", dtype=float).astype(int)[:nx]
        elif f in (29, 30):
            self.SS = _numbers(d / f"shuffle_data_{f}_D{nx}.txt", dtype=float).astype(int)[:nx*CF_NUM]
        self.y = np.zeros(nx)
        self.z = np.zeros(nx)

    def _sr(self, x, nx, Os, Mr, rate, s_flag, r_flag):
        y = self.y
        if s_flag == 1:
            if r_flag == 1:
                y[:nx] = x[:nx] - Os[:nx]
                y[:nx] *= rate
                return Mr[:nx*nx].reshape(nx, nx) @ y[:nx]
            return (x[:nx] - Os[:nx])*rate
        if r_flag == 1:
            y[:nx] = x[:nx]*rate
            return Mr[:nx*nx].reshape(nx, nx) @ y[:nx]
        return x[:nx]*rate

    def ellips(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        i = np.arange(nx)
        return float(np.sum(np.power(10.0, 6.0*i/(nx - 1))*z*z))

    def bent_cigar(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        return float(z[0]*z[0] + np.sum(1.0e6*z[1:]*z[1:]))

    def discus(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        return float(1.0e6*z[0]*z[0] + np.sum(z[1:]*z[1:]))

    def rosenbrock(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 2.048/100.0, s, r) + 1.0
        t1 = z[:-1]*z[:-1] - z[1:]
        t2 = z[:-1] - 1.0
        return float(np.sum(100.0*t1*t1 + t2*t2))

    def schaffer_F7(self, x, nx, Os, Mr, s, r):
        self._sr(x, nx, Os, Mr, 1.0, s, r)
        y = self.y[:nx]
        zz = np.power(y[:-1]*y[:-1] + y[1:]*y[1:], 0.5)
        tmp = np.sin(50.0*np.power(zz, 0.2))
        f = float(np.sum(np.power(zz, 0.5) + np.power(zz, 0.5)*tmp*tmp))
        return f*f/(nx - 1)/(nx - 1)

    def ackley(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        sum1 = -0.2*math.sqrt(float(np.sum(z*z))/nx)
        sum2 = float(np.sum(np.cos(2.0*PI*z)))/nx
        return E - 20.0*math.exp(sum1) - math.exp(sum2) + 20.0

    _WJ = np.arange(21)
    _WA = np.power(0.5, _WJ)
    _WB = np.power(3.0, _WJ)

    def weierstrass(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 0.5/100.0, s, r)
        terms = self._WA*np.cos(2.0*PI*self._WB*(z[:, None] + 0.5))
        sum2 = float(np.sum(self._WA*np.cos(2.0*PI*self._WB*0.5)))
        return float(np.sum(terms)) - nx*sum2

    def griewank(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 600.0/100.0, s, r)
        p = float(np.prod(np.cos(z/np.sqrt(1.0 + np.arange(nx)))))
        return 1.0 + float(np.sum(z*z))/4000.0 - p

    def rastrigin(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 5.12/100.0, s, r)
        return float(np.sum(z*z - 10.0*np.cos(2.0*PI*z) + 10.0))

    def step_rastrigin(self, x, nx, Os, Mr, s, r):
        return self.rastrigin(x, nx, Os, Mr, s, r)

    def schwefel(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1000.0/100.0, s, r) + 4.209687462275036e+002
        f = np.empty(nx)
        hi, lo = z > 500, z < -500
        mid = ~(hi | lo)
        zh = np.fmod(z[hi], 500)
        f[hi] = -(500.0 - zh)*np.sin(np.power(500.0 - zh, 0.5)) + ((z[hi] - 500.0)/100)**2/nx
        zl = np.fmod(np.abs(z[lo]), 500)
        f[lo] = -(-500.0 + zl)*np.sin(np.power(500.0 - zl, 0.5)) + ((z[lo] + 500.0)/100)**2/nx
        f[mid] = -z[mid]*np.sin(np.power(np.abs(z[mid]), 0.5))
        return float(np.sum(f)) + 4.189828872724338e+002*nx

    _K2 = np.power(2.0, np.arange(1, 33))

    def katsuura(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 5.0/100.0, s, r)
        t2 = self._K2*z[:, None]
        temp = np.sum(np.abs(t2 - np.floor(t2 + 0.5))/self._K2, axis=1)
        f = float(np.prod(np.power(1.0 + np.arange(1, nx + 1)*temp, 10.0/math.pow(1.0*nx, 1.2))))
        t = 10.0/nx/nx
        return f*t - t

    def bi_rastrigin(self, x, nx, Os, Mr, s, r):
        mu0, d = 2.5, 1.0
        sc = 1.0 - 1.0/(2.0*math.pow(nx + 20.0, 0.5) - 8.2)
        mu1 = -math.pow((mu0*mu0 - d)/sc, 0.5)
        y = self.y
        if s == 1:
            y[:nx] = x[:nx] - Os[:nx]
        else:
            y[:nx] = x[:nx]
        y[:nx] *= 10.0/100.0
        tmpx = 2*y[:nx]
        tmpx[Os[:nx] < 0.0] *= -1.0
        zz = tmpx.copy()
        tmpx = tmpx + mu0
        tmp1 = float(np.sum((tmpx - mu0)**2))
        tmp2 = float(np.sum((tmpx - mu1)**2))*sc + d*nx
        if r == 1:
            y[:nx] = Mr[:nx*nx].reshape(nx, nx) @ zz
            tmp = float(np.sum(np.cos(2.0*PI*y[:nx])))
        else:
            tmp = float(np.sum(np.cos(2.0*PI*zz)))
        return min(tmp1, tmp2) + 10.0*(nx - tmp)

    def grie_rosen(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 5.0/100.0, s, r) + 1.0
        a = z
        b = np.roll(z, -1)
        t1 = a*a - b
        t2 = a - 1.0
        temp = 100.0*t1*t1 + t2*t2
        return float(np.sum((temp*temp)/4000.0 - np.cos(temp) + 1.0))

    def escaffer6(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        a, b = z, np.roll(z, -1)
        sq = a*a + b*b
        t1 = np.sin(np.sqrt(sq))
        t1 = t1*t1
        t2 = 1.0 + 0.001*sq
        return float(np.sum(0.5 + (t1 - 0.5)/(t2*t2)))

    def happycat(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 5.0/100.0, s, r) - 1.0
        r2, sz = float(np.sum(z*z)), float(np.sum(z))
        return math.pow(abs(r2 - nx), 2*(1.0/8.0)) + (0.5*r2 + sz)/nx + 0.5

    def hgbat(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 5.0/100.0, s, r) - 1.0
        r2, sz = float(np.sum(z*z)), float(np.sum(z))
        return math.pow(abs(math.pow(r2, 2.0) - math.pow(sz, 2.0)), 2*(1.0/4.0)) + (0.5*r2 + sz)/nx + 0.5

    def zakharov(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        sum1 = float(np.sum(z*z))
        sum2 = float(np.sum(0.5*np.arange(1, nx + 1)*z))
        return sum1 + math.pow(sum2, 2) + math.pow(sum2, 4)

    def levy(self, x, nx, Os, Mr, s, r):
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        w = 1.0 + (z - 1.0)/4.0
        term1 = math.pow(math.sin(PI*w[0]), 2)
        term3 = math.pow(w[-1] - 1, 2)*(1 + math.pow(math.sin(2*PI*w[-1]), 2))
        wi = w[:-1]
        return term1 + float(np.sum((wi - 1)**2*(1 + 10*np.sin(PI*wi + 1)**2))) + term3

    _HYBRID = {
        1: ((0.2, 0.4, 0.4), ("zakharov", "rosenbrock", "rastrigin")),
        2: ((0.3, 0.3, 0.4), ("ellips", "schwefel", "bent_cigar")),
        3: ((0.3, 0.3, 0.4), ("bent_cigar", "rosenbrock", "bi_rastrigin")),
        4: ((0.2, 0.2, 0.2, 0.4), ("ellips", "ackley", "schaffer_F7", "rastrigin")),
        5: ((0.2, 0.2, 0.3, 0.3), ("bent_cigar", "hgbat", "rastrigin", "rosenbrock")),
        6: ((0.2, 0.2, 0.3, 0.3), ("escaffer6", "hgbat", "rosenbrock", "schwefel")),
        7: ((0.1, 0.2, 0.2, 0.2, 0.3), ("katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin")),
        8: ((0.2, 0.2, 0.2, 0.2, 0.2), ("ellips", "ackley", "rastrigin", "hgbat", "discus")),
        9: ((0.2, 0.2, 0.2, 0.2, 0.2), ("bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6")),
        10: ((0.1, 0.1, 0.2, 0.2, 0.2, 0.2), ("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_F7")),
    }

    def hybrid(self, k, x, nx, Os, Mr, S, s, r):
        gp, names = self._HYBRID[k]
        cf = len(gp)
        g_nx = [int(math.ceil(gp[i]*nx)) for i in range(cf - 1)]
        g_nx.append(nx - sum(g_nx))
        g = np.concatenate(([0], np.cumsum(g_nx)[:-1])).astype(int)
        z = self._sr(x, nx, Os, Mr, 1.0, s, r)
        self.y[:nx] = z[S[:nx] - 1]
        fit = 0.0
        for i, name in enumerate(names):
            fit += getattr(self, name)(self.y[g[i]:g[i] + g_nx[i]], g_nx[i], Os, Mr, 0, 0)
        return fit

    @staticmethod
    def cf_cal(x, nx, Os, delta, bias, fit):
        cf = len(fit)
        w = np.empty(cf)
        for i in range(cf):
            fit[i] += bias[i]
            wi = float(np.sum((x[:nx] - Os[i*nx:(i + 1)*nx])**2))
            w[i] = math.pow(1.0/wi, 0.5)*math.exp(-wi/2.0/nx/math.pow(delta[i], 2.0)) if wi != 0 else INF
        if w.max() == 0:
            w[:] = 1.0
            w_sum = float(cf)
        else:
            w_sum = float(np.sum(w))
        return float(np.sum(w/w_sum*np.asarray(fit)))

    _COMPOSITION = {
        1: (("rosenbrock", "ellips", "rastrigin"), (None, (10000, 1e10), None), (10, 20, 30), (0, 100, 200)),
        2: (("rastrigin", "griewank", "schwefel"), (None, (1000, 100), None), (10, 20, 30), (0, 100, 200)),
        3: (("rosenbrock", "ackley", "schwefel", "rastrigin"), (None, (1000, 100), None, None),
            (10, 20, 30, 40), (0, 100, 200, 300)),
        4: (("ackley", "ellips", "griewank", "rastrigin"), ((1000, 100), (10000, 1e10), (1000, 100), None),
            (10, 20, 30, 40), (0, 100, 200, 300)),
        5: (("rastrigin", "happycat", "ackley", "discus", "rosenbrock"),
            ((10000, 1e3), (1000, 1e3), (1000, 100), (10000, 1e10), None),
            (10, 20, 30, 40, 50), (0, 100, 200, 300, 400)),
        6: (("escaffer6", "schwefel", "griewank", "rosenbrock", "rastrigin"),
            ((10000, 2e7), None, (1000, 100), None, (10000, 1e3)),
            (10, 20, 20, 30, 40), (0, 100, 200, 300, 400)),
        7: (("hgbat", "rastrigin", "schwefel", "bent_cigar", "ellips", "escaffer6"),
            ((10000, 1000), (10000, 1e3), (10000, 4e3), (10000, 1e30), (10000, 1e10), (10000, 2e7)),
            (10, 20, 30, 40, 50, 60), (0, 100, 200, 300, 400, 500)),
        8: (("ackley", "griewank", "discus", "rosenbrock", "happycat", "escaffer6"),
            ((1000, 100), (1000, 100), (10000, 1e10), None, (1000, 1e3), (10000, 2e7)),
            (10, 20, 30, 40, 50, 60), (0, 100, 200, 300, 400, 500)),
        9: ((5, 6, 7), (None, None, None), (10, 30, 50), (0, 100, 200)),
        10: ((5, 8, 9), (None, None, None), (10, 30, 50), (0, 100, 200)),
    }

    def composition(self, k, x, nx):
        comps, scales, delta, bias = self._COMPOSITION[k]
        fit = []
        for i, comp in enumerate(comps):
            Os_i = self.Os[i*nx:(i + 1)*nx]
            Mr_i = self.M[i*nx*nx:(i + 1)*nx*nx]
            if isinstance(comp, int):
                v = self.hybrid(comp, x, nx, Os_i, Mr_i, self.SS[i*nx:(i + 1)*nx], 1, 1)
            else:
                v = getattr(self, comp)(x, nx, Os_i, Mr_i, 1, 1)
            if scales[i] is not None:
                a, b = scales[i]
                v = a*v/b
            fit.append(v)
        return self.cf_cal(x, nx, self.Os, delta, bias, fit)

    _BASIC = {1: "bent_cigar", 3: "zakharov", 4: "rosenbrock", 5: "rastrigin", 6: "schaffer_F7",
              7: "bi_rastrigin", 8: "step_rastrigin", 9: "levy", 10: "schwefel"}

    def _evaluate_one(self, x: np.ndarray) -> float:
        f, nx = self.function_number, self.dimension
        if f <= 10:
            v = getattr(self, self._BASIC[f])(x, nx, self.Os, self.M, 1, 1)
        elif f <= 20:
            v = self.hybrid(f - 10, x, nx, self.Os, self.M, self.SS, 1, 1)
        else:
            v = self.composition(f - 20, x, nx)
        return float(v) + self.bias

    def __call__(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=float)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"Expected shape (n, {self.dimension}), received {x.shape}")
        return np.array([self._evaluate_one(point) for point in x], dtype=float)
