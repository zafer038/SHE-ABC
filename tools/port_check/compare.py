import csv
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "src"))
from guabc.cec2017 import CEC2017Function, FUNCTIONS, DIMENSIONS

HERE = Path(__file__).parent
EXE = HERE / "cec17.exe"
CDIR = PKG / "external" / "cec2017"


def points(f, d, fn):
    rng = np.random.default_rng(1000*f + d)
    rnd = rng.uniform(-100, 100, (1000, d))
    corners = rng.choice([-100.0, 100.0], size=(100, d))
    mixed = rng.uniform(-100, 100, (100, d))
    mask = rng.random((100, d)) < 0.5
    mixed[mask] = rng.choice([-100.0, 100.0], size=mask.sum())
    o = fn.Os[:d].copy()
    special = [o, o + 1e-8, o - 1e-8]
    special += [o + 1e-3*rng.standard_normal(d) for _ in range(5)]
    special += [np.clip(o + rng.standard_normal(d), -100, 100) for _ in range(5)]
    special += [np.zeros(d), np.full(d, 100.0), np.full(d, -100.0)]
    kinds = ["random"]*1000 + ["boundary"]*200 + ["special"]*len(special)
    return np.vstack([rnd, corners, mixed, np.array(special)]), kinds


def official(f, d, X):
    with tempfile.TemporaryDirectory() as td:
        pin, pout = Path(td) / "x.bin", Path(td) / "v.bin"
        np.ascontiguousarray(X, dtype=np.float64).tofile(pin)
        subprocess.run([str(EXE), str(f), str(d), str(len(X)), str(pin), str(pout)], cwd=CDIR, check=True,
                       stdout=subprocess.DEVNULL)
        return np.fromfile(pout, dtype=np.float64)


def case(args):
    f, d = args
    fn = CEC2017Function(f, d)
    X, kinds = points(f, d, fn)
    c = official(f, d, X)
    p = fn(X)
    ad = np.abs(p - c)
    rd = ad/np.maximum(1.0, np.abs(c))
    kinds = np.array(kinds)
    out = {"F": f, "D": d, "points": len(X), "max_abs_diff": float(ad.max()), "max_rel_diff": float(rd.max()),
           "identical_share": float((ad == 0).mean()), "error_at_shift_official": float(c[1200] - 100.0*f),
           "error_at_shift_port": float(p[1200] - 100.0*f)}
    for k in ("random", "boundary", "special"):
        out[f"max_rel_diff_{k}"] = float(rd[kinds == k].max())
    return out


if __name__ == "__main__":
    jobs = [(f, d) for d in DIMENSIONS for f in FUNCTIONS]
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(case, jobs))
    with open(HERE / "port_validation.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    worst = max(rows, key=lambda r: r["max_rel_diff"])
    print("cases:", len(rows), "points:", sum(r["points"] for r in rows))
    print("max relative difference:", f"{worst['max_rel_diff']:.3g}", "at F", worst["F"], "D", worst["D"])
    print("max absolute difference:", f"{max(r['max_abs_diff'] for r in rows):.3g}")
    print("share of points with identical values:", f"{np.mean([r['identical_share'] for r in rows]):.3f}")
    for r in sorted(rows, key=lambda r: -r["max_rel_diff"])[:6]:
        print(r)
