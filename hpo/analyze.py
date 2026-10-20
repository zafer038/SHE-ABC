"""Summarise an HPO comparison: test/val mIoU per algorithm, paired tests vs SHE-ABC (eshaguabc_v6),
Friedman ranks and anytime curves.

    python -m hpo.analyze --results results/finetune
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

REF = "eshaguabc_v6"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    final, tests, tests_all = root / "final_test.json", {}, {}
    if final.exists():
        ft = json.loads(final.read_text(encoding="utf-8"))
        tests = {(x["algorithm"], x["seed"]): x["test"]["miou"] for x in ft["runs"]}
        # mean test mIoU over the run seed and the extra training seeds (fine-tuning only)
        tests_all = {(x["algorithm"], x["seed"]): float(np.mean([x["test"]["miou"]] + [y["test"]["miou"] for y in x["retrain"]]))
                     for x in ft["runs"] if x.get("retrain")}
        if ft.get("reproducibility_val_abs_diff"):
            d = ft["reproducibility_val_abs_diff"]
            print(f"final test: re-scored vs stored val mIoU |diff| max {d['max']:.4f}, mean {d['mean']:.4f}")
    else:
        print("final_test.json not found: test columns empty (run `python -m hpo.final_test` after all runs)")
    runs = {}
    for p in sorted(root.glob("*/seed_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if tests and r["fingerprint"] != ft["fingerprint"]:
            raise SystemExit(f"{p} does not match the fingerprint of final_test.json")
        runs.setdefault(r["algorithm"], {})[r["seed"]] = {
            "val": r["best_val_miou"], "test": tests.get((r["algorithm"], r["seed"]), np.nan),
            "test_all": tests_all.get((r["algorithm"], r["seed"]), np.nan),
            "curve": [1 - e["best_so_far"] for e in r["log"]], "minutes": r["search_seconds"]/60}
    algos = sorted(runs, key=lambda a: (a != REF, a))
    seeds = sorted(set.intersection(*(set(v) for v in runs.values())))
    print(f"results: {root}  algorithms: {len(algos)}  common seeds: {len(seeds)}\n")
    measures = [("test", "test mIoU (training seed of the run)")]
    if tests_all:
        measures.append(("test_all", "test mIoU, mean over the run seed and the extra training seeds"))
    tables = {}
    for key, title in measures:
        print(f"\n== {title}")
        print(f"{'algorithm':16s} {'test mIoU mean±std':>22s} {'val mIoU mean':>14s} {'min/run':>8s}   vs {REF} (test, Wilcoxon)")
        table = tables[key] = {}
        for a in algos:
            t = np.array([runs[a][s][key] for s in seeds]); v = np.array([runs[a][s]["val"] for s in seeds])
            table[a] = t
            cmp = ""
            if a != REF and REF in runs:
                ref = np.array([runs[REF][s][key] for s in seeds])
                if np.all(np.isfinite(t)) and len(seeds) >= 6 and (ref != t).any():
                    p = wilcoxon(ref, t).pvalue
                    better = int((ref > t).sum()); worse = int((ref < t).sum())
                    verdict = "v6 better" if p < .05 and np.median(ref - t) > 0 else "v6 worse" if p < .05 else "no sig. difference"
                    cmp = f"{better}/{worse} runs, p={p:.3g} -> {verdict}"
                elif not np.all(np.isfinite(t)):
                    cmp = "(no test mIoU yet)"
                else:
                    cmp = "(need >= 6 seeds for a paired test)"
            mins = np.mean([runs[a][s]["minutes"] for s in seeds])
            tcol = f"{t.mean():10.4f} ± {t.std(ddof=1) if len(t) > 1 else 0:.4f}" if np.all(np.isfinite(t)) else f"{'-':>19s}"
            print(f"{a:16s} {tcol}   {v.mean():12.4f} {mins:8.1f}   {cmp}")
        mat = np.array([table[a] for a in algos]).T  # seeds x algorithms, higher is better
        if len(seeds) >= 2 and np.all(np.isfinite(mat)):
            ranks = np.array([rankdata(-row) for row in mat]).mean(0)
            print("\nFriedman mean rank (1 = best): " + ", ".join(f"{a} {r:.2f}" for a, r in sorted(zip(algos, ranks), key=lambda z: z[1])))
            if len(algos) >= 3:
                print(f"Friedman test p = {friedmanchisquare(*mat.T).pvalue:.3g}")
    # anytime curves: median best-so-far validation mIoU vs evaluations
    curves = {a: np.median(np.array([runs[a][s]["curve"] for s in seeds]), axis=0) for a in algos}
    n = min(len(c) for c in curves.values())
    marks = sorted(set(int(round(f*n)) for f in (0.1, 0.25, 0.5, 0.75, 1.0)))
    print("\nmedian best-so-far val mIoU at evaluation:", marks)
    for a in algos:
        print(f"  {a:16s} " + " ".join(f"{curves[a][m - 1]:.4f}" for m in marks))
    out = root / "summary.json"
    out.write_text(json.dumps({"seeds": seeds, **{f"{k}_miou": {a: v[a].tolist() for a in algos} for k, v in tables.items()},
                               "anytime_median": {a: curves[a].tolist() for a in algos}}, indent=1), encoding="utf-8")
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()
