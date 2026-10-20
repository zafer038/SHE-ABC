"""Run an HPO comparison (algorithms x seeds) for one experiment config.

    python -m hpo.run_hpo --config configs/hpo_finetune.json
    python -m hpo.run_hpo --config configs/hpo_postproc.json
    python -m hpo.run_hpo --config configs/hpo_finetune.json --dry-run   # no GPU, synthetic objective

Protocol:
- Searches see ONLY the training and validation sets. The test set is evaluated once, after all
  runs, by `python -m hpo.final_test` (never inside a run).
- The data split is created once and stored in data/split.json; every later call reuses it.
- Every result carries an experiment fingerprint (config, algorithm settings, all code, split and
  data files). Resuming refuses to mix results whose fingerprint or settings differ.
- GPU runs use deterministic settings (see hpo.segmentation.set_determinism).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede the first CUDA/cuBLAS call

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpo.common import DECODERS, DIMENSIONS, HPOEvaluator, run_algorithm  # noqa: E402


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())


def data_digest(data_dir: Path) -> str:
    """Hash of every image/label file (names + contents)."""
    h = hashlib.sha256()
    for p in sorted(data_dir.rglob("*.tif")):
        h.update(p.relative_to(data_dir).as_posix().encode())
        h.update(bytes.fromhex(sha_file(p)))
    return h.hexdigest()


def load_algorithms(cfg) -> list:
    return cfg.get("algorithms") or json.loads((ROOT / cfg["algorithms_file"]).read_text(encoding="utf-8"))["algorithms"]


def experiment_fingerprint(cfg_path: Path, cfg: dict, split_hash: str | None, data_hash: str | None) -> dict:
    tracked = {"config": sha_file(cfg_path),
               "algorithms": sha_bytes(json.dumps(load_algorithms(cfg), sort_keys=True).encode()),
               "split": split_hash, "data": data_hash}
    for p in [*sorted((ROOT / "hpo").glob("*.py")), *sorted((ROOT / "src" / "guabc").glob("*.py"))]:
        tracked[p.relative_to(ROOT).as_posix()] = sha_file(p)
    return {"fingerprint": sha_bytes(json.dumps(tracked, sort_keys=True).encode()), "tracked": tracked}


def atomic_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.replace(path)


def get_split(cfg, names):
    """Create the split file once (fixed seed); afterwards always reuse it unchanged.
    cfg["split"] = {"tiles": [train, val, test]}: whole source tiles per subset (hpo.data.tile_split);
    a list of fractions gives the earlier random patch-level split."""
    from hpo.data import split, tile_split
    path = ROOT / cfg["split_file"]
    if not path.exists():
        payload = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": cfg["split_seed"]}
        if isinstance(cfg["split"], dict):
            tr, va, te, tiles = tile_split(names, tuple(cfg["split"]["tiles"]), cfg["split_seed"],
                                           cfg["split"].get("patches_per_tile", 400))
            payload.update({"type": "tile", "tiles": tiles})
        else:
            tr, va, te = split(len(names), tuple(cfg["split"]), cfg["split_seed"])
            payload.update({"type": "random", "fractions": cfg["split"]})
        payload.update({"train": [names[i] for i in tr], "val": [names[i] for i in va], "test": [names[i] for i in te]})
        atomic_write(path, payload)
        print(f"split created -> {path}", flush=True)
    s = json.loads(path.read_text(encoding="utf-8"))
    index = {n: i for i, n in enumerate(names)}
    idx = {k: np.array([index[n] for n in s[k]]) for k in ("train", "val", "test")}
    if set(idx["train"]) & set(idx["test"]) or set(idx["val"]) & set(idx["test"]) or set(idx["train"]) & set(idx["val"]):
        raise SystemExit("split.json has overlapping sets")
    return idx, sha_file(path)


def class_shares(labels, idx):
    """Share of each class among labelled pixels per subset; every class must occur in every subset."""
    from hpo.data import IGNORE
    from hpo.segmentation import NUM_CLASSES
    out = {}
    for k in ("train", "val", "test"):
        n = np.zeros(IGNORE + 1, np.int64)
        for i in idx[k]:  # per patch, to keep memory small
            n += np.bincount(labels[i].ravel(), minlength=IGNORE + 1)
        n = n[:NUM_CLASSES]
        if (n == 0).any():
            raise SystemExit(f"{k} set lacks class(es) {np.flatnonzero(n == 0).tolist()}; choose another split_seed")
        out[k] = (n/n.sum()).round(4).tolist()
    return out


class SyntheticObjective:
    """--dry-run: cheap stand-in with a known optimum, to test the pipeline without a GPU."""

    def __init__(self, dim):
        self.c = np.linspace(-40, 40, dim)

    def __call__(self, x):
        z = (np.asarray(x) - self.c)/50
        v = float(np.sum(z**2 - 0.1*np.cos(6*z)) + 0.1*len(z))
        return v/(1 + v), {"miou": 1/(1 + v)}


def build_experiment(cfg, dry_run):
    """Returns (make_objective(seed), info, split_hash, data_hash). No test-set access here."""
    exp, decode = cfg["experiment"], DECODERS[cfg["experiment"]]
    if dry_run:
        return (lambda seed: SyntheticObjective(DIMENSIONS[exp])), {"dry_run": True}, None, None
    import torch
    from hpo.data import load_patches
    from hpo import segmentation as seg
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not available; use --dry-run to test the pipeline")
    seg.set_determinism()
    data_dir = ROOT / cfg["data_dir"]
    images, labels, info = load_patches(data_dir)
    idx, split_hash = get_split(cfg, info.pop("names"))
    data_hash = data_digest(data_dir)
    info["class_share"] = class_shares(labels, idx)
    data = seg.GPUData(images, labels)
    info.update({"train": len(idx["train"]), "val": len(idx["val"]), "test": len(idx["test"]),
                 "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, **seg.library_versions()})
    print(f"data: {info}", flush=True)
    if exp == "finetune":
        return (lambda seed: seg.FinetuneObjective(data, idx["train"], idx["val"], cfg["epochs"], decode, seed)), \
            info, split_hash, data_hash
    # postproc: one base model trained once; only validation logits are cached for the search
    model, base_info = seg.load_or_train_base(cfg, data, idx, ROOT)
    info["base_model"] = base_info
    print(f"base model: val mIoU {base_info['val']['miou']:.4f}", flush=True)
    objective = seg.PostprocObjective(seg.predict_logits(model, data, idx["val"], flips=("id", "h", "v")),
                                      data.y[idx["val"]], decode)
    del model
    return (lambda seed: objective), info, split_hash, data_hash


def check_compatible(path: Path, fp: str, algo: dict, seed: int, budget: int):
    r = json.loads(path.read_text(encoding="utf-8"))
    problems = [k for k, ok in (("fingerprint", r.get("fingerprint") == fp), ("algorithm_spec", r.get("algorithm_spec") == algo),
                                ("seed", r.get("seed") == seed), ("budget", r.get("budget") == budget),
                                ("evaluations", r.get("evaluations") == budget)) if not ok]
    if problems:
        raise SystemExit(f"Existing result is incompatible ({', '.join(problems)}): {path}\n"
                         "Code, settings, split or data changed. Use a new output_dir instead of mixing results.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated algorithm names")
    ap.add_argument("--reverse", action="store_true", help="process (algorithm, seed) pairs from the end")
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    exp, D = cfg["experiment"], DIMENSIONS[cfg["experiment"]]
    out = ROOT / cfg["output_dir"] / ("dry_run" if args.dry_run else "")
    budget = cfg["budget"] if not args.dry_run else cfg.get("dry_run_budget", cfg["budget"])
    all_algos = load_algorithms(cfg)
    algos = [a for a in all_algos if not args.only or a["name"] in args.only.split(",")]
    tasks = [(a, s) for s in cfg["seeds"] for a in algos]  # seed-major: every algorithm gets seed 0 first
    if args.reverse:
        tasks.reverse()
    make_objective, info, split_hash, data_hash = build_experiment(cfg, args.dry_run)
    fp = experiment_fingerprint(cfg_path, cfg, split_hash, data_hash)
    meta_path = out / "metadata.json"
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8"))["fingerprint"] != fp["fingerprint"]:
        raise SystemExit(f"{meta_path} belongs to a different experiment fingerprint; use a new output_dir.")
    atomic_write(meta_path, {"fingerprint": fp["fingerprint"], "tracked": fp["tracked"], "config": cfg,
                             "algorithms": all_algos, "data": info, "python": sys.version,
                             "platform": platform.platform(), "dry_run": args.dry_run})
    print(f"experiment fingerprint {fp['fingerprint'][:12]}", flush=True)
    for n, (algo, seed) in enumerate(tasks, 1):
        path = out / algo["name"] / f"seed_{seed:02d}.json"
        if path.exists():
            check_compatible(path, fp["fingerprint"], algo, seed, budget)
            print(f"[{n}/{len(tasks)}] {algo['name']} seed={seed} cached (fingerprint ok)", flush=True)
            continue
        started = time.perf_counter()
        evaluator = HPOEvaluator(objective=make_objective(seed), dimension=D, max_fes=budget)
        diagnostics = run_algorithm(algo, evaluator, seed)
        result = {"experiment": exp, "fingerprint": fp["fingerprint"], "algorithm": algo["name"],
                  "algorithm_spec": algo, "seed": seed, "budget": budget, "evaluations": evaluator.evaluations,
                  "best_val_objective": evaluator.best_value, "best_val_miou": 1 - evaluator.best_value,
                  "best_x": evaluator.best_position.tolist(), "best_info": evaluator.best_info or {},
                  "search_seconds": time.perf_counter() - started, "diagnostics": diagnostics,
                  "log": evaluator.log}
        atomic_write(path, result)
        print(f"[{n}/{len(tasks)}] {algo['name']} seed={seed} best val mIoU={result['best_val_miou']:.4f} "
              f"evals={evaluator.evaluations} time={result['search_seconds']/60:.1f} min", flush=True)
    print("done. Test set untouched; run `python -m hpo.final_test --config ...` once all runs are finished.")


if __name__ == "__main__":
    main()
