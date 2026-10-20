"""One-time test-set evaluation, run once after ALL HPO runs of an experiment are finished.

    python -m hpo.final_test --config configs/hpo_finetune.json
    python -m hpo.final_test --config configs/hpo_postproc.json

- Refuses to start unless every (algorithm, seed) run exists and matches the experiment fingerprint.
- Refuses to run a second time (results/<exp>/final_test.json exists), so the test set is used once.
- Fine-tuning: each run's best setting is retrained with the run's seed; the validation mIoU is
  re-measured too, and its difference to the stored value is reported (run-to-run reproducibility).
  The setting is also retrained with each training seed in test_retrain_seeds (config), and the
  validation and test mIoU of these models are stored as well (sensitivity to the training seed).
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hpo.common import DECODERS  # noqa: E402
from hpo.run_hpo import atomic_write, data_digest, experiment_fingerprint, get_split, load_algorithms  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg_path = ROOT / args.config
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    exp, decode = cfg["experiment"], DECODERS[cfg["experiment"]]
    out = ROOT / cfg["output_dir"]
    final_path = out / "final_test.json"
    if final_path.exists():
        raise SystemExit(f"{final_path} already exists: the test set has been used once already.")
    if not (ROOT / cfg["split_file"]).exists():
        raise SystemExit("split.json missing: run the HPO experiment first")
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))

    import torch
    from hpo.data import load_patches
    from hpo import segmentation as seg
    seg.set_determinism()
    data_dir = ROOT / cfg["data_dir"]
    images, labels, info = load_patches(data_dir)
    idx, split_hash = get_split(cfg, info.pop("names"))
    fp = experiment_fingerprint(cfg_path, cfg, split_hash, data_digest(data_dir))["fingerprint"]
    if fp != meta["fingerprint"]:
        raise SystemExit("Current code/config/split/data differ from the HPO runs (fingerprint mismatch).")
    runs = []
    for algo in load_algorithms(cfg):
        for seed in cfg["seeds"]:
            p = out / algo["name"] / f"seed_{seed:02d}.json"
            if not p.exists():
                raise SystemExit(f"missing run {p}: finish all HPO runs before the test evaluation")
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["fingerprint"] != fp:
                raise SystemExit(f"{p} has a different fingerprint")
            runs.append(r)
    data = seg.GPUData(images, labels)
    started = time.perf_counter()
    rows = []
    if exp == "finetune":
        for n, r in enumerate(runs, 1):
            hp = decode(np.array(r["best_x"]))
            model = seg.train(hp, data, idx["train"], cfg["epochs"], r["seed"])
            val = seg.evaluate_model(model, data, idx["val"])
            test = seg.evaluate_model(model, data, idx["test"])
            row = {"algorithm": r["algorithm"], "seed": r["seed"], "hp": hp, "diverged": model.diverged,
                   "val_stored": r["best_val_miou"], "val_rescored": val["miou"], "test": test, "retrain": []}
            del model
            torch.cuda.empty_cache()
            for ts in cfg.get("test_retrain_seeds", []):
                m = seg.train(hp, data, idx["train"], cfg["epochs"], ts)
                bad = bool(m.diverged)
                row["retrain"].append({"train_seed": ts, "diverged": bad,
                                       "val": {"miou": 0.0} if bad else seg.evaluate_model(m, data, idx["val"]),
                                       "test": {"miou": 0.0} if bad else seg.evaluate_model(m, data, idx["test"])})
                del m
                torch.cuda.empty_cache()
            rows.append(row)
            extra = np.mean([x["test"]["miou"] for x in row["retrain"]]) if row["retrain"] else np.nan
            print(f"[{n}/{len(runs)}] {r['algorithm']} seed={r['seed']} test mIoU={test['miou']:.4f} "
                  f"(val stored {r['best_val_miou']:.4f} / re-scored {val['miou']:.4f}; "
                  f"other training seeds: test {extra:.4f})", flush=True)
    else:
        model, base = seg.load_or_train_base(cfg, data, idx, ROOT)
        te_logits = seg.predict_logits(model, data, idx["test"], flips=("id", "h", "v"))
        obj = seg.PostprocObjective(te_logits, data.y[idx["test"]], decode)
        base_test = seg.evaluate_model(model, data, idx["test"])
        for r in runs:
            p = decode(np.array(r["best_x"]))
            test = seg.scores(obj.predict_cm(p))
            rows.append({"algorithm": r["algorithm"], "seed": r["seed"], "params": p,
                         "val_stored": r["best_val_miou"], "test": test})
        print(f"base model test mIoU (no post-processing): {base_test['miou']:.4f}", flush=True)
    diffs = [abs(x["val_rescored"] - x["val_stored"]) for x in rows if "val_rescored" in x]
    payload = {"fingerprint": fp, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "seconds": time.perf_counter() - started, "runs": rows,
               "reproducibility_val_abs_diff": {"max": max(diffs), "mean": float(np.mean(diffs))} if diffs else None}
    if exp == "postproc":
        payload["base_model_test"] = base_test
    atomic_write(final_path, payload)
    print(f"final test -> {final_path}")
    if diffs:
        print(f"re-scored vs stored validation mIoU: max |diff| {max(diffs):.4f}, mean {np.mean(diffs):.4f}")


if __name__ == "__main__":
    main()
