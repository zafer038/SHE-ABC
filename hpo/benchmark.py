from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import torch
    from hpo.data import load_patches
    from hpo import segmentation as seg
    from hpo.common import decode_finetune
    from hpo.run_hpo import get_split, load_algorithms

    cfg = json.loads((ROOT / "configs" / "hpo_finetune.json").read_text(encoding="utf-8"))
    algos = load_algorithms(cfg)
    seg.set_determinism()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    t0 = time.perf_counter()
    images, labels, info = load_patches(ROOT / cfg["data_dir"])
    idx, _ = get_split(cfg, info.pop("names"))
    print(f"data loaded in {time.perf_counter() - t0:.0f}s: {info}")
    print(f"split: train {len(idx['train'])}, val {len(idx['val'])}, test {len(idx['test'])} (test not used)")
    data = seg.GPUData(images, labels)

    def run(x, seed):
        obj = seg.FinetuneObjective(data, idx["train"], idx["val"], cfg["epochs"], decode_finetune, seed=seed)
        t = time.perf_counter()
        _, inf = obj(x)
        return time.perf_counter() - t, inf

    times = []
    mid = np.zeros(6)
    same = []
    for k in range(3):
        dt, inf = run(mid, 0)
        times.append(dt); same.append(inf["miou"])
        print(f"  same seed repeat {k + 1}: {dt:.1f}s  val mIoU={inf['miou']:.5f}")
    seeds = [same[0]]
    for s in (1, 2):
        dt, inf = run(mid, s)
        times.append(dt); seeds.append(inf["miou"])
        print(f"  seed {s}: {dt:.1f}s  val mIoU={inf['miou']:.5f}")
    for x in (np.full(6, 50.0), np.full(6, -50.0)):
        dt, inf = run(x, 0)
        times.append(dt)
        print(f"  batch={inf['hp']['batch_size']:2d}: {dt:.1f}s  val mIoU={inf['miou']:.4f}")
    per_eval = float(np.mean(times[1:]))
    n_runs = len(algos)*len(cfg["seeds"])
    total_h = n_runs*cfg["budget"]*per_eval/3600
    print(f"\nnoise floor: same-seed range {max(same) - min(same):.5f} "
          f"({'deterministic' if max(same) == min(same) else 'NOT bit-identical'}), "
          f"seed-to-seed std {np.std(seeds, ddof=1):.5f}")
    print(f"~{per_eval:.1f}s per evaluation -> fine-tuning experiment: {len(algos)} algorithms x "
          f"{len(cfg['seeds'])} seeds x {cfg['budget']} evaluations ~= {total_h:.1f} hours")
    out = ROOT / "results" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "data": info, "seconds": times,
                               "same_seed_miou": same, "seed_miou": seeds, "per_eval_seconds": per_eval,
                               "estimated_hours": total_h, "versions": seg.library_versions()}, indent=1),
                   encoding="utf-8")
    print(f"-> {out}  (send this file back)")


if __name__ == "__main__":
    main()
