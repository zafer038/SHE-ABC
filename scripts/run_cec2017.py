from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from guabc.cec2017 import BIAS, CEC2017Function, DIMENSIONS, official_run_seed
from guabc.evaluation_cec2017 import BudgetedEvaluator2017
from guabc.optimizers import OPTIMIZERS
# eshaguabc_v1..v6 and the leave-one-out variants are registered inside guabc.optimizers.


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    # Task order only; results are unaffected. Lets two machines share one
    # package from opposite ends without running the same task twice.
    parser.add_argument("--reverse", action="store_true")
    return parser.parse_args()


def _result_path(output_dir, dimension, function_number, algorithm_name, run):
    return (
        output_dir
        / f"D{dimension}"
        / f"F{function_number:02d}"
        / algorithm_name
        / f"run_{run:02d}.json"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_hash_file(file_path)))
    return digest.hexdigest()


def _experiment_provenance(config_path: Path) -> dict:
    tracked = {
        "config": _hash_file(config_path),
        "eshaguabc": _hash_file(ROOT / "src/guabc/eshaguabc.py"),
        "eshaguabc_v2": _hash_file(ROOT / "src/guabc/eshaguabc_v2.py"),
        "eshaguabc_v3": _hash_file(ROOT / "src/guabc/eshaguabc_v3.py"),
        "eshaguabc_v4": _hash_file(ROOT / "src/guabc/eshaguabc_v4.py"),
        "eshaguabc_v5": _hash_file(ROOT / "src/guabc/eshaguabc_v5.py"),
        "eshaguabc_v6": _hash_file(ROOT / "src/guabc/eshaguabc_v6.py"),
        "eshaguabc_v6_ablation": _hash_file(ROOT / "src/guabc/eshaguabc_v6_ablation.py"),
        "package_init": _hash_file(ROOT / "src/guabc/__init__.py"),
        "cec_adapter": _hash_file(ROOT / "src" / "guabc" / "cec2022.py"),
        "cec2017_port": _hash_file(ROOT / "src" / "guabc" / "cec2017.py"),
        "evaluator_cec2017": _hash_file(ROOT / "src" / "guabc" / "evaluation_cec2017.py"),
        "official_cec2017_c_source": _hash_file(ROOT / "external" / "cec2017" / "cec17_test_func.cpp"),
        "official_cec2017_input_data": _hash_tree(ROOT / "external" / "cec2017" / "input_data"),
        "evaluator": _hash_file(ROOT / "src" / "guabc" / "evaluation.py"),
        "optimizers": _hash_file(ROOT / "src" / "guabc" / "optimizers.py"),
        "runner": _hash_file(Path(__file__).resolve()),
        "official_cec_python": _hash_file(
            ROOT / "external" / "cec2022-python" / "Python" / "CEC2022.py"
        ),
        "official_cec_input_data": _hash_tree(
            ROOT / "external" / "cec2022-python" / "Python" / "input_data"
        ),
    }
    fingerprint = hashlib.sha256(
        json.dumps(tracked, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {"experiment_fingerprint": fingerprint, "sha256": tracked}


def _is_complete_result(
    path: Path,
    config: dict,
    optimizer,
    dimension: int,
    function_number: int,
    algorithm_name: str,
    run: int,
    experiment_fingerprint: str,
) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    required = {"fes_used", "best_error", "trace", "optimizer_parameters"}
    if not required.issubset(result):
        return False
    expected = {
        "protocol": config["protocol"],
        "dimension": dimension,
        "function": function_number,
        "algorithm": algorithm_name,
        "run": run,
        "max_fes": int(config["max_fes"][str(dimension)]),
        "optimizer_parameters": asdict(optimizer),
        "experiment_fingerprint": experiment_fingerprint,
    }
    mismatches = [key for key, value in expected.items() if result.get(key) != value]
    if mismatches:
        raise RuntimeError(
            f"Existing result is incompatible ({', '.join(mismatches)}): {path}. "
            "Use a new output directory for a changed protocol or optimizer."
        )
    return True


def _atomic_json_write(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _run_task(task):
    (
        config,
        output_dir_text,
        dimension,
        function_number,
        algorithm_name,
        run,
        experiment_fingerprint,
    ) = task
    output_dir = Path(output_dir_text)
    result_path = _result_path(
        output_dir, dimension, function_number, algorithm_name, run
    )
    optimizer = OPTIMIZERS[algorithm_name]
    if _is_complete_result(
        result_path,
        config,
        optimizer,
        dimension,
        function_number,
        algorithm_name,
        run,
        experiment_fingerprint,
    ):
        return dimension, function_number, algorithm_name, run, None, True

    max_fes = int(config["max_fes"][str(dimension)])
    objective = CEC2017Function(function_number, dimension)
    seed = official_run_seed(dimension, function_number, run, int(config["runs"]))
    evaluator = BudgetedEvaluator2017(
        objective=objective,
        dimension=dimension,
        max_fes=max_fes,
        optimum=BIAS[function_number],
    )
    started = time.perf_counter()
    rng = np.random.Generator(np.random.MT19937(seed))
    diagnostics = optimizer.run(evaluator, rng, int(config["population_size"]))
    if evaluator.terminated_at is not None:
        termination_reason = "target_reached"
        terminated_at = evaluator.terminated_at
    elif evaluator.evaluations >= max_fes:
        termination_reason = "max_fes"
        terminated_at = max_fes
    else:
        termination_reason = diagnostics.get("termination_reason", "optimizer_stop")
        terminated_at = evaluator.evaluations
    result = {
        "protocol": config["protocol"],
        "dimension": dimension,
        "function": function_number,
        "algorithm": algorithm_name,
        "run": run,
        "seed": seed,
        "experiment_fingerprint": experiment_fingerprint,
        "optimizer_parameters": asdict(optimizer),
        "optimizer_diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"best_position", "best_value"}
        },
        "max_fes": max_fes,
        "fes_used": evaluator.evaluations,
        "terminated_at": terminated_at,
        "termination_reason": termination_reason,
        "best_value": evaluator.best_value,
        "best_error": evaluator.best_error,
        "best_position": evaluator.best_position.tolist(),
        "trace": evaluator.finalize_trace(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json_write(result_path, result)
    return dimension, function_number, algorithm_name, run, evaluator.best_error, False


def main():
    args = parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers if args.workers is not None else int(config.get("workers", 1))
    if workers < 1:
        raise ValueError("workers must be at least 1")

    provenance = _experiment_provenance(config_path)
    metadata = {
        "config": config,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "workers": workers,
        "official_source": "https://github.com/P-N-Suganthan/CEC2017-BoundContrained (CEC17_fast_pow-C++.zip, cec17_test_func.cpp)",
        "provenance": provenance,
    }
    _atomic_json_write(output_dir / "metadata.json", metadata)

    tasks = []
    for dimension in config["dimensions"]:
        if int(dimension) not in DIMENSIONS:
            raise ValueError("CEC 2017 is defined for D=10, 30, 50 and 100")
        for function_number in config["functions"]:
            for algorithm_name in config["algorithms"]:
                if algorithm_name not in OPTIMIZERS:
                    raise KeyError(f"Unknown or unaudited optimizer: {algorithm_name}")
                for run in range(int(config["runs"])):
                    tasks.append(
                        (
                            config,
                            str(output_dir),
                            int(dimension),
                            int(function_number),
                            algorithm_name,
                            run,
                            provenance["experiment_fingerprint"],
                        )
                    )

    if args.reverse:
        tasks.reverse()
    iterator = map(_run_task, tasks)
    executor = None
    if workers > 1:
        executor = ProcessPoolExecutor(max_workers=workers)
        iterator = executor.map(_run_task, tasks, chunksize=1)
    try:
        for completed, item in enumerate(iterator, start=1):
            dimension, function_number, algorithm_name, run, error, skipped = item
            status = "cached" if skipped else f"error={error:.6e}"
            print(
                f"[{completed}/{len(tasks)}] D={dimension} F={function_number} "
                f"{algorithm_name} run={run} {status}",
                flush=True,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
