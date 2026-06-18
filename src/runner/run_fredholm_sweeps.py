"""FDSNet r-sweep launcher for the Fredholm integral-equation datasets.

Single-phase sweep with ``K = 1`` (fully-linear FDSNet). For each (N, L, m)
anchor the rank ``r`` is swept over a small grid below ``m`` so the
off-diagonal low-rank assumption stays meaningful. One run per
(dataset, r); metrics are logged to the console.

Usage:
    python src/runner/run_fredholm_sweeps.py --datasets-dir datasets
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


SWEEPS: dict[str, dict] = {
    "fredholm_N1600":  {"N": 1600,  "L": 6, "m": 25,  "r_grid": [10, 12, 14, 16]},
    "fredholm_N6400":  {"N": 6400,  "L": 8, "m": 25,  "r_grid": [10, 12, 14, 16]},
    "fredholm_N14400": {"N": 14400, "L": 6, "m": 225, "r_grid": [10, 12, 14, 16]},
}

K_FIXED = 1
SEED = 42
SPLIT = (5_000, 5_000)
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 2000
PATIENCE = 150


@dataclass
class SweepTask:
    dataset: str
    r: int
    K: int


def _build_spec(task: SweepTask, datasets_dir: str):
    from runner.experiment import ExperimentSpec
    n_tr, n_te = SPLIT
    return ExperimentSpec(
        model="fdsnet", dataset=task.dataset,
        n_train=n_tr, n_test=n_te,
        batch_size=BATCH_SIZE,
        seed=SEED,
        datasets_dir=datasets_dir,
        device="cpu",
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        lr=LR,
    )


def _worker_loop(gpu_id: int, task_queue, result_queue, datasets_dir: str):
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import runner.builders as builders_mod
    from runner.experiment import run_one_experiment

    device_str = "cuda:0" if (gpu_id >= 0 and torch.cuda.is_available()) else "cpu"

    while True:
        item = task_queue.get()
        if item is None:
            break
        task: SweepTask = item
        spec = _build_spec(task, datasets_dir)
        spec.device = device_str
        dataset = task.dataset

        original_cfg = deepcopy(builders_mod.FDSNET_CONFIGS[dataset])
        patched_cfg = deepcopy(original_cfg)
        patched_cfg["r"] = task.r
        patched_cfg["K"] = task.K
        builders_mod.FDSNET_CONFIGS[dataset] = patched_cfg

        t0 = time.time()
        try:
            result = run_one_experiment(spec)
        finally:
            builders_mod.FDSNET_CONFIGS[dataset] = original_cfg

        dt = time.time() - t0
        status = result.get("status") if isinstance(result, dict) else "unknown"
        print(f"[worker-done] gpu{gpu_id} {dataset} r={task.r} K={task.K} "
              f"{status} dt={dt:.1f}s", flush=True)
        result_queue.put({
            "dataset": dataset, "r": task.r, "K": task.K,
            "status": status, "wall_seconds": dt, "gpu_id": gpu_id,
            "best_val_loss": result.get("best_val_loss"),
            "best_val_rel_l2": result.get("best_val_rel_l2"),
            "test_rel_l2": result.get("inference_test_rel_l2"),
            "params": result.get("params"),
        })


def _dispatch(tasks: list[SweepTask], datasets_dir: str, n_gpus: int,
              per_gpu: int = 1) -> list[dict]:
    if not tasks:
        return []

    if n_gpus == 0:
        import runner.builders as builders_mod
        from runner.experiment import run_one_experiment
        out = []
        for task in tasks:
            spec = _build_spec(task, datasets_dir)
            spec.device = "cpu"
            dataset = task.dataset
            original_cfg = deepcopy(builders_mod.FDSNET_CONFIGS[dataset])
            patched_cfg = deepcopy(original_cfg)
            patched_cfg["r"] = task.r
            patched_cfg["K"] = task.K
            builders_mod.FDSNET_CONFIGS[dataset] = patched_cfg
            t0 = time.time()
            try:
                result = run_one_experiment(spec)
            finally:
                builders_mod.FDSNET_CONFIGS[dataset] = original_cfg
            dt = time.time() - t0
            status = result.get("status") if isinstance(result, dict) else "unknown"
            out.append({
                "dataset": dataset, "r": task.r, "K": task.K,
                "status": status, "wall_seconds": dt, "gpu_id": -1,
                "best_val_loss": result.get("best_val_loss"),
                "best_val_rel_l2": result.get("best_val_rel_l2"),
                "test_rel_l2": result.get("inference_test_rel_l2"),
                "params": result.get("params"),
            })
            print(f"[cpu] {dataset} r={task.r} K={task.K} "
                  f"{status} {dt:.1f}s", flush=True)
        return out

    n_workers = max(1, n_gpus * per_gpu)
    print(f"[launcher] spawning {n_workers} workers ({per_gpu} per GPU)", flush=True)
    ctx = mp.get_context("spawn")
    task_q = ctx.Queue()
    result_q = ctx.Queue()
    for t in tasks:
        task_q.put(t)
    for _ in range(n_workers):
        task_q.put(None)
    workers = []
    for w_idx in range(n_workers):
        gpu = w_idx % n_gpus
        p = ctx.Process(target=_worker_loop,
                        args=(gpu, task_q, result_q, datasets_dir))
        p.start()
        workers.append(p)

    total = len(tasks)
    results = []
    while len(results) < total:
        r = result_q.get()
        results.append(r)
        print(f"[{len(results)}/{total}] gpu{r['gpu_id']} "
              f"{r['dataset']} r={r['r']} K={r['K']} "
              f"{r['status']} {r['wall_seconds']:.1f}s", flush=True)
    for p in workers:
        p.join()
    return results


def _validate_sweep_grid() -> None:
    for dataset, sw in SWEEPS.items():
        N, L, m = sw["N"], sw["L"], sw["m"]
        if N % (2 ** L) != 0:
            raise ValueError(f"{dataset}: N={N} not divisible by 2**L={2**L}")
        m_derived = N // (2 ** L)
        if m_derived != m:
            raise ValueError(f"{dataset}: declared m={m} != N/2^L={m_derived}")
        for r in sw["r_grid"]:
            if r >= m:
                raise ValueError(
                    f"{dataset}: r={r} >= m={m}; rank must be < m for the "
                    f"HODLR off-diagonal low-rank assumption to hold."
                )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--per-gpu", type=int, default=4)
    args = ap.parse_args()
    if args.per_gpu < 1:
        ap.error("--per-gpu must be >= 1")

    _validate_sweep_grid()

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[launcher] visible GPUs: {n_gpus}")
    print(f"[launcher] seed: {SEED}")

    tasks: list[SweepTask] = []
    for dataset, sw in SWEEPS.items():
        for r in sw["r_grid"]:
            tasks.append(SweepTask(dataset=dataset, r=r, K=K_FIXED))
    print(f"[launcher] {len(tasks)} runs scheduled "
          f"({len(SWEEPS)} datasets x {len(SWEEPS['fredholm_N1600']['r_grid'])} r-values)")

    results = _dispatch(tasks, args.datasets_dir, n_gpus, args.per_gpu)

    print("\n[summary] per-(dataset, r):")
    print(f"  {'dataset':<18}{'r':<5}{'best_val_loss':<18}{'test_rel_l2'}")
    print(f"  " + "-" * 60)
    rows_ok = [r for r in results if r["status"] == "ok"]
    for r in sorted(rows_ok, key=lambda x: (x["dataset"], x["r"])):
        bv = r.get("best_val_loss")
        tv = r.get("test_rel_l2")
        bv_s = f"{bv:.4e}" if isinstance(bv, (int, float)) else "n/a"
        tv_s = f"{tv:.4e}" if isinstance(tv, (int, float)) else "n/a"
        print(f"  {r['dataset']:<18}{r['r']:<5}{bv_s:<18}{tv_s}")

    n_failed = sum(1 for r in results if r["status"] != "ok")
    if n_failed:
        print(f"[launcher] WARNING: {n_failed} failures")
        return 1
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
