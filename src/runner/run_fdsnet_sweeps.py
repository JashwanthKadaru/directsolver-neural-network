"""FDSNet K- and r-sweep launcher.

Two phases per dataset, run sequentially:

* Phase 1 (K sweep): fixed r, vary K over a small grid. The K with the
  lowest validation loss is picked per dataset.
* Phase 2 (r sweep): vary r with K fixed to the Phase 1 winner.

Within each phase, experiments run in parallel across visible GPUs (or
sequentially on CPU). All results are logged to the console; nothing is
written to disk.

Usage:
    python src/runner/run_fdsnet_sweeps.py --datasets-dir datasets
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
    "nlse_1d": {
        "k_grid": [3, 5, 7], "phase1_r": 10,
        "r_grid": [2, 4, 6, 8, 10],
        "lr": 1e-3, "max_epochs": 2000,
    },
    "nlse_2d": {
        "k_grid": [3, 5, 7], "phase1_r": 10,
        "r_grid": [2, 4, 6, 8, 10],
        "lr": 1e-3, "max_epochs": 2000,
    },
    "burgers_1d": {
        "k_grid": [3, 5, 7], "phase1_r": 6,
        "r_grid": [2, 4, 6, 8, 10],
        "lr": 1e-3, "max_epochs": 2000,
    },
    "darcy_2d": {
        "k_grid": [3, 5, 7], "phase1_r": 6,
        "r_grid": [6, 9, 12],
        "lr": 2e-4, "max_epochs": 1000,
    },
}

SEED = 42
PATIENCE = 200
SPLIT = (20_000, 20_000)
BATCH_SIZE = 128


@dataclass
class SweepTask:
    dataset: str
    r: int
    K: int
    lr: float
    max_epochs: int


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
        max_epochs=task.max_epochs,
        patience=PATIENCE,
        lr=task.lr,
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
            "test_rel_mse": result.get("inference_test_rel_mse"),
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
                "test_rel_mse": result.get("inference_test_rel_mse"),
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


def _by_dataset_r_K(results: list[dict]) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        v = r.get("best_val_loss")
        if not isinstance(v, (int, float)):
            continue
        out[(r["dataset"], r["r"], r["K"])] = {"best_val_loss": v}
    return out


def _print_phase_summary(label: str, results: list[dict]) -> None:
    print(f"\n[{label}] per-(dataset, r, K):")
    print(f"  {'dataset':<14}{'r':<5}{'K':<5}{'best_val_loss'}")
    print(f"  " + "-" * 50)
    agg = _by_dataset_r_K(results)
    for key in sorted(agg.keys()):
        dataset, r, K = key
        print(f"  {dataset:<14}{r:<5}{K:<5}{agg[key]['best_val_loss']:.4e}")


def _pick_best_K_per_dataset(results: list[dict], datasets: list[str]) -> dict[str, int]:
    agg = _by_dataset_r_K(results)
    out: dict[str, int] = {}
    for dataset in datasets:
        candidates = [(K, agg[(d, r, K)]["best_val_loss"])
                      for (d, r, K) in agg if d == dataset]
        if not candidates:
            raise RuntimeError(f"Phase 1 has no successful runs for {dataset!r}")
        candidates.sort(key=lambda x: (x[1], x[0]))
        out[dataset] = candidates[0][0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", required=True)
    ap.add_argument("--per-gpu", type=int, default=4)
    args = ap.parse_args()
    if args.per_gpu < 1:
        ap.error("--per-gpu must be >= 1")

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[launcher] visible GPUs: {n_gpus}")
    print(f"[launcher] seed: {SEED}")

    phase1_tasks: list[SweepTask] = []
    for dataset, sw in SWEEPS.items():
        for K in sw["k_grid"]:
            phase1_tasks.append(SweepTask(
                dataset=dataset, r=sw["phase1_r"], K=K,
                lr=sw["lr"], max_epochs=sw["max_epochs"],
            ))
    print(f"[launcher] Phase 1: {len(phase1_tasks)} runs (K sweep)")
    phase1_results = _dispatch(phase1_tasks, args.datasets_dir, n_gpus, args.per_gpu)
    _print_phase_summary("phase1_K_sweep", phase1_results)

    best_K_per_dataset = _pick_best_K_per_dataset(phase1_results, list(SWEEPS.keys()))
    for dataset, K in best_K_per_dataset.items():
        print(f"[launcher] {dataset}: best K from Phase 1 = {K}")

    phase2_tasks: list[SweepTask] = []
    for dataset, sw in SWEEPS.items():
        K = best_K_per_dataset[dataset]
        for r in sw["r_grid"]:
            phase2_tasks.append(SweepTask(
                dataset=dataset, r=r, K=K,
                lr=sw["lr"], max_epochs=sw["max_epochs"],
            ))
    print(f"[launcher] Phase 2: {len(phase2_tasks)} runs (r sweep at best K)")
    phase2_results = _dispatch(phase2_tasks, args.datasets_dir, n_gpus, args.per_gpu)
    _print_phase_summary("phase2_r_sweep", phase2_results)

    final_agg = _by_dataset_r_K(phase2_results)
    print("\n[final] best (r, K) per dataset (lowest best_val_loss):")
    print(f"  {'dataset':<14}{'r':<5}{'K':<5}{'best_val_loss'}")
    print(f"  " + "-" * 50)
    for dataset in SWEEPS:
        cands = [(r, K, final_agg[(d, r, K)]) for (d, r, K) in final_agg if d == dataset]
        if not cands:
            continue
        cands.sort(key=lambda x: x[2]["best_val_loss"])
        r, K, v = cands[0]
        print(f"  {dataset:<14}{r:<5}{K:<5}{v['best_val_loss']:.4e}")

    n_failed = (sum(1 for r in phase1_results if r["status"] != "ok")
                + sum(1 for r in phase2_results if r["status"] != "ok"))
    if n_failed:
        print(f"[launcher] WARNING: {n_failed} failed runs across both phases")
        return 1
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
