"""Ablation launcher: trains every (model, dataset, split) combination
across the four PDE benchmarks and five model families. Auto-detects
available GPUs and dispatches one worker per GPU (or sequential CPU
fallback). All progress and final per-configuration metrics are logged
to the console.

Usage:
    python src/runner/run_all.py --datasets-dir datasets
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


MODELS = ["fdsnet", "mnnh", "deeponet", "mlp", "fno"]
DATASETS_LIST = ["nlse_1d", "burgers_1d", "nlse_2d", "darcy_2d"]
SPLITS = {
    "nlse_1d":    [(20_000, 20_000)],
    "burgers_1d": [(20_000, 20_000)],
    "nlse_2d":    [(20_000, 20_000)],
    "darcy_2d":   [(20_000, 20_000)],
}
SEED = 42

BATCH_SIZE = {
    "nlse_1d":    128,
    "burgers_1d": 128,
    "nlse_2d":    128,
    "darcy_2d":   128,
}


def _cost(model: str, dataset: str, n_train: int) -> int:
    base = {"fdsnet": 6, "deeponet": 3, "fno": 2, "mnnh": 1, "mlp": 1}
    data = {"nlse_2d": 4, "burgers_1d": 4, "darcy_2d": 2, "nlse_1d": 1}
    return base[model] * data[dataset] * (n_train // 5_000)


def build_specs(datasets_dir: str) -> list:
    from runner.experiment import ExperimentSpec
    specs = []
    for dataset in DATASETS_LIST:
        for n_tr, n_te in SPLITS[dataset]:
            for model in MODELS:
                max_epochs = 1000 if dataset == "darcy_2d" else 2000
                lr = 2e-4 if dataset == "darcy_2d" else 1e-3
                specs.append(ExperimentSpec(
                    model=model, dataset=dataset,
                    n_train=n_tr, n_test=n_te,
                    batch_size=BATCH_SIZE[dataset],
                    seed=SEED,
                    datasets_dir=datasets_dir,
                    device="cpu",
                    max_epochs=max_epochs,
                    lr=lr,
                ))
    specs.sort(key=lambda s: -_cost(s.model, s.dataset, s.n_train))
    return specs


def _worker_loop(gpu_id: int, task_queue, results_queue, datasets_dir: str):
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from runner.experiment import run_one_experiment

    if gpu_id >= 0 and torch.cuda.is_available():
        device_str = "cuda:0"
    else:
        device_str = "cpu"

    while True:
        item = task_queue.get()
        if item is None:
            break
        spec = item
        spec.device = device_str
        spec.datasets_dir = datasets_dir
        t0 = time.time()
        try:
            result = run_one_experiment(spec)
            status = result.get("status", "unknown")
        except Exception as e:
            status = "crashed"
            result = {"status": status, "error": repr(e)}
            print(f"[worker-error] gpu{gpu_id} {spec.tag} {type(e).__name__}: {e}", flush=True)
        dt = time.time() - t0
        print(f"[worker-done] gpu{gpu_id} {spec.tag} {status} dt={dt:.1f}s", flush=True)
        results_queue.put({
            "model": spec.model, "dataset": spec.dataset,
            "split": spec.split_tag, "seed": spec.seed,
            "tag": spec.tag, "wall_seconds": dt, "status": status,
            "gpu_id": gpu_id,
            "test_rel_l2": result.get("inference_test_rel_l2"),
            "test_rel_mse": result.get("inference_test_rel_mse"),
            "best_val_rel_l2": result.get("best_val_rel_l2"),
            "best_val_loss": result.get("best_val_loss"),
            "params": result.get("params"),
        })


def _print_results(results: list[dict]) -> None:
    print("\n" + "=" * 96)
    print(f"{'model':<14}{'dataset':<14}{'split':<10}{'status':<10}"
          f"{'test_rel_l2':<18}{'best_val_rel_l2'}")
    print("-" * 96)
    rows_sorted = sorted(
        results, key=lambda r: (r["model"], r["dataset"], r["split"])
    )
    for r in rows_sorted:
        tv = r.get("test_rel_l2")
        bv = r.get("best_val_rel_l2")
        tv_s = f"{tv:.4e}" if isinstance(tv, (int, float)) else "n/a"
        bv_s = f"{bv:.4e}" if isinstance(bv, (int, float)) else "n/a"
        print(f"{r['model']:<14}{r['dataset']:<14}{r['split']:<10}"
              f"{r['status']:<10}{tv_s:<18}{bv_s}")
    print("=" * 96)


def main():
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

    specs = build_specs(args.datasets_dir)
    n_dataset_splits = sum(len(v) for v in SPLITS.values())
    print(f"[launcher] {len(specs)} experiments scheduled "
          f"({len(MODELS)} models x {n_dataset_splits} (dataset, split) combos)")

    results: list[dict] = []

    if n_gpus == 0:
        from runner.experiment import run_one_experiment
        for spec in specs:
            spec.device = "cpu"
            t0 = time.time()
            r = run_one_experiment(spec)
            dt = time.time() - t0
            print(f"[cpu] {spec.tag} {r.get('status')} {dt:.1f}s", flush=True)
            results.append({
                "model": spec.model, "dataset": spec.dataset,
                "split": spec.split_tag, "seed": spec.seed,
                "tag": spec.tag, "wall_seconds": dt,
                "status": r.get("status", "unknown"),
                "gpu_id": -1,
                "test_rel_l2": r.get("inference_test_rel_l2"),
                "test_rel_mse": r.get("inference_test_rel_mse"),
                "best_val_rel_l2": r.get("best_val_rel_l2"),
                "best_val_loss": r.get("best_val_loss"),
                "params": r.get("params"),
            })
    else:
        n_workers = n_gpus * args.per_gpu
        print(f"[launcher] spawning {n_workers} workers ({args.per_gpu} per GPU)")
        ctx = mp.get_context("spawn")
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        for s in specs:
            task_q.put(s)
        for _ in range(n_workers):
            task_q.put(None)

        workers = []
        for w_idx in range(n_workers):
            gpu = w_idx % n_gpus
            p = ctx.Process(
                target=_worker_loop,
                args=(gpu, task_q, result_q, args.datasets_dir),
            )
            p.start()
            workers.append(p)

        n_done = 0
        n_failed = 0
        total = len(specs)
        while n_done < total:
            r = result_q.get()
            n_done += 1
            if r["status"] != "ok":
                n_failed += 1
            print(f"[{n_done}/{total}] gpu{r['gpu_id']} {r['tag']} "
                  f"{r['status']} {r['wall_seconds']:.1f}s", flush=True)
            results.append(r)

        for p in workers:
            p.join()

        print(f"[launcher] done. failures: {n_failed}")

    _print_results(results)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
