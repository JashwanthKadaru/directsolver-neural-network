"""Generalization study for FDSNet on NLSE-1D across nonlinearity strengths beta.

Trains FDSNet configurations on four train/test splits of the 101 beta values
and reports per-protocol metrics. Per-beta and global test errors are
logged to the console.

Protocols:

* P1: Dense interpolation     - train every other beta, test held-out half.
* P2: Sparse interpolation    - train every fifth beta, test the rest.
* P3: One-sided extrapolation - train low band, test high band.
* P4: Two-sided extrapolation - train middle band, test both tails.

Usage:
    python src/runner/run_nlse_betas.py --betas-dir datasets/nlse_1d_betas
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


FDSNET_CFGS = [
    {"N": 320, "L": 6, "r": 12, "K": 5},
]

SEED = 42
BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 2000
PATIENCE = 200
MIN_EPOCHS = 50

N_TRAIN_CAP = 40_000
N_TEST_CAP = None
N_VAL_CAP = 8_000


def _build_protocols(file_paths: list[str]) -> dict[str, dict]:
    """Returns a dict ``protocol_name -> {"train_idx", "test_idx"}``."""
    n = len(file_paths)
    assert n == 101, f"Expected 101 beta files, found {n}"

    p1_train = list(range(0, n, 2))
    p1_test  = list(range(1, n, 2))

    p2_train = list(range(0, n, 5))
    p2_test  = [i for i in range(n) if i not in set(p2_train)]

    p3_train = list(range(0, 61))
    p3_test  = list(range(61, n))

    p4_train = list(range(30, 71))
    p4_test  = list(range(0, 30)) + list(range(71, n))

    return {
        "P1_dense_interp":     {"train_idx": p1_train, "test_idx": p1_test},
        "P2_sparse_interp":    {"train_idx": p2_train, "test_idx": p2_test},
        "P3_extrap_high":      {"train_idx": p3_train, "test_idx": p3_test},
        "P4_extrap_two_sided": {"train_idx": p4_train, "test_idx": p4_test},
    }


def _beta_of(path: str) -> float:
    return float(re.search(r"beta=([0-9.]+)\.npz", path).group(1))


def _load_beta_files(file_paths: list[str], indices: list[int],
                     n_cap: int | None, rng: np.random.Generator
                     ) -> tuple[np.ndarray, np.ndarray]:
    V_list, uG_list = [], []
    for i in indices:
        with np.load(file_paths[i]) as d:
            V_list.append(d["V"])
            uG_list.append(d["uG"])
    V = np.concatenate(V_list, axis=0)
    uG = np.concatenate(uG_list, axis=0)
    if V.ndim == 3 and V.shape[-1] == 1:
        V = V[..., 0]
    if uG.ndim == 3 and uG.shape[-1] == 1:
        uG = uG[..., 0]
    perm = rng.permutation(V.shape[0])
    V = V[perm]
    uG = uG[perm]
    if n_cap is not None:
        V = V[:n_cap]
        uG = uG[:n_cap]
    return V.astype(np.float32), uG.astype(np.float32)


def _train_fdsnet_one_protocol(spec: "ProtocolSpec") -> dict:
    """Trains FDSNet on one (protocol, config) combination and returns
    metrics. All progress is logged to the console; nothing is written to
    disk."""
    import torch
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "src" / "models"))
    sys.path.insert(0, str(ROOT / "src" / "losses"))

    from loss import (relative_mean_squared_error,
                      relative_l2_non_meansquared_error)
    from fdsnet_model import FDSNet_Linear_1D
    from runner.adapters import FDSNetAdapter

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    rng = np.random.default_rng(spec.seed)

    device = torch.device(spec.device)
    _gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
    cfg = spec.cfg
    _stage = (f"{_gpu}:fdsnet:nlse_1d:{spec.protocol}:"
              f"N={cfg['N']}:L={cfg['L']}:r={cfg['r']}:K={cfg['K']}")
    adapter = None

    try:
        V_tr, uG_tr = _load_beta_files(spec.file_paths, spec.train_idx,
                                       n_cap=N_TRAIN_CAP, rng=rng)
        V_te, uG_te = _load_beta_files(spec.file_paths, spec.test_idx,
                                       n_cap=N_TEST_CAP, rng=rng)
        Xtr = torch.from_numpy(V_tr).to(device, non_blocking=True)
        Ytr = torch.from_numpy(uG_tr).to(device, non_blocking=True)
        Xte = torch.from_numpy(V_te).to(device, non_blocking=True)
        Yte = torch.from_numpy(uG_te).to(device, non_blocking=True)

        n_test_total = Xte.size(0)
        n_val = min(N_VAL_CAP, n_test_total)
        val_idx_np = rng.choice(n_test_total, size=n_val, replace=False)
        val_idx = torch.from_numpy(val_idx_np).to(device, dtype=torch.long)
        Xval = Xte.index_select(0, val_idx).contiguous()
        Yval = Yte.index_select(0, val_idx).contiguous()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        n_train = Xtr.size(0)
        bs = spec.batch_size

        model = FDSNet_Linear_1D(
            N=cfg["N"], L=cfg["L"], r=cfg["r"], K=cfg["K"],
        )
        adapter = FDSNetAdapter(model, dim=1).to(device)
        n_params = sum(p.numel() for p in model.parameters())

        optimizer = torch.optim.NAdam(adapter.parameters(), lr=spec.lr)
        criterion = relative_mean_squared_error
        metric = relative_l2_non_meansquared_error

        def _flat(t): return t.reshape(t.size(0), -1)

        best_val_rl2 = float("inf")
        best_epoch = -1
        best_train_loss = best_val_loss = float("nan")
        best_train_rl2 = best_val_rl2_record = float("nan")
        best_state = None
        epochs_since_improve = 0
        total_epochs = 0

        wall_t0 = time.perf_counter()
        for epoch in range(1, spec.max_epochs + 1):
            adapter.train()
            perm = torch.randperm(n_train, device=device)
            tr_loss_sum = tr_rl2_sum = 0.0
            for i in range(0, n_train, bs):
                idx = perm[i:i + bs]
                x = Xtr.index_select(0, idx)
                y = Ytr.index_select(0, idx)
                optimizer.zero_grad(set_to_none=True)
                pred = adapter(x)
                loss = criterion(_flat(pred), _flat(y))
                loss.backward()
                optimizer.step()
                b = x.size(0)
                tr_loss_sum += loss.item() * b
                with torch.no_grad():
                    tr_rl2_sum += metric(_flat(pred.detach()), _flat(y)).item() * b
            tr_loss = tr_loss_sum / n_train
            tr_rl2 = tr_rl2_sum / n_train

            adapter.eval()
            va_loss_sum = va_rl2_sum = 0.0
            n_val_loop = Xval.size(0)
            with torch.no_grad():
                for i in range(0, n_val_loop, bs):
                    xb = Xval[i:i + bs]
                    yb = Yval[i:i + bs]
                    pred = adapter(xb)
                    b = xb.size(0)
                    va_loss_sum += criterion(_flat(pred), _flat(yb)).item() * b
                    va_rl2_sum += metric(_flat(pred), _flat(yb)).item() * b
            va_loss = va_loss_sum / n_val_loop
            va_rl2 = va_rl2_sum / n_val_loop
            total_epochs = epoch

            if epoch == 1 or epoch % 10 == 0:
                print(f"[{_stage}] epoch={epoch} train_loss={tr_loss:.6e} "
                      f"val_loss={va_loss:.6e} elapsed_s={time.perf_counter() - wall_t0:.1f}",
                      flush=True)

            if va_rl2 < best_val_rl2:
                best_val_rl2 = va_rl2
                best_epoch = epoch
                best_train_loss, best_val_loss = tr_loss, va_loss
                best_train_rl2, best_val_rl2_record = tr_rl2, va_rl2
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
            if epoch >= MIN_EPOCHS and epochs_since_improve >= spec.patience:
                break

        wall_seconds = time.perf_counter() - wall_t0
        print(f"[train-done] {_stage} epochs={total_epochs} "
              f"best_epoch={best_epoch} train_s={wall_seconds:.1f}", flush=True)

        if best_state is not None:
            adapter.load_state_dict(best_state)

        adapter.eval()
        per_beta = {}
        global_rmse_sum = 0.0
        global_rl2_sum = 0.0
        global_n = 0
        with torch.no_grad():
            for i in spec.test_idx:
                with np.load(spec.file_paths[i]) as d:
                    V = d["V"]
                    uG = d["uG"]
                if V.ndim == 3 and V.shape[-1] == 1:
                    V = V[..., 0]
                if uG.ndim == 3 and uG.shape[-1] == 1:
                    uG = uG[..., 0]
                Xb = torch.from_numpy(V.astype(np.float32)).to(device)
                Yb = torch.from_numpy(uG.astype(np.float32)).to(device)
                nb = Xb.size(0)
                rmse_sum = 0.0
                rl2_sum = 0.0
                for j in range(0, nb, bs):
                    xj = Xb[j:j + bs]
                    yj = Yb[j:j + bs]
                    pj = adapter(xj)
                    bj = xj.size(0)
                    rmse_sum += criterion(_flat(pj), _flat(yj)).item() * bj
                    rl2_sum  += metric(_flat(pj),  _flat(yj)).item() * bj
                rmse_b = rmse_sum / nb
                rl2_b  = rl2_sum / nb
                beta_key = f"{_beta_of(spec.file_paths[i]):.4f}"
                per_beta[beta_key] = {"rel_mse": rmse_b, "rel_l2": rl2_b, "n": int(nb)}
                global_rmse_sum += rmse_sum
                global_rl2_sum  += rl2_sum
                global_n        += nb

        test_rel_mse_global = global_rmse_sum / max(global_n, 1)
        test_rel_l2_global  = global_rl2_sum  / max(global_n, 1)

        print(f"[per-beta] {_stage}", flush=True)
        for beta_key in sorted(per_beta.keys()):
            v = per_beta[beta_key]
            print(f"  beta={beta_key} rel_mse={v['rel_mse']:.6e} "
                  f"rel_l2={v['rel_l2']:.6e} n={v['n']}", flush=True)
        print(f"[global-test] {_stage} rel_mse={test_rel_mse_global:.6e} "
              f"rel_l2={test_rel_l2_global:.6e} n={global_n}", flush=True)

        return {
            "status": "ok",
            "protocol": spec.protocol,
            "model": "fdsnet",
            "dataset": "nlse_1d",
            "config": cfg,
            "params": int(n_params),
            "n_train": int(Xtr.size(0)),
            "n_test": int(Xte.size(0)),
            "n_train_betas": len(spec.train_idx),
            "n_test_betas": len(spec.test_idx),
            "total_epochs_trained": total_epochs,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_train_rel_l2": best_train_rl2,
            "best_val_rel_l2": best_val_rl2_record,
            "test_rel_mse_global": test_rel_mse_global,
            "test_rel_l2_global": test_rel_l2_global,
            "test_n_samples_global": int(global_n),
            "wall_time_seconds": wall_seconds,
        }

    except Exception as e:
        tb = traceback.format_exc()
        print(f"[experiment-error] {_stage} {type(e).__name__}: {e}", flush=True)
        print(tb, flush=True)
        return {"status": "failed", "protocol": spec.protocol,
                "config": cfg, "error": repr(e)}
    finally:
        try:
            del adapter
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()


@dataclass
class ProtocolSpec:
    protocol: str
    cfg: dict
    file_paths: list[str]
    train_idx: list[int]
    test_idx: list[int]
    device: str
    seed: int = SEED
    batch_size: int = BATCH_SIZE
    lr: float = LR
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE

    @property
    def tag(self) -> str:
        c = self.cfg
        return f"nlse_1d_betas__fdsnet__r{c['r']}_K{c['K']}__{self.protocol}"


def _worker_loop(gpu_id: int, task_q, result_q):
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    device_str = "cuda:0" if (gpu_id >= 0 and torch.cuda.is_available()) else "cpu"

    while True:
        item = task_q.get()
        if item is None:
            break
        spec: ProtocolSpec = item
        spec.device = device_str
        t0 = time.time()
        try:
            res = _train_fdsnet_one_protocol(spec)
            status = res.get("status", "unknown")
        except Exception as e:
            status = "crashed"
            res = {"status": status, "error": repr(e)}
            print(f"[worker-error] gpu{gpu_id} {spec.tag} {type(e).__name__}: {e}", flush=True)
        dt = time.time() - t0
        print(f"[worker-done] gpu{gpu_id} {spec.tag} {status} dt={dt:.1f}s", flush=True)
        result_q.put({
            "tag": spec.tag, "wall_seconds": dt, "status": status,
            "gpu_id": gpu_id,
            "protocol": spec.protocol, "cfg": spec.cfg,
            "best_val_rel_l2": res.get("best_val_rel_l2"),
            "test_rel_l2_global": res.get("test_rel_l2_global"),
            "test_rel_mse_global": res.get("test_rel_mse_global"),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas-dir", required=True)
    ap.add_argument("--per-gpu", type=int, default=2)
    args = ap.parse_args()
    if args.per_gpu < 1:
        ap.error("--per-gpu must be >= 1")

    file_paths = sorted(
        glob.glob(os.path.join(args.betas_dir, "nlse_1d_demo_2K_beta=*.npz")),
        key=_beta_of,
    )
    if not file_paths:
        ap.error(f"No beta files found under {args.betas_dir}")
    print(f"[launcher] found {len(file_paths)} beta files; "
          f"beta range [{_beta_of(file_paths[0]):.2f}, {_beta_of(file_paths[-1]):.2f}]")

    protocols = _build_protocols(file_paths)
    for name, p in protocols.items():
        print(f"[launcher] {name}: train_betas={len(p['train_idx'])} "
              f"test_betas={len(p['test_idx'])}")

    specs: list[ProtocolSpec] = []
    for cfg in FDSNET_CFGS:
        for name, p in protocols.items():
            specs.append(ProtocolSpec(
                protocol=name, cfg=cfg,
                file_paths=file_paths,
                train_idx=p["train_idx"],
                test_idx=p["test_idx"],
                device="cpu", seed=SEED,
            ))
    print(f"[launcher] configs: {FDSNET_CFGS}")
    print(f"[launcher] seed: {SEED}")

    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[launcher] visible GPUs: {n_gpus}")
    print(f"[launcher] {len(specs)} runs scheduled")

    results: list[dict] = []
    if n_gpus == 0:
        for spec in specs:
            spec.device = "cpu"
            t0 = time.time()
            r = _train_fdsnet_one_protocol(spec)
            dt = time.time() - t0
            print(f"[cpu] {spec.tag} {r.get('status')} {dt:.1f}s")
            results.append({
                "tag": spec.tag, "wall_seconds": dt,
                "status": r.get("status", "unknown"),
                "gpu_id": -1,
                "protocol": spec.protocol, "cfg": spec.cfg,
                "best_val_rel_l2": r.get("best_val_rel_l2"),
                "test_rel_l2_global": r.get("test_rel_l2_global"),
                "test_rel_mse_global": r.get("test_rel_mse_global"),
            })
    else:
        n_workers = min(len(specs), n_gpus * args.per_gpu)
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
            p = ctx.Process(target=_worker_loop, args=(gpu, task_q, result_q))
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
                  f"{r['status']} {r['wall_seconds']:.1f}s")
            results.append(r)
        for p in workers:
            p.join()
        print(f"[launcher] done. failures: {n_failed}")

    print("\n[summary] per-(r, K, protocol):")
    print(f"  {'r':<5}{'K':<5}{'protocol':<26}"
          f"{'test_rel_l2':<18}{'best_val_rel_l2'}")
    print(f"  " + "-" * 70)
    rows_ok = [r for r in results if r["status"] == "ok"]
    rows_ok.sort(key=lambda r: (r["cfg"]["r"], r["cfg"]["K"], r["protocol"]))
    for r in rows_ok:
        c = r["cfg"]
        tv = r.get("test_rel_l2_global")
        bv = r.get("best_val_rel_l2")
        tv_s = f"{tv:.4e}" if isinstance(tv, (int, float)) else "n/a"
        bv_s = f"{bv:.4e}" if isinstance(bv, (int, float)) else "n/a"
        print(f"  {c['r']:<5}{c['K']:<5}{r['protocol']:<26}{tv_s:<18}{bv_s}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
