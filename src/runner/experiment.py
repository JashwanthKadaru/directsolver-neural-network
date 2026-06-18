"""Single-experiment trainer.

``run_one_experiment(spec)`` trains one ``(model, dataset, split, seed)``
configuration to early stopping, then times pure-forward inference. All
results are returned as a dict and logged to the console; nothing is
written to disk.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "losses"))

from loss import (                                                     # noqa: E402
    relative_mean_squared_error, relative_l2_non_meansquared_error,
)
from runner.builders import build, DATASETS                            # noqa: E402
from runner.data import load_split                                     # noqa: E402


MAX_EPOCHS = 2000
PATIENCE = 200
MIN_EPOCHS = 50
WARMUP_BATCHES = 5
TIMING_BATCHES = 100


@dataclass
class ExperimentSpec:
    model: str
    dataset: str
    n_train: int
    n_test: int
    batch_size: int
    seed: int
    datasets_dir: str
    device: str
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    lr: float = 1e-3

    @property
    def split_tag(self) -> str:
        return f"{self.n_train//1000}k_{self.n_test//1000}k"

    @property
    def tag(self) -> str:
        return f"{self.dataset}__{self.model}__{self.split_tag}__seed{self.seed}"


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _flatten_batch(t: torch.Tensor) -> torch.Tensor:
    return t.reshape(t.size(0), -1)


def _eval_on_gpu(adapter, X, Y, batch_size, criterion, metric):
    adapter.eval()
    n = X.size(0)
    loss_sum = 0.0
    metric_sum = 0.0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = Y[i:i+batch_size]
            pred = adapter(xb)
            pred_f = _flatten_batch(pred)
            y_f = _flatten_batch(yb)
            b = xb.size(0)
            loss_sum += criterion(pred_f, y_f).item() * b
            metric_sum += metric(pred_f, y_f).item() * b
    return loss_sum / n, metric_sum / n


def _time_inference_full(adapter, X, Y, batch_size, device, criterion, metric):
    """Timed pure-forward pass over device-resident tensors, with warmup.

    The timed region is capped at ``TIMING_BATCHES`` batches so per-sample
    timing is well-conditioned without stalling on very large splits.
    """
    adapter.eval()
    is_cuda = device.type == "cuda"
    n = X.size(0)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n_batches_total = (n + batch_size - 1) // batch_size
    n_timed = min(TIMING_BATCHES, n_batches_total)

    with torch.no_grad():
        for i in range(min(WARMUP_BATCHES, max(1, n_batches_total))):
            s = i * batch_size
            adapter(X[s:s+batch_size])
        if is_cuda:
            torch.cuda.synchronize(device)

    rel_mse_acc = torch.zeros((), device=device)
    rel_l2_acc = torch.zeros((), device=device)
    n_samples = 0
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(n_timed):
            s = i * batch_size
            xb = X[s:s+batch_size]
            yb = Y[s:s+batch_size]
            pred = adapter(xb)
            pred_f = _flatten_batch(pred)
            y_f = _flatten_batch(yb)
            b = xb.size(0)
            rel_mse_acc = rel_mse_acc + criterion(pred_f, y_f) * b
            rel_l2_acc = rel_l2_acc + metric(pred_f, y_f) * b
            n_samples += b
    if is_cuda:
        torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - t0
    ms_per_sample = (total_seconds / n_samples) * 1000.0
    return total_seconds, ms_per_sample, (rel_mse_acc / n_samples).item(), (rel_l2_acc / n_samples).item()


def run_one_experiment(spec: ExperimentSpec) -> dict:
    """Train one experiment to early stopping, then time inference.

    Returns a dict of metrics. All progress is logged to the console; no
    files are written.
    """
    _seed_everything(spec.seed)

    device = torch.device(spec.device)
    adapter = None

    try:
        Xtr, Ytr, Xte, Yte = load_split(
            datasets_dir=Path(spec.datasets_dir),
            dataset_name=spec.dataset,
            n_train=spec.n_train,
            n_test=spec.n_test,
            seed=spec.seed,
        )
        Xtr = Xtr.to(device, non_blocking=True)
        Ytr = Ytr.to(device, non_blocking=True)
        Xte = Xte.to(device, non_blocking=True)
        Yte = Yte.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        bs = spec.batch_size
        n_train = Xtr.size(0)

        adapter, n_params, model_cfg = build(spec.model, spec.dataset, device)
        optimizer = torch.optim.NAdam(adapter.parameters(), lr=spec.lr)
        criterion = relative_mean_squared_error
        metric = relative_l2_non_meansquared_error

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
            tr_loss_sum = 0.0
            tr_rl2_sum = 0.0
            for i in range(0, n_train, bs):
                idx = perm[i:i+bs]
                x = Xtr.index_select(0, idx)
                y = Ytr.index_select(0, idx)
                optimizer.zero_grad(set_to_none=True)
                pred = adapter(x)
                pred_f = _flatten_batch(pred)
                y_f = _flatten_batch(y)
                loss = criterion(pred_f, y_f)
                loss.backward()
                optimizer.step()
                b = x.size(0)
                tr_loss_sum += loss.item() * b
                with torch.no_grad():
                    tr_rl2_sum += metric(pred_f.detach(), y_f).item() * b
            tr_loss = tr_loss_sum / n_train
            tr_rl2 = tr_rl2_sum / n_train

            va_loss, va_rl2 = _eval_on_gpu(adapter, Xte, Yte, bs, criterion, metric)
            total_epochs = epoch

            if epoch == 1 or epoch % 10 == 0:
                _gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
                print(
                    f"[{_gpu}:{spec.model}:{spec.dataset}:{spec.split_tag}:seed={spec.seed}] "
                    f"epoch={epoch} train_loss={tr_loss:.6e} val_loss={va_loss:.6e} "
                    f"elapsed_s={time.perf_counter()-wall_t0:.1f}",
                    flush=True,
                )

            if va_rl2 < best_val_rl2:
                best_val_rl2 = va_rl2
                best_epoch = epoch
                best_train_loss = tr_loss
                best_val_loss = va_loss
                best_train_rl2 = tr_rl2
                best_val_rl2_record = va_rl2
                best_state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if epoch >= MIN_EPOCHS and epochs_since_improve >= spec.patience:
                break

        wall_seconds = time.perf_counter() - wall_t0
        _gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "cpu")
        _stage_tag = f"{_gpu}:{spec.model}:{spec.dataset}:{spec.split_tag}:seed={spec.seed}"
        print(
            f"[train-done] {_stage_tag} epochs={total_epochs} best_epoch={best_epoch} "
            f"train_s={wall_seconds:.1f}",
            flush=True,
        )

        if best_state is not None:
            adapter.load_state_dict(best_state)

        (infer_train_total_s, infer_train_ms,
         infer_train_rel_mse, infer_train_rel_l2) = _time_inference_full(
            adapter, Xtr, Ytr, bs, device, criterion, metric,
        )
        print(
            f"[infer-train] {_stage_tag} ms_per_sample={infer_train_ms:.3f} "
            f"rel_mse={infer_train_rel_mse:.6e} rel_l2={infer_train_rel_l2:.6e}",
            flush=True,
        )
        (infer_test_total_s, infer_test_ms,
         infer_test_rel_mse, infer_test_rel_l2) = _time_inference_full(
            adapter, Xte, Yte, bs, device, criterion, metric,
        )
        print(
            f"[infer-test]  {_stage_tag} ms_per_sample={infer_test_ms:.3f} "
            f"rel_mse={infer_test_rel_mse:.6e} rel_l2={infer_test_rel_l2:.6e}",
            flush=True,
        )

        metrics = {
            "status": "ok",
            "dataset": spec.dataset,
            "model": spec.model,
            "split": spec.split_tag,
            "config": model_cfg,
            "params": int(n_params),
            "total_epochs_trained": total_epochs,
            "best_epoch": best_epoch,
            "best_train_loss": best_train_loss,
            "best_val_loss": best_val_loss,
            "best_train_rel_l2": best_train_rl2,
            "best_val_rel_l2": best_val_rl2_record,
            "inference_train_total_seconds": infer_train_total_s,
            "inference_test_total_seconds": infer_test_total_s,
            "inference_train_per_sample_ms": infer_train_ms,
            "inference_test_per_sample_ms": infer_test_ms,
            "inference_train_rel_mse": infer_train_rel_mse,
            "inference_test_rel_mse": infer_test_rel_mse,
            "inference_train_rel_l2": infer_train_rel_l2,
            "inference_test_rel_l2": infer_test_rel_l2,
            "wall_time_seconds": wall_seconds,
            "train_size": spec.n_train,
            "test_size": spec.n_test,
            "batch_size": spec.batch_size,
            "max_epochs": spec.max_epochs,
            "patience": spec.patience,
            "device": str(device),
            "seed": spec.seed,
            "lr": spec.lr,
        }
        return metrics

    except Exception as e:
        err = {
            "status": "failed",
            "dataset": spec.dataset,
            "model": spec.model,
            "split": spec.split_tag,
            "device": spec.device,
            "seed": spec.seed,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }
        print(
            f"[experiment-error] {spec.tag} {type(e).__name__}: {e}",
            flush=True,
        )
        print(err["traceback"], flush=True)
        return err
    finally:
        try:
            del adapter
        except Exception:
            pass
        if device.type == "cuda":
            torch.cuda.empty_cache()
