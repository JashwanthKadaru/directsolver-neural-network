# Fast Direct Solver Neural Network (FDSNet)

Reference implementation and experiment launchers for the Fast Direct Solver Neural Network (FDSNet) — a neural-network architecture for learning (i) HODLR-structured operators that arise in integral-equation formulations of linear elliptic PDEs, and (ii) nonlinear solution operators associated with PDEs.

The repository contains implementation of the FDSNet model, four baseline operator-learning models for ablation (FNO, MNN-H, DeepONet, MLP), the shared training harness, and launchers for the main ablation, the FDSNet K- and r-sweep, the Fredholm rank sweep, and the NLSE-1D generalization study across the nonlinearity parameter $\beta$. The results corresponding to the same experiments were reported in the accompanying paper.

## Authors

- Jashwanth Reddy Kadaru (IIIT Bangalore)
- Vaishnavi Gujjula (IIIT Bangalore)

## Requirements (libraries & modules)

- Python 3.10+
- PyTorch 2.4+
- NumPy 1.26+
- `kagglehub`

## How to run training & evaluation for experiments

```bash
# 1. Setup environment and install packages
bash setup.sh
source .venv/bin/activate

# 2. Setup kaggle username and api key
export KAGGLE_USERNAME            #
export KAGGLE_KEY                 #

# 3. Run download scripts to fetch data from kaggle to local
python src/scripts/download_datasets.py
python src/scripts/download_fredholm.py

# 4. Run scripts
python src/runner/run_all.py             --datasets-dir datasets
python src/runner/run_fdsnet_sweeps.py     --datasets-dir datasets
python src/runner/run_fredholm_sweeps.py --datasets-dir datasets
python src/runner/run_nlse_betas.py      --betas-dir datasets/nlse_1d_betas
```

## Repository layout

```
src/
  losses/loss.py             relative L2 loss and metric
  models/
    fdsnet_model.py          Fast Direct Solver Neural Network 
    fno_model.py             Fourier Neural Operator
    mnnh_model.py            H-matrix-based multiscale network
    deeponet_model.py        DeepONet with MLP or CNN branch
    mlp_model.py             plain MLP control
  runner/
    experiment.py            single-experiment trainer with early stopping
    builders.py              model factory and per-dataset FDSNet configs
    adapters.py              per-model I/O adapters
    data.py                  .npz loaders and seeded train/test splits
    kdtree.py                HODLR k-d-tree permutation for 2D inputs
    run_all.py               main ablation launcher
    run_fdsnet_sweeps.py       FDSNet K-sweep then r-sweep
    run_fredholm_sweeps.py   Fredholm rank sweep
    run_nlse_betas.py        NLSE-1D beta-generalization study
  scripts/
    download_datasets.py
    download_fredholm.py
setup.sh
requirements.txt
LICENSE
```

All four launchers share the same CLI surface:

| Flag | Default | What it does |
|---|---|---|
| `--datasets-dir DIR` | required (except betas) | Path to the canonical `.npz` dir. |
| `--betas-dir DIR` | required for `run_nlse_betas` | Path to the β `.npz` files. |
| `--per-gpu N` | 4 (2 for betas) | Worker processes per GPU. |

## Datasets

All datasets are publicly hosted on Kaggle Hub and fetched by the download
scripts:

- `nlse_1d_dataset.npz` — 1D nonlinear Schrödinger equation.
- `burgers_1d_dataset.npz` — 1D Burgers' equation.
- `nlse_2d_dataset.npz` — 2D nonlinear Schrödinger equation.
- `darcys_flow_2d_dataset.npz` — 2D Darcy flow.
- `Fredholm_IE_dataset_N=*.txt.npz` — 2D Fredholm integral equation, supplied as length-N 1D vectors at three grid sizes (N=1600, 6400, 14400).
- `nlse_1d_demo_2K_beta=*.npz` — NLSE-1D at 101 values of the nonlinearity
  parameter $\beta$, 2K samples each.

## Citation

If you build on this reference implementation in academic work, please
cite the paper:

```bibtex
@misc{***2026fdsnet,
  title        = {A fast direct solver based neural network for solving PDEs},
  author       = {Kadaru, Jashwanth Reddy and Gujjula, Vaishnavi},
  year         = {2026},
  howpublished = {Preprint},
  url          = {https://github.com/JashwanthKadaru/directsolver-neural-network},
}
```

## License

MIT — see [LICENSE](LICENSE).
