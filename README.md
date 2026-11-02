# Does Bangalore Traffic Explain House Prices?
### Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Overview

This repository contains the complete implementation, datasets, Jupyter notebooks, model checkpoints, and server logging subsystem for the research project:
> **"Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads"**

Urban real estate valuation models overwhelmingly rely on intrinsic listing attributes (square footage, BHK, bathrooms). However, in hyper-growth tech hubs like Bangalore, housing prices are fundamentally constrained by spatial frictions, commuter bottlenecks, and transit accessibility.

This project implements a multi-modal spatial-economic deep learning pipeline that investigates whether explicit road-network congestion features and spatial graph topologies improve price predictions and out-of-locality generalization.

---

## 2. Theoretical Grounding & Nature (2025) Paper Findings

Our spatial modeling pipeline incorporates the mathematical framework and findings from:
> **Ravichandran et al.** (2025). *"Gaussian dual adjacency graph based spatial correlated and temporal time dependent traffic prediction in Bangalore City"*. **Nature Scientific Reports**, 15:41208. [DOI: 10.1038/s41598-025-25075-4](s41598-025-25075-4.pdf).

### Key Formulations Incorporated:
1. **Algorithm 1 — Local-Global Invariant IQR Preprocessing (Eq. 5 & 8)**:
   - Outlier filtering using $Q_3 - Q_1$ evaluated globally and across local sub-environments to prevent overfitting and retain clean distributions.
2. **Algorithm 2 — Gaussian Dual Adjacency Matrix ($DAM$) (Eq. 9 & 10)**:
   - Combines the physical road network adjacency $A_{\text{static}}$ with a dynamic spatial feature correlation matrix $A_{\text{dynamic}}$ using a Gaussian RBF kernel:
     $$A_{\text{dynamic}}(i, j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
     $$DAM = A_{\text{static}} + A_{\text{dynamic}}$$
     $$\widetilde{DAM} = D^{-1/2}(DAM) \cdot DAM \cdot D^{-1/2}(DAM)$$
   - Dynamically bridges geographically distant Bangalore bottlenecks (e.g. Silk Board, Marathahalli, Tin Factory) that physical distance alone fails to capture.

---

## 3. Repository Architecture

```text
├── data/
│   ├── Bengaluru_House_Data.csv                   # Raw Kaggle Bangalore listings (~13.3k rows)
│   ├── bangalore_tech_hubs.csv                    # Centroids for the 5 major employment corridors
│   ├── bangalore_metro_stations.csv               # 64 operational BMRCL Namma Metro stations
│   ├── bangalore_localities_master.csv            # 152 canonical Bangalore localities
│   ├── bangalore_locality_traffic_matrix.csv      # Peak vs off-peak travel times & congestion ratios
│   ├── bangalore_housing_cleaned_tabular.csv      # Cleaned listing attributes (BHK, sqft, price)
│   ├── bangalore_housing_spatial_commute_dataset.csv # Unified modeling dataset (10,008 rows, 57 cols)
│   ├── bangalore_locality_graph_edges.csv         # Physical road network edges (872 directed edges)
│   ├── bangalore_locality_graph_nodes.csv         # Inductive non-leaked GNN node features
│   └── bangalore_graph_pyg.pt                     # PyTorch Geometric tensor bundle
├── notebooks/
│   ├── 01_bangalore_gdag_preprocessing_and_eda.ipynb # Data ingestion, IQR cleaning & spatial EDA
│   ├── 02_gaussian_dual_adjacency_graph_gdag.ipynb   # Gaussian Dual Adjacency Matrix (DAM) & PyG
│   ├── 03_baseline_models_exp1_and_exp2.ipynb        # Exp 1 Tabular vs Exp 2 Commute-Aware models
│   └── 04_gdag_gnn_and_bayesian_uncertainty_exp3_exp4.ipynb # Exp 3 GNN & Exp 4 Uncertainty Quantification
├── src/
│   ├── logger.py                                  # Process telemetry and rotating server logger
│   ├── gdag_preprocessing.py                      # Algorithm 1: Local-Global Invariant IQR & Min-Max
│   ├── gdag_spatial_graph.py                      # Algorithm 2: Gaussian Dual Adjacency Graph (GDAG)
│   ├── models.py                                  # Architectures (XGBoost, MLP, GDAG-GNN, UQ Heads)
│   └── evaluation.py                              # Point metrics (RMSE, MAPE) & UQ metrics (NLL, PICP, CRPS)
├── scripts/
│   ├── build_bangalore_dataset.py                 # Reproducible end-to-end dataset builder
│   ├── verify_dataset.py                          # Data integrity & zero leakage validation
│   ├── generate_notebooks.py                      # Programmatic Jupyter Notebook generator
│   └── run_all_experiments.py                     # Master experiment orchestrator with server logs
├── logs/
│   ├── server.log                                 # Persistent server lifecycle & telemetry log
│   └── training.log                               # Epoch losses, fold evaluations & metrics
├── models/
│   ├── benchmark_results.csv                      # Comparative metrics table for Exp 1-4
│   └── gdag_hybrid_housing_model.pt               # Trained PyTorch checkpoint
├── DATASET_DOCUMENTATION.md                       # Comprehensive data sheet & schema definitions
└── README.md
```

---

## 4. Empirical Benchmark Results

| Model | Evaluation Split | RMSE (log-price) | MAPE (%) | $R^2$ Score | 90% PICP Coverage | CRPS |
|---|---|---|---|---|---|---|
| **Exp 1: Baseline XGBoost** (Tabular only) | Random Test | 0.2658 | 5.09% | 0.6735 | — | — |
| **Exp 1: Baseline MLP** (Tabular only) | Random Test | 0.3013 | 5.67% | 0.5804 | — | — |
| **Exp 2: Commute XGBoost** (+ Commute) | Random Test | **0.2240** | **4.20%** | **0.7681** | — | — |
| **Exp 3: GDAG-GNN Hybrid** (Dual Graph) | Random Test | 0.2335 | 4.49% | 0.7479 | — | — |
| **Exp 4: MC Dropout Head** (UQ) | Random Test | 0.2336 | 4.49% | 0.7477 | 30.56% | 0.1567 |
| **Exp 4: Conformal Prediction** ($\alpha=0.10$) | Random Test | 0.2335 | 4.49% | 0.7479 | **92.27%** | **0.1306** |
| **Exp 1 (Tabular Only)** | **5-Fold LLO** (Unseen) | 0.2717 ± 0.024 | 5.24% | 0.6580 | — | — |
| **Exp 2 (Tabular + Commute)** | **5-Fold LLO** (Unseen) | 0.2651 ± 0.018 | 4.98% | 0.6790 | — | — |
| **Exp 3 (GDAG-GNN Hybrid)** | **5-Fold LLO** (Unseen) | 0.2757 ± 0.010 | 5.12% | 0.6620 | — | — |
| **Exp 4 (Conformal Prediction)** | **5-Fold LLO** (Unseen) | 0.2757 ± 0.010 | 5.12% | 0.6620 | **86.50%** | 0.1640 |

---

## 5. Quickstart & Execution

### Clone & Setup
```bash
git clone https://github.com/hihihi-svg/Does-Bangalore-Traffic-Explain-House-Prices-.git
cd Does-Bangalore-Traffic-Explain-House-Prices-
pip install -r requirements.txt # or: pip install torch pandas numpy xgboost rapidfuzz psutil matplotlib seaborn
```

### Run All Experiments & Stream Server Logs
```bash
python scripts/run_all_experiments.py
```

### Inspect Live Logs
```bash
# Windows PowerShell
Get-Content logs/training.log -Wait -Tail 30

# Linux / macOS
tail -f logs/training.log
```

---

## 6. Citation & Reference
If using this repository, please cite:
- **Project Proposal:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*
- **Nature Scientific Reports (2025):** Ravichandran, S.K., Shieh, CS., Horng, MF. et al. Gaussian dual adjacency graph based spatial correlated and temporal time dependent traffic prediction in Bangalore City. *Sci Rep* 15, 41208 (2025). https://doi.org/10.1038/s41598-025-25075-4
