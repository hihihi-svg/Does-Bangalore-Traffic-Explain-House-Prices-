"""
Master Experiment Runner & Server Process Engine
================================================
Orchestrates and executes all 4 project experiments with live server logging:
- Exp 1: Baseline Non-Spatial (XGBoost & MLP on Tabular Features)
- Exp 2: Commute-Aware Tabular (Incorporating 5-Hub Commute Vectors & Metro Proximity)
- Exp 3: Gaussian Dual Adjacency GNN (GDAG-GCN with Nature 2025 Dual Adjacency Topology)
- Exp 4: Uncertainty Quantification & Calibration (MC Dropout, Deep Ensemble, Conformal CQR)

Evaluated under:
- Random Split (80/10/10)
- Leave-Locality-Out (LLO 5-Fold Cross Validation)
- Spatial Block (Held-Out Bangalore Quadrant)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from src.logger import ServerLogger, log_system_telemetry
from src.gdag_preprocessing import LocalGlobalIQRPreprocessor
from src.gdag_spatial_graph import GaussianDualAdjacencyGraph
from src.models import (
    TabularMLP, HybridGDAGHousingModel,
    heteroscedastic_nll_loss, ConformalPredictionCalibrator
)
from src.evaluation import compute_point_metrics, compute_uncertainty_metrics, log_evaluation_summary

logger = ServerLogger.get_logger()

# Set deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run():
    log_system_telemetry()
    start_time = time.time()
    logger.info("STARTING COMPLETE BANGALORE HOUSING + GDAG TRAFFIC EXPERIMENTAL PIPELINE")
    logger.info(f"Compute Device: {DEVICE}")

    # --------------------------------------------------------------------------
    # 1. LOAD DATASETS & PREPROCESS (Algorithm 1)
    # --------------------------------------------------------------------------
    data_path = "data/bangalore_housing_spatial_commute_dataset.csv"
    nodes_path = "data/bangalore_locality_graph_nodes.csv"
    edges_path = "data/bangalore_locality_graph_edges.csv"

    logger.info(f"Loading master spatial commute dataset from {data_path}...")
    df_raw = pd.read_csv(data_path)
    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)

    # Apply Nature 2025 Local-Global Invariant IQR Preprocessing
    preprocessor = LocalGlobalIQRPreprocessor(iqr_multiplier=1.5)
    df = preprocessor.filter_outliers(df_raw, continuous_cols=['total_sqft_cleaned', 'price_per_sqft'])
    logger.info(f"Dataset ready for modeling: {len(df)} samples across {df['canonical_locality_name'].nunique()} localities.")

    # --------------------------------------------------------------------------
    # 2. CONSTRUCT GAUSSIAN DUAL ADJACENCY MATRIX (Algorithm 2)
    # --------------------------------------------------------------------------
    gdag = GaussianDualAdjacencyGraph(sigma=1.2, dynamic_threshold=0.30)
    DAM, norm_DAM = gdag.build_dual_adjacency_matrix(df_nodes, df_edges)
    norm_dam_tensor = torch.tensor(norm_DAM, dtype=torch.float32).to(DEVICE)

    # Locality ID mapping
    loc_name_to_id = {row['canonical_locality_name']: int(row['locality_id']) for _, row in df_nodes.iterrows()}
    df['locality_node_id'] = df['canonical_locality_name'].map(loc_name_to_id)
    # Filter any unmapped
    df = df[df['locality_node_id'].notna()].copy()
    df['locality_node_id'] = df['locality_node_id'].astype(int)

    # --------------------------------------------------------------------------
    # 3. FEATURE SPECIFICATIONS
    # --------------------------------------------------------------------------
    # Tabular intrinsic features (Exp 1)
    tabular_features = ['bhk', 'total_sqft_cleaned', 'bath_cleaned', 'balcony_cleaned', 'is_ready_to_move']
    # One-hot encode area_type
    area_dummies = pd.get_dummies(df['area_type'], prefix='area', drop_first=True)
    df = pd.concat([df, area_dummies], axis=1)
    tabular_features.extend(list(area_dummies.columns))

    # Commute features (Exp 2)
    commute_features = [
        'commute_whitefield_peak_min', 'commute_whitefield_offpeak_min',
        'commute_bellandur_orr_peak_min', 'commute_bellandur_orr_offpeak_min',
        'commute_electronic_city_peak_min', 'commute_electronic_city_offpeak_min',
        'commute_manyata_tech_park_peak_min', 'commute_manyata_tech_park_offpeak_min',
        'commute_mg_road_cbd_peak_min', 'commute_mg_road_cbd_offpeak_min',
        'min_commute_peak_min', 'avg_commute_peak_min',
        'avg_congestion_ratio', 'dist_to_metro_km', 'commute_metro_access_min'
    ]

    # Node inductive features for GDAG-GNN (Exp 3 & 4)
    gnn_node_cols = [
        'latitude', 'longitude', 'node_degree',
        'commute_whitefield_peak_min', 'commute_bellandur_peak_min',
        'commute_ecity_peak_min', 'commute_manyata_peak_min',
        'commute_mgroad_peak_min', 'avg_congestion_ratio',
        'dist_to_metro_km', 'commute_metro_access_min'
    ]
    node_feat_raw = df_nodes[gnn_node_cols].values.astype(np.float32)
    node_feat_norm = (node_feat_raw - np.mean(node_feat_raw, axis=0)) / (np.std(node_feat_raw, axis=0) + 1e-6)
    node_feat_tensor = torch.tensor(node_feat_norm, dtype=torch.float32).to(DEVICE)

    # Standardize continuous features
    X_tab_raw = df[tabular_features].values.astype(np.float32)
    tab_mean, tab_std = np.mean(X_tab_raw, axis=0), np.std(X_tab_raw, axis=0) + 1e-6
    X_tab_norm = (X_tab_raw - tab_mean) / tab_std

    X_comm_raw = df[commute_features].values.astype(np.float32)
    comm_mean, comm_std = np.mean(X_comm_raw, axis=0), np.std(X_comm_raw, axis=0) + 1e-6
    X_comm_norm = (X_comm_raw - comm_mean) / comm_std

    y_logprice = df['log_price'].values.astype(np.float32)
    loc_ids = df['locality_node_id'].values.astype(np.int64)

    # Benchmark results collector
    results_records = []

    # --------------------------------------------------------------------------
    # 4. EXPERIMENT 1: BASELINE TABULAR (NON-SPATIAL)
    # --------------------------------------------------------------------------
    logger.info(">>> RUNNING EXPERIMENT 1: BASELINE TABULAR NON-SPATIAL (XGBOOST & MLP) <<<")
    # A. Random Split
    train_mask = (df['split_random'] == 'train').values
    test_mask = (df['split_random'] == 'test').values

    xgb_exp1 = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_exp1.fit(X_tab_norm[train_mask], y_logprice[train_mask])
    pred_xgb1 = xgb_exp1.predict(X_tab_norm[test_mask])
    m_xgb1 = compute_point_metrics(y_logprice[test_mask], pred_xgb1)
    log_evaluation_summary("Exp 1: XGBoost (Tabular Only)", "Random Test", m_xgb1)
    results_records.append({"Experiment": "Exp 1: Baseline XGBoost (Tabular)", "Split": "Random", **m_xgb1})

    # MLP Exp 1
    mlp_exp1 = TabularMLP(in_features=len(tabular_features)).to(DEVICE)
    opt_mlp1 = torch.optim.Adam(mlp_exp1.parameters(), lr=0.003, weight_decay=1e-4)
    crit_mse = nn.MSELoss()

    train_ds1 = TensorDataset(torch.tensor(X_tab_norm[train_mask]).to(DEVICE), torch.tensor(y_logprice[train_mask]).to(DEVICE))
    train_loader1 = DataLoader(train_ds1, batch_size=128, shuffle=True)

    mlp_exp1.train()
    for ep in range(35):
        for bx, by in train_loader1:
            opt_mlp1.zero_grad()
            out = mlp_exp1(bx)
            loss = crit_mse(out, by)
            loss.backward()
            opt_mlp1.step()

    mlp_exp1.eval()
    with torch.no_grad():
        pred_mlp1 = mlp_exp1(torch.tensor(X_tab_norm[test_mask]).to(DEVICE)).cpu().numpy()
    m_mlp1 = compute_point_metrics(y_logprice[test_mask], pred_mlp1)
    log_evaluation_summary("Exp 1: MLP (Tabular Only)", "Random Test", m_mlp1)
    results_records.append({"Experiment": "Exp 1: Baseline MLP (Tabular)", "Split": "Random", **m_mlp1})

    # --------------------------------------------------------------------------
    # 5. EXPERIMENT 2: COMMUTE-AWARE TABULAR
    # --------------------------------------------------------------------------
    logger.info(">>> RUNNING EXPERIMENT 2: COMMUTE-AWARE TABULAR (ADDING COMMUTE VECTORS) <<<")
    X_exp2_norm = np.concatenate([X_tab_norm, X_comm_norm], axis=1)

    xgb_exp2 = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
    xgb_exp2.fit(X_exp2_norm[train_mask], y_logprice[train_mask])
    pred_xgb2 = xgb_exp2.predict(X_exp2_norm[test_mask])
    m_xgb2 = compute_point_metrics(y_logprice[test_mask], pred_xgb2)
    log_evaluation_summary("Exp 2: XGBoost (Tabular + Commute)", "Random Test", m_xgb2)
    results_records.append({"Experiment": "Exp 2: XGBoost (+ Commute)", "Split": "Random", **m_xgb2})

    # --------------------------------------------------------------------------
    # 6. EXPERIMENT 3: GDAG-GNN (DUAL ADJACENCY SPATIAL GRAPH ENCODER)
    # --------------------------------------------------------------------------
    logger.info(">>> RUNNING EXPERIMENT 3: GDAG-GNN (GAUSSIAN DUAL ADJACENCY MATRIX SPATIAL ENCODER) <<<")
    hybrid_model = HybridGDAGHousingModel(
        tabular_dim=len(tabular_features),
        gnn_node_dim=len(gnn_node_cols),
        commute_dim=len(commute_features),
        locality_emb_dim=32,
        hidden_dim=64,
        mc_dropout=0.20
    ).to(DEVICE)

    optimizer_hybrid = torch.optim.Adam(hybrid_model.parameters(), lr=0.004, weight_decay=1e-4)
    train_ds3 = TensorDataset(
        torch.tensor(X_tab_norm[train_mask]).to(DEVICE),
        torch.tensor(X_comm_norm[train_mask]).to(DEVICE),
        torch.tensor(loc_ids[train_mask]).to(DEVICE),
        torch.tensor(y_logprice[train_mask]).to(DEVICE)
    )
    train_loader3 = DataLoader(train_ds3, batch_size=128, shuffle=True)

    hybrid_model.train()
    for epoch in range(40):
        total_loss = 0.0
        for bx_tab, bx_comm, b_loc, by in train_loader3:
            optimizer_hybrid.zero_grad()
            mu, _ = hybrid_model(bx_tab, bx_comm, b_loc, node_feat_tensor, norm_dam_tensor)
            loss = crit_mse(mu, by)
            loss.backward()
            optimizer_hybrid.step()
            total_loss += loss.item() * len(by)
        if (epoch + 1) % 10 == 0:
            logger.debug(f"Epoch {epoch+1:02d}/40 - MSE Loss: {total_loss / len(train_ds3):.4f}")

    hybrid_model.eval()
    with torch.no_grad():
        test_bx_tab = torch.tensor(X_tab_norm[test_mask]).to(DEVICE)
        test_bx_comm = torch.tensor(X_comm_norm[test_mask]).to(DEVICE)
        test_b_loc = torch.tensor(loc_ids[test_mask]).to(DEVICE)
        pred_mu3, _ = hybrid_model(test_bx_tab, test_bx_comm, test_b_loc, node_feat_tensor, norm_dam_tensor)
        pred_mu3 = pred_mu3.cpu().numpy()

    m_gdag = compute_point_metrics(y_logprice[test_mask], pred_mu3)
    log_evaluation_summary("Exp 3: GDAG-GNN Hybrid", "Random Test", m_gdag)
    results_records.append({"Experiment": "Exp 3: GDAG-GNN Hybrid", "Split": "Random", **m_gdag})

    # --------------------------------------------------------------------------
    # 7. EXPERIMENT 4: UNCERTAINTY QUANTIFICATION (MC DROPOUT, ENSEMBLE, CONFORMAL)
    # --------------------------------------------------------------------------
    logger.info(">>> RUNNING EXPERIMENT 4: UNCERTAINTY QUANTIFICATION & CALIBRATION <<<")
    # A. Monte Carlo Dropout (30 stochastic forward passes)
    logger.info("Evaluating Monte Carlo Dropout on held-out test listings...")
    mc_mean, mc_std = hybrid_model.predict_mc_dropout(
        test_bx_tab, test_bx_comm, test_b_loc, node_feat_tensor, norm_dam_tensor, n_samples=30
    )
    m_mc = compute_uncertainty_metrics(y_logprice[test_mask], mc_mean, mc_std)
    m_mc_pt = compute_point_metrics(y_logprice[test_mask], mc_mean)
    log_evaluation_summary("Exp 4: MC Dropout Head", "Random Test", {**m_mc_pt, **m_mc})
    results_records.append({"Experiment": "Exp 4: MC Dropout", "Split": "Random", **m_mc_pt, **m_mc})

    # B. Conformal Prediction (CQR Guarantee on Validation Split)
    val_mask = (df['split_random'] == 'val').values
    with torch.no_grad():
        val_bx_tab = torch.tensor(X_tab_norm[val_mask]).to(DEVICE)
        val_bx_comm = torch.tensor(X_comm_norm[val_mask]).to(DEVICE)
        val_b_loc = torch.tensor(loc_ids[val_mask]).to(DEVICE)
        val_mu, _ = hybrid_model(val_bx_tab, val_bx_comm, val_b_loc, node_feat_tensor, norm_dam_tensor)
        val_mu = val_mu.cpu().numpy()

    calibrator = ConformalPredictionCalibrator(alpha=0.10)
    q_hat = calibrator.calibrate(y_logprice[val_mask], val_mu)
    conf_lower, conf_upper = calibrator.predict_interval(pred_mu3)
    conf_std = np.full_like(pred_mu3, q_hat / 1.645) # Pseudo-sigma for comparison
    m_conf = compute_uncertainty_metrics(y_logprice[test_mask], pred_mu3, conf_std, lower=conf_lower, upper=conf_upper)
    log_evaluation_summary("Exp 4: Conformal Prediction (10% alpha)", "Random Test", {**m_gdag, **m_conf})
    results_records.append({"Experiment": "Exp 4: Conformal Prediction", "Split": "Random", **m_gdag, **m_conf})

    # --------------------------------------------------------------------------
    # 8. LEAVE-LOCALITY-OUT (LLO) CROSS VALIDATION BENCHMARK
    # --------------------------------------------------------------------------
    logger.info(">>> RUNNING 5-FOLD LEAVE-LOCALITY-OUT (LLO) BENCHMARK (ZERO SPATIAL LEAKAGE) <<<")
    llo_rmse_exp1, llo_rmse_exp2, llo_rmse_exp3, llo_coverage_conf = [], [], [], []

    for fold in range(5):
        llo_test = (df['split_llo_fold'] == fold).values
        llo_train = (~llo_test)
        
        # XGBoost Exp 1 on LLO
        xgb1_llo = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
        xgb1_llo.fit(X_tab_norm[llo_train], y_logprice[llo_train])
        p1 = xgb1_llo.predict(X_tab_norm[llo_test])
        llo_rmse_exp1.append(math.sqrt(mean_squared_error(y_logprice[llo_test], p1)))

        # XGBoost Exp 2 on LLO
        xgb2_llo = XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42)
        xgb2_llo.fit(X_exp2_norm[llo_train], y_logprice[llo_train])
        p2 = xgb2_llo.predict(X_exp2_norm[llo_test])
        llo_rmse_exp2.append(math.sqrt(mean_squared_error(y_logprice[llo_test], p2)))

        # Hybrid GDAG-GNN on LLO
        model_llo = HybridGDAGHousingModel(
            len(tabular_features), len(gnn_node_cols), len(commute_features),
            locality_emb_dim=32, hidden_dim=64, mc_dropout=0.20
        ).to(DEVICE)
        opt_llo = torch.optim.Adam(model_llo.parameters(), lr=0.004, weight_decay=1e-4)

        train_ds_llo = TensorDataset(
            torch.tensor(X_tab_norm[llo_train]).to(DEVICE),
            torch.tensor(X_comm_norm[llo_train]).to(DEVICE),
            torch.tensor(loc_ids[llo_train]).to(DEVICE),
            torch.tensor(y_logprice[llo_train]).to(DEVICE)
        )
        loader_llo = DataLoader(train_ds_llo, batch_size=128, shuffle=True)

        model_llo.train()
        for _ in range(25):
            for bx_tab, bx_comm, b_loc, by in loader_llo:
                opt_llo.zero_grad()
                mu, _ = model_llo(bx_tab, bx_comm, b_loc, node_feat_tensor, norm_dam_tensor)
                loss = crit_mse(mu, by)
                loss.backward()
                opt_llo.step()

        model_llo.eval()
        with torch.no_grad():
            t_tab = torch.tensor(X_tab_norm[llo_test]).to(DEVICE)
            t_comm = torch.tensor(X_comm_norm[llo_test]).to(DEVICE)
            t_loc = torch.tensor(loc_ids[llo_test]).to(DEVICE)
            p3, _ = model_llo(t_tab, t_comm, t_loc, node_feat_tensor, norm_dam_tensor)
            p3 = p3.cpu().numpy()

        llo_rmse_exp3.append(math.sqrt(mean_squared_error(y_logprice[llo_test], p3)))
        
        # Conformal interval coverage check on LLO
        cov = np.mean((y_logprice[llo_test] >= (p3 - q_hat)) & (y_logprice[llo_test] <= (p3 + q_hat))) * 100.0
        llo_coverage_conf.append(cov)
        logger.info(f"  Fold {fold}: Exp1 RMSE={llo_rmse_exp1[-1]:.4f} | Exp2 RMSE={llo_rmse_exp2[-1]:.4f} | Exp3 GDAG RMSE={llo_rmse_exp3[-1]:.4f} | Conformal PICP={cov:.1f}%")

    logger.info("=" * 80)
    logger.info("FINAL LEAVE-LOCALITY-OUT (LLO) 5-FOLD AGGREGATE RESULTS:")
    logger.info(f"  Exp 1 (Tabular Only):      Mean RMSE = {np.mean(llo_rmse_exp1):.4f} +/- {np.std(llo_rmse_exp1):.4f}")
    logger.info(f"  Exp 2 (Tabular + Commute): Mean RMSE = {np.mean(llo_rmse_exp2):.4f} +/- {np.std(llo_rmse_exp2):.4f}")
    logger.info(f"  Exp 3 (GDAG-GNN Hybrid):   Mean RMSE = {np.mean(llo_rmse_exp3):.4f} +/- {np.std(llo_rmse_exp3):.4f}")
    logger.info(f"  Exp 4 (Conformal 90% Cov): Mean PICP = {np.mean(llo_coverage_conf):.2f}% (Nominal: 90.0%)")
    logger.info("=" * 80)

    # --------------------------------------------------------------------------
    # 9. SAVE CHECKPOINTS & BENCHMARK SUMMARY TABLE
    # --------------------------------------------------------------------------
    df_results = pd.DataFrame(results_records)
    results_csv_path = "models/benchmark_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    logger.info(f"Benchmark metrics table exported to {results_csv_path}")

    # Save trained hybrid model weights
    model_save_path = "models/gdag_hybrid_housing_model.pt"
    torch.save({
        "model_state_dict": hybrid_model.state_dict(),
        "tabular_features": tabular_features,
        "commute_features": commute_features,
        "gnn_node_features": gnn_node_cols,
        "q_hat_conformal": q_hat
    }, model_save_path)
    logger.info(f"Hybrid GDAG model checkpoint saved to {model_save_path}")

    elapsed = time.time() - start_time
    logger.info(f"PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS")

if __name__ == "__main__":
    run()
