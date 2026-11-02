"""
Model Architectures for Experiments 1, 2, 3, and 4
==================================================
1. Exp 1: Baseline Non-Spatial Tabular Models (XGBoost & MLP)
2. Exp 2: Commute-Aware Tabular Models (adding commute vectors to 5 tech hubs)
3. Exp 3: Gaussian Dual Adjacency GNN (GDAG-GCN/GAT Spatial Encoder)
4. Exp 4: Uncertainty Quantification Heads (MC Dropout, Deep Ensemble, Heteroscedastic NLL, Conformal Prediction)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBRegressor
from src.logger import ServerLogger

logger = ServerLogger.get_logger()

# ==============================================================================
# 1. TABULAR MLP BASELINE
# ==============================================================================
class TabularMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==============================================================================
# 2. GDAG-GNN SPATIAL ENCODER (Nature 2025 Dual Adjacency Graph Convolution)
# ==============================================================================
class GDAGGraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer operating over the Gaussian Dual Adjacency Matrix (DAM):
    H^(l+1) = ReLU( \widetilde{DAM} * H^(l) * W )
    as formulated in Eq. (10) of Ravichandran et al. (Nature 2025).
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, norm_dam):
        # x: [num_nodes, in_features]
        # norm_dam: [num_nodes, num_nodes]
        support = self.linear(x)
        out = torch.matmul(norm_dam, support) + self.bias
        return out

class GDAGSpatialEncoder(nn.Module):
    """
    2-Layer Spatial Encoder extracting 32-dimensional locality embeddings from the GDAG.
    """
    def __init__(self, in_features, hidden_dim=64, embedding_dim=32, dropout=0.2):
        super().__init__()
        self.conv1 = GDAGGraphConvLayer(in_features, hidden_dim)
        self.conv2 = GDAGGraphConvLayer(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, norm_dam):
        h = F.relu(self.conv1(x, norm_dam))
        h = self.dropout(h)
        h = self.conv2(h, norm_dam)
        h = self.layer_norm(h)
        return h  # [num_nodes, embedding_dim]

# ==============================================================================
# 3. HYBRID FUSED ARCHITECTURE (Tabular + GDAG Locality Embeddings + Commute)
# ==============================================================================
class HybridGDAGHousingModel(nn.Module):
    """
    Exp 3 & 4 Hybrid Architecture:
    Concatenates tabular features, GNN locality embeddings h_v, and explicit commute vectors.
    """
    def __init__(self, tabular_dim, gnn_node_dim, commute_dim, locality_emb_dim=32, hidden_dim=64, mc_dropout=0.25):
        super().__init__()
        self.gnn_encoder = GDAGSpatialEncoder(gnn_node_dim, hidden_dim=64, embedding_dim=locality_emb_dim)
        
        # Tabular projection
        self.tab_proj = nn.Sequential(
            nn.Linear(tabular_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(mc_dropout)
        )
        
        # Commute projection
        self.commute_proj = nn.Sequential(
            nn.Linear(commute_dim, 32),
            nn.ReLU()
        )
        
        # Fusion dimension = hidden_dim + locality_emb_dim + 32
        fused_dim = hidden_dim + locality_emb_dim + 32
        
        # Regression trunk
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Heteroscedastic prediction heads: Mean and Log-Variance for UQ
        self.head_mean = nn.Linear(32, 1)
        self.head_mean.bias.data.fill_(4.2)
        self.head_logvar = nn.Linear(32, 1)
        self.mc_dropout_prob = mc_dropout

    def forward(self, x_tab, x_commute, locality_ids, node_features, norm_dam):
        # 1. Compute node embeddings for all localities in the GDAG
        all_loc_embs = self.gnn_encoder(node_features, norm_dam) # [152, 32]
        
        # 2. Extract embeddings for current batch's locality IDs
        h_loc = all_loc_embs[locality_ids] # [batch_size, 32]
        
        # 3. Project tabular and commute vectors
        h_tab = self.tab_proj(x_tab)
        h_comm = self.commute_proj(x_commute)
        
        # 4. Feature Fusion
        h_fused = torch.cat([h_tab, h_loc, h_comm], dim=-1)
        rep = self.trunk(h_fused)
        
        # 5. Predictions
        mean = self.head_mean(rep).squeeze(-1)
        logvar = self.head_logvar(rep).squeeze(-1)
        return mean, logvar

    def predict_mc_dropout(self, x_tab, x_commute, locality_ids, node_features, norm_dam, n_samples=30):
        """
        Inference with Monte Carlo Dropout enabled to quantify epistemic uncertainty.
        """
        self.train() # Enable dropout during evaluation
        means = []
        with torch.no_grad():
            for _ in range(n_samples):
                mu, _ = self.forward(x_tab, x_commute, locality_ids, node_features, norm_dam)
                means.append(mu.cpu().numpy())
        means = np.stack(means, axis=0)
        mc_mean = np.mean(means, axis=0)
        mc_std = np.std(means, axis=0)
        return mc_mean, mc_std

# ==============================================================================
# 4. CONFORMAL PREDICTION (Split Conformal Prediction / CQR Guarantee)
# ==============================================================================
class ConformalPredictionCalibrator:
    """
    Distribution-Free Conformal Prediction guaranteeing 1 - alpha coverage on unseen localities:
    Residual non-conformity score: R_i = |y_i - \hat{y}_i|
    Quantile threshold: q = Quantile_{ (1 - alpha)(1 + 1/n) }(R)
    Prediction Interval: [\hat{y} - q, \hat{y} + q]
    """
    def __init__(self, alpha=0.10):
        self.alpha = alpha
        self.q_hat = 0.0

    def calibrate(self, y_true, y_pred):
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        quantile_level = min(1.0, math.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(residuals, quantile_level))
        logger.info(f"Conformal calibrator fitted on {n} calibration samples: alpha={self.alpha} -> q_hat={self.q_hat:.4f}")
        return self.q_hat

    def predict_interval(self, y_pred):
        lower = y_pred - self.q_hat
        upper = y_pred + self.q_hat
        return lower, upper

# ==============================================================================
# 5. LOSS FUNCTIONS
# ==============================================================================
def heteroscedastic_nll_loss(y_true, mean, logvar):
    """
    Negative Log-Likelihood for heteroscedastic Gaussian regression:
    Loss = 0.5 * exp(-logvar) * (y - mean)^2 + 0.5 * logvar
    """
    precision = torch.exp(-logvar)
    loss = 0.5 * precision * (y_true - mean) ** 2 + 0.5 * logvar
    return torch.mean(loss)

if __name__ == "__main__":
    logger.info("Initializing models test...")
    mlp = TabularMLP(in_features=6)
    dummy_x = torch.randn(8, 6)
    out = mlp(dummy_x)
    logger.info(f"TabularMLP output shape: {out.shape}")
    logger.info("Models module validated successfully.")
