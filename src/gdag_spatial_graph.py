"""
Algorithm 2: Gaussian Dual Adjacency Graph (GDAG) & Feature Extraction
======================================================================
Implements the core spatial graph methodology from:
Ravichandran et al. (Nature Scientific Reports 2025, 15:41208)

Key mathematical formulations:
1. Static Adjacency Matrix A_static (Physical Road Graph Topology)
2. Dynamic Adjacency Matrix A_dynamic via Gaussian RBF Kernel:
   A_dynamic(i, j) = exp(- ||x_i - x_j||^2 / (2 * sigma^2))
   reflecting dynamic semantic traffic and commute associations across Bangalore.
3. Dual Adjacency Matrix (DAM):
   DAM = A_static + A_dynamic
4. Normalized Graph Laplacian / Dual Adjacency:
   D^(-1/2) * DAM * D^(-1/2)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import math
import numpy as np
import pandas as pd
import torch
from src.logger import ServerLogger

logger = ServerLogger.get_logger()

class GaussianDualAdjacencyGraph:
    def __init__(self, sigma=1.0, dynamic_threshold=0.35, max_neighbors=6):
        self.sigma = sigma
        self.dynamic_threshold = dynamic_threshold
        self.max_neighbors = max_neighbors
        self.num_nodes = 0
        self.DAM = None
        self.normalized_DAM = None

    def build_dual_adjacency_matrix(self, df_nodes: pd.DataFrame, df_static_edges: pd.DataFrame):
        """
        Constructs the static, dynamic, and dual adjacency matrices.
        """
        self.num_nodes = len(df_nodes)
        logger.info(f"Constructing Gaussian Dual Adjacency Graph (GDAG) across {self.num_nodes} locality nodes...")

        # 1. Static Adjacency Matrix A_static
        A_static = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for _, edge in df_static_edges.iterrows():
            u = int(edge['source_locality_id'])
            v = int(edge['target_locality_id'])
            w = float(edge.get('spatial_edge_weight', 1.0))
            if u < self.num_nodes and v < self.num_nodes:
                A_static[u, v] = w
                A_static[v, u] = w  # Undirected physical road connection
        
        # Add self-loops to static adjacency
        np.fill_diagonal(A_static, 1.0)
        logger.info(f"Static Adjacency Matrix initialized: non-zero entries={np.count_nonzero(A_static)}")

        # 2. Dynamic Gaussian Kernel Adjacency Matrix A_dynamic
        # Uses commute time vectors to 5 tech hubs + congestion ratios + metro access
        dynamic_cols = [
            'commute_whitefield_peak_min', 'commute_bellandur_peak_min',
            'commute_ecity_peak_min', 'commute_manyata_peak_min',
            'commute_mgroad_peak_min', 'avg_congestion_ratio',
            'dist_to_metro_km'
        ]
        X_dyn = df_nodes[dynamic_cols].values.astype(np.float32)
        # Standardize for Gaussian kernel computation
        X_dyn_std = (X_dyn - np.mean(X_dyn, axis=0)) / (np.std(X_dyn, axis=0) + 1e-6)

        A_dynamic = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        gamma = 1.0 / (2.0 * (self.sigma ** 2))

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                diff = X_dyn_std[i] - X_dyn_std[j]
                dist_sq = np.dot(diff, diff)
                k_val = float(np.exp(-gamma * dist_sq))
                if k_val >= self.dynamic_threshold:
                    A_dynamic[i, j] = k_val
                    A_dynamic[j, i] = k_val

        np.fill_diagonal(A_dynamic, 1.0)
        logger.info(f"Dynamic Gaussian Adjacency Matrix computed (sigma={self.sigma}, cutoff={self.dynamic_threshold}): non-zero entries={np.count_nonzero(A_dynamic)}")

        # 3. Dual Adjacency Matrix (DAM) = A_static + A_dynamic
        DAM = A_static + A_dynamic
        # Normalize weights to [0, 1] range
        DAM = DAM / DAM.max()
        self.DAM = DAM

        # 4. Normalized Dual Adjacency Matrix: D^(-1/2) * DAM * D^(-1/2)
        deg = np.sum(DAM, axis=1)
        deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
        deg_inv_sqrt[deg == 0] = 0.0
        D_mat = np.diag(deg_inv_sqrt)
        self.normalized_DAM = D_mat @ DAM @ D_mat
        logger.info(f"Dual Adjacency Matrix (DAM) normalized successfully. Shape={DAM.shape}, Total edges={np.count_nonzero(DAM > 0.05)}")

        return self.DAM, self.normalized_DAM

    def to_pyg_tensors(self, df_nodes: pd.DataFrame, feature_cols: list):
        """
        Converts the normalized dual adjacency matrix into PyG edge_index and edge_attr.
        """
        assert self.DAM is not None, "Call build_dual_adjacency_matrix first."
        
        # Node features tensor
        X_raw = df_nodes[feature_cols].values.astype(np.float32)
        X_mean = np.mean(X_raw, axis=0)
        X_std = np.std(X_raw, axis=0) + 1e-6
        X_norm = (X_raw - X_mean) / X_std
        x_tensor = torch.tensor(X_norm, dtype=torch.float32)

        # Convert DAM to edge_index and edge_attr
        sources, targets, weights = [], [], []
        threshold = 0.05
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and self.DAM[i, j] > threshold:
                    sources.append(i)
                    targets.append(j)
                    weights.append(self.DAM[i, j])

        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        logger.info(f"Converted GDAG to PyG tensors: {x_tensor.shape[0]} nodes, {edge_index.shape[1]} dual-adjacency edges, {len(feature_cols)} features per node.")
        return x_tensor, edge_index, edge_attr

if __name__ == "__main__":
    df_nodes = pd.read_csv("data/bangalore_locality_graph_nodes.csv")
    df_edges = pd.read_csv("data/bangalore_locality_graph_edges.csv")
    gdag = GaussianDualAdjacencyGraph(sigma=1.2, dynamic_threshold=0.30)
    DAM, norm_DAM = gdag.build_dual_adjacency_matrix(df_nodes, df_edges)
    logger.info("GDAG module test passed successfully.")
