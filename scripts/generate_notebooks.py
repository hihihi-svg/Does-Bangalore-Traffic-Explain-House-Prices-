"""
Automated Generator for Project Jupyter Notebooks (.ipynb)
===========================================================
Creates 4 comprehensive, modular Jupyter Notebooks adhering to nbformat v4:
1. 01_bangalore_gdag_preprocessing_and_eda.ipynb
2. 02_gaussian_dual_adjacency_graph_gdag.ipynb
3. 03_baseline_models_exp1_and_exp2.ipynb
4. 04_gdag_gnn_and_bayesian_uncertainty_exp3_exp4.ipynb
"""

import os
import nbformat as nbf

os.makedirs('notebooks', exist_ok=True)

def create_notebook(filename, cells):
    nb = nbf.v4.new_notebook()
    nb['cells'] = cells
    nb['metadata'] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11"
        }
    }
    with open(filename, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"[OK] Generated notebook: {filename}")

# ==============================================================================
# NOTEBOOK 1: Preprocessing & EDA
# ==============================================================================
cells_nb1 = [
    nbf.v4.new_markdown_cell("""# 01. Bangalore Housing & Traffic EDA with Local-Global Invariant IQR Preprocessing
**Project:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*  
**Paper Grounding:** Ravichandran et al. (*Nature Scientific Reports*, 2025, 15:41208)  
**Method:** Algorithm 1: Local-Global Invariant Inter Quartile Range (IQR) & Min-Max Normalization

---
### Objectives:
1. Initialize centralized server and training logs.
2. Load the Bangalore Real Estate Listings dataset and Spatial Commute Matrix.
3. Implement and apply **Algorithm 1 (Local-Global Invariant IQR)** from the Nature 2025 paper to filter spatial outliers without overfitting.
4. Exploratory Data Analysis (EDA) on house prices, peak/off-peak travel times to the 5 major tech corridors, and metro proximity."""),

    nbf.v4.new_code_cell("""import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.logger import ServerLogger, log_system_telemetry
from src.gdag_preprocessing import LocalGlobalIQRPreprocessor

# Initialize live server logging
logger = ServerLogger.get_logger()
log_system_telemetry()
logger.info("Notebook 01: Preprocessing & EDA initialized.")"""),

    nbf.v4.new_markdown_cell("""## 1. Load Data Assets"""),

    nbf.v4.new_code_cell("""# Load master spatial commute dataset
df_raw = pd.read_csv('../data/bangalore_housing_spatial_commute_dataset.csv')
df_hubs = pd.read_csv('../data/bangalore_tech_hubs.csv')
df_metro = pd.read_csv('../data/bangalore_metro_stations.csv')

print(f"Master Dataset Shape: {df_raw.shape}")
print(f"Number of Tech Hubs: {len(df_hubs)}")
print(f"Number of Operational Metro Stations: {len(df_metro)}")
df_hubs"""),

    nbf.v4.new_markdown_cell("""## 2. Algorithm 1: Local-Global Invariant IQR Preprocessing
As formulated in Eq. (5) & (8) of Ravichandran et al. (Nature 2025):
$$ResO = X(IQR) = X(Q_3) - X(Q_1)$$
Filtered bounds: $[Q_1 - 1.5 \times IQR, Q_3 + 1.5 \times IQR]$ evaluated both globally and conditioned on local sub-environments (BHK and locality segments)."""),

    nbf.v4.new_code_cell("""preprocessor = LocalGlobalIQRPreprocessor(iqr_multiplier=1.5)
df_clean = preprocessor.filter_outliers(df_raw, continuous_cols=['total_sqft_cleaned', 'price_per_sqft'])

print(f"Initial raw rows: {len(df_raw)}")
print(f"Cleaned rows:     {len(df_clean)}")
print(f"Retained:         {len(df_clean)/len(df_raw)*100:.2f}%")"""),

    nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis: Price vs. Commute Metrics"""),

    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of Log-Price
sns.histplot(df_clean['log_price'], kde=True, ax=axes[0], color='navy')
axes[0].set_title("Log-Price Distribution", fontsize=12)
axes[0].set_xlabel("log(Price in Lakhs)")

# Price vs Square Footage
sns.scatterplot(x='total_sqft_cleaned', y='price_lakhs', data=df_clean, alpha=0.3, ax=axes[1], color='teal')
axes[1].set_title("Price vs Total Sqft", fontsize=12)
axes[1].set_xlabel("Total Area (sqft)")
axes[1].set_ylabel("Price (Lakhs INR)")

# Price vs Min Peak Commute Time
sns.scatterplot(x='min_commute_peak_min', y='log_price', data=df_clean, alpha=0.3, ax=axes[2], color='crimson')
axes[2].set_title("Log-Price vs Min Peak Commute to Tech Hub", fontsize=12)
axes[2].set_xlabel("Peak Commute to Nearest Tech Hub (min)")
axes[2].set_ylabel("log(Price in Lakhs)")

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 4. Corridor Congestion Analysis: Peak vs. Off-Peak Friction"""),

    nbf.v4.new_code_cell("""corridors = ['whitefield', 'bellandur_orr', 'electronic_city', 'manyata_tech_park', 'mg_road_cbd']
congestion_summary = []

for c in corridors:
    peak_col = f"commute_{c}_peak_min"
    offpeak_col = f"commute_{c}_offpeak_min"
    ratio_col = f"congestion_ratio_{c}"
    congestion_summary.append({
        "Corridor": c.replace('_', ' ').title(),
        "Mean Off-Peak (min)": df_clean[offpeak_col].mean(),
        "Mean Peak (min)": df_clean[peak_col].mean(),
        "Mean Congestion Ratio": df_clean[ratio_col].mean()
    })

pd.DataFrame(congestion_summary)"""),

    nbf.v4.new_markdown_cell("""## 5. Summary & Server Log Checkpoint"""),

    nbf.v4.new_code_cell("""logger.info(f"Notebook 01 execution completed. Verified {len(df_clean)} cleaned listings.")
print("Check logs/server.log and logs/training.log to view live backend logs.")""")
]

create_notebook("notebooks/01_bangalore_gdag_preprocessing_and_eda.ipynb", cells_nb1)

# ==============================================================================
# NOTEBOOK 2: Gaussian Dual Adjacency Graph (GDAG)
# ==============================================================================
cells_nb2 = [
    nbf.v4.new_markdown_cell("""# 02. Gaussian Dual Adjacency Graph (GDAG) Construction
**Project:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*  
**Paper Grounding:** Ravichandran et al. (*Nature Scientific Reports*, 2025, 15:41208)  
**Method:** Algorithm 2: Gaussian Dual Adjacency Graph (GDAG) & Dynamic Adjacency Matrix

---
### Objectives:
1. Construct the **Static Adjacency Matrix** $A_{\text{static}}$ from physical road connectivity.
2. Compute the **Gaussian RBF Dynamic Adjacency Matrix** $A_{\text{dynamic}}$ based on traffic congestion profiles.
3. Formulate the **Dual Adjacency Matrix** $DAM = A_{\text{static}} + A_{\text{dynamic}}$ and its normalized Laplacian.
4. Visualize how $DAM$ creates semantic traffic bridges between disjoint bottleneck corridors in Bangalore (e.g. Silk Board and Marathahalli)."""),

    nbf.v4.new_code_cell("""import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.logger import ServerLogger
from src.gdag_spatial_graph import GaussianDualAdjacencyGraph

logger = ServerLogger.get_logger()
logger.info("Notebook 02: GDAG Spatial Graph initialized.")"""),

    nbf.v4.new_markdown_cell("""## 1. Load Locality Nodes and Road Graph Edges"""),

    nbf.v4.new_code_cell("""df_nodes = pd.read_csv('../data/bangalore_locality_graph_nodes.csv')
df_edges = pd.read_csv('../data/bangalore_locality_graph_edges.csv')

print(f"Number of canonical locality nodes: {len(df_nodes)}")
print(f"Number of physical road edges:       {len(df_edges)}")
df_nodes.head(4)"""),

    nbf.v4.new_markdown_cell("""## 2. Compute Gaussian Dual Adjacency Matrix (DAM)
$$A_{\text{dynamic}}(i, j) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)$$
$$DAM = A_{\text{static}} + A_{\text{dynamic}}$$
$$\widetilde{DAM} = D^{-1/2}(DAM) \cdot DAM \cdot D^{-1/2}(DAM)$$"""),

    nbf.v4.new_code_cell("""gdag = GaussianDualAdjacencyGraph(sigma=1.2, dynamic_threshold=0.30)
DAM, norm_DAM = gdag.build_dual_adjacency_matrix(df_nodes, df_edges)

print(f"Dual Adjacency Matrix shape: {DAM.shape}")
print(f"Non-zero elements:            {np.count_nonzero(DAM > 0.05)}")
print(f"Matrix density:               {np.count_nonzero(DAM > 0.05) / (DAM.shape[0]**2)*100:.2f}%")"""),

    nbf.v4.new_markdown_cell("""## 3. Adjacency Heatmap Visualization: Static vs Dynamic vs Dual"""),

    nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Sample 30 localities for clean visualization
sample_n = 30
sns.heatmap(norm_DAM[:sample_n, :sample_n], ax=axes[0], cmap='Blues', cbar=False)
axes[0].set_title(f"Normalized DAM (First {sample_n} Localities)", fontsize=11)

# Degree distribution of DAM
deg = np.sum(DAM > 0.1, axis=1)
sns.histplot(deg, kde=True, ax=axes[1], color='teal')
axes[1].set_title("Locality Node Degree Distribution in GDAG", fontsize=11)
axes[1].set_xlabel("Degree (Connected Localities)")

# Eigenvalue Spectrum of Normalized DAM
eigenvalues = np.linalg.eigvalsh(norm_DAM)
axes[2].plot(np.sort(eigenvalues), marker='o', color='purple', markersize=3)
axes[2].set_title("Graph Laplacian Eigenvalue Spectrum", fontsize=11)
axes[2].set_xlabel("Eigenvalue Index")
axes[2].set_ylabel("Eigenvalue Magnitude")

plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 4. Convert GDAG to PyTorch Geometric Tensors"""),

    nbf.v4.new_code_cell("""feature_cols = [
    'latitude', 'longitude', 'node_degree',
    'commute_whitefield_peak_min', 'commute_bellandur_peak_min',
    'commute_ecity_peak_min', 'commute_manyata_peak_min',
    'commute_mgroad_peak_min', 'avg_congestion_ratio',
    'dist_to_metro_km', 'commute_metro_access_min'
]
x_tensor, edge_index, edge_attr = gdag.to_pyg_tensors(df_nodes, feature_cols)

print(f"Node feature tensor shape: {x_tensor.shape}")
print(f"Edge index tensor shape:    {edge_index.shape}")
print(f"Edge attr tensor shape:     {edge_attr.shape}")
logger.info("Notebook 02 execution completed. PyG tensors validated.")""")
]

create_notebook("notebooks/02_gaussian_dual_adjacency_graph_gdag.ipynb", cells_nb2)

# ==============================================================================
# NOTEBOOK 3: Baseline Models (Exp 1 & Exp 2)
# ==============================================================================
cells_nb3 = [
    nbf.v4.new_markdown_cell("""# 03. Baseline Real Estate Valuation Models (Exp 1 vs Exp 2)
**Project:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*  
**Research Question 1 (Predictive):** *Do peak-hour commute features improve price prediction over non-spatial tabular baselines?*

---
### Objectives:
1. **Experiment 1**: Train non-spatial tabular baselines (XGBoost & MLP) using structural property features only.
2. **Experiment 2**: Train commute-aware tabular models incorporating directional commute times to 5 tech hubs, metro distance, and congestion ratios.
3. Quantify the predictive gain of traffic features under:
   - **Random Split** (Standard reference)
   - **Leave-Locality-Out (LLO 5-Fold)** (Zero spatial leakage)
4. Analyze feature importance to understand which tech corridors drive house valuations."""),

    nbf.v4.new_code_cell("""import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

from src.logger import ServerLogger
from src.evaluation import compute_point_metrics, log_evaluation_summary

logger = ServerLogger.get_logger()
logger.info("Notebook 03: Baseline Models Exp 1 & 2 initialized.")"""),

    nbf.v4.new_markdown_cell("""## 1. Load Prepared Dataset and Feature Definitions"""),

    nbf.v4.new_code_cell("""df = pd.read_csv('../data/bangalore_housing_spatial_commute_dataset.csv')

# Tabular listing attributes (Exp 1)
tabular_features = ['bhk', 'total_sqft_cleaned', 'bath_cleaned', 'balcony_cleaned', 'is_ready_to_move']
area_dummies = pd.get_dummies(df['area_type'], prefix='area', drop_first=True)
df = pd.concat([df, area_dummies], axis=1)
tabular_features.extend(list(area_dummies.columns))

# Explicit Commute & Transit Features (Exp 2)
commute_features = [
    'commute_whitefield_peak_min', 'commute_whitefield_offpeak_min',
    'commute_bellandur_orr_peak_min', 'commute_bellandur_orr_offpeak_min',
    'commute_electronic_city_peak_min', 'commute_electronic_city_offpeak_min',
    'commute_manyata_tech_park_peak_min', 'commute_manyata_tech_park_offpeak_min',
    'commute_mg_road_cbd_peak_min', 'commute_mg_road_cbd_offpeak_min',
    'min_commute_peak_min', 'avg_commute_peak_min',
    'avg_congestion_ratio', 'dist_to_metro_km', 'commute_metro_access_min'
]

print(f"Exp 1 Tabular Features ({len(tabular_features)}): {tabular_features}")
print(f"Exp 2 Commute Features ({len(commute_features)}): {commute_features}")"""),

    nbf.v4.new_markdown_cell("""## 2. Train Exp 1 vs. Exp 2 on Random Split Benchmark"""),

    nbf.v4.new_code_cell("""train_mask = (df['split_random'] == 'train').values
test_mask = (df['split_random'] == 'test').values

# Feature matrices
X_tab = df[tabular_features].values
X_comm = df[commute_features].values
X_all = np.concatenate([X_tab, X_comm], axis=1)
y = df['log_price'].values

# Fit Exp 1 (Tabular Only)
xgb_exp1 = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
xgb_exp1.fit(X_tab[train_mask], y[train_mask])
pred_exp1 = xgb_exp1.predict(X_tab[test_mask])
m_exp1 = compute_point_metrics(y[test_mask], pred_exp1)

# Fit Exp 2 (Tabular + Commute)
xgb_exp2 = XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42)
xgb_exp2.fit(X_all[train_mask], y[train_mask])
pred_exp2 = xgb_exp2.predict(X_all[test_mask])
m_exp2 = compute_point_metrics(y[test_mask], pred_exp2)

df_comp = pd.DataFrame([
    {"Model": "Exp 1: XGBoost (Tabular Only)", **m_exp1},
    {"Model": "Exp 2: XGBoost (Tabular + Commute)", **m_exp2}
])
df_comp"""),

    nbf.v4.new_markdown_cell("""## 3. Feature Importance Analysis (Which Corridors Matter Most?)"""),

    nbf.v4.new_code_cell("""all_feature_names = tabular_features + commute_features
importances = xgb_exp2.feature_importances_
feat_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False).head(12)

plt.figure(figsize=(10, 5))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Top Feature Importances in Commute-Aware Real Estate Valuation", fontsize=12)
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 4. Leave-Locality-Out (LLO 5-Fold) Generalization Gap"""),

    nbf.v4.new_code_cell("""llo_exp1_scores = []
llo_exp2_scores = []

for fold in range(5):
    test_fold = (df['split_llo_fold'] == fold).values
    train_fold = (~test_fold)
    
    # Train Exp 1
    m1 = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42)
    m1.fit(X_tab[train_fold], y[train_fold])
    p1 = m1.predict(X_tab[test_fold])
    llo_exp1_scores.append(compute_point_metrics(y[test_fold], p1)['rmse'])
    
    # Train Exp 2
    m2 = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.08, random_state=42)
    m2.fit(X_all[train_fold], y[train_fold])
    p2 = m2.predict(X_all[test_fold])
    llo_exp2_scores.append(compute_point_metrics(y[test_fold], p2)['rmse'])

print(f"5-Fold LLO Mean RMSE (Exp 1 Tabular Only):    {np.mean(llo_exp1_scores):.4f} +/- {np.std(llo_exp1_scores):.4f}")
print(f"5-Fold LLO Mean RMSE (Exp 2 Tabular+Commute): {np.mean(llo_exp2_scores):.4f} +/- {np.std(llo_exp2_scores):.4f}")
logger.info("Notebook 03 execution completed.")""")
]

create_notebook("notebooks/03_baseline_models_exp1_and_exp2.ipynb", cells_nb3)

# ==============================================================================
# NOTEBOOK 4: GDAG-GNN & Uncertainty Quantification (Exp 3 & Exp 4)
# ==============================================================================
cells_nb4 = [
    nbf.v4.new_markdown_cell("""# 04. GDAG-GNN & Uncertainty-Aware Price Prediction (Exp 3 & Exp 4)
**Project:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*  
**Paper Grounding:** Ravichandran et al. (*Nature Scientific Reports*, 2025, 15:41208)  
**Methods:**
- **Exp 3**: Gaussian Dual Adjacency GNN (`GDAGSpatialEncoder`) + Tabular Feature Fusion.
- **Exp 4**: Uncertainty Quantification (Monte Carlo Dropout, Deep Ensemble, Heteroscedastic NLL, Conformal Prediction CQR).

---
### Objectives:
1. Train the hybrid GDAG-GNN model end-to-end in PyTorch.
2. Evaluate point prediction accuracy vs tabular baselines.
3. Quantify spatial uncertainty on unseen localities (Leave-Locality-Out splits).
4. Calibrate prediction intervals using Conformal Prediction (10% nominal miscoverage guarantee).
5. Compute the inferential economic price penalty per 10 minutes of peak commute."""),

    nbf.v4.new_code_cell("""import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.logger import ServerLogger
from src.gdag_spatial_graph import GaussianDualAdjacencyGraph
from src.models import HybridGDAGHousingModel, ConformalPredictionCalibrator
from src.evaluation import compute_point_metrics, compute_uncertainty_metrics

logger = ServerLogger.get_logger()
logger.info("Notebook 04: GDAG-GNN & Bayesian Uncertainty initialized.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {DEVICE}")"""),

    nbf.v4.new_markdown_cell("""## 1. Load Data Assets & Construct Normalized DAM"""),

    nbf.v4.new_code_cell("""df = pd.read_csv('../data/bangalore_housing_spatial_commute_dataset.csv')
df_nodes = pd.read_csv('../data/bangalore_locality_graph_nodes.csv')
df_edges = pd.read_csv('../data/bangalore_locality_graph_edges.csv')

# Build Dual Adjacency Matrix
gdag = GaussianDualAdjacencyGraph(sigma=1.2, dynamic_threshold=0.30)
DAM, norm_DAM = gdag.build_dual_adjacency_matrix(df_nodes, df_edges)
norm_dam_tensor = torch.tensor(norm_DAM, dtype=torch.float32).to(DEVICE)

# Node features
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

# Tabular & Commute features
tabular_features = ['bhk', 'total_sqft_cleaned', 'bath_cleaned', 'balcony_cleaned', 'is_ready_to_move']
area_dummies = pd.get_dummies(df['area_type'], prefix='area', drop_first=True)
df = pd.concat([df, area_dummies], axis=1)
tabular_features.extend(list(area_dummies.columns))

commute_features = [
    'commute_whitefield_peak_min', 'commute_whitefield_offpeak_min',
    'commute_bellandur_orr_peak_min', 'commute_bellandur_orr_offpeak_min',
    'commute_electronic_city_peak_min', 'commute_electronic_city_offpeak_min',
    'commute_manyata_tech_park_peak_min', 'commute_manyata_tech_park_offpeak_min',
    'commute_mg_road_cbd_peak_min', 'commute_mg_road_cbd_offpeak_min',
    'min_commute_peak_min', 'avg_commute_peak_min',
    'avg_congestion_ratio', 'dist_to_metro_km', 'commute_metro_access_min'
]

# Locality mapping
loc_map = {row['canonical_locality_name']: int(row['locality_id']) for _, row in df_nodes.iterrows()}
df['locality_node_id'] = df['canonical_locality_name'].map(loc_map)
df = df[df['locality_node_id'].notna()].copy()
df['locality_node_id'] = df['locality_node_id'].astype(int)

# Standardize inputs
X_tab = (df[tabular_features].values - np.mean(df[tabular_features].values, axis=0)) / (np.std(df[tabular_features].values, axis=0) + 1e-6)
X_comm = (df[commute_features].values - np.mean(df[commute_features].values, axis=0)) / (np.std(df[commute_features].values, axis=0) + 1e-6)
loc_ids = df['locality_node_id'].values
y = df['log_price'].values.astype(np.float32)"""),

    nbf.v4.new_markdown_cell("""## 2. Train Hybrid GDAG-GNN Model (Exp 3)"""),

    nbf.v4.new_code_cell("""train_mask = (df['split_random'] == 'train').values
val_mask = (df['split_random'] == 'val').values
test_mask = (df['split_random'] == 'test').values

model = HybridGDAGHousingModel(
    tabular_dim=len(tabular_features),
    gnn_node_dim=len(gnn_node_cols),
    commute_dim=len(commute_features),
    locality_emb_dim=32,
    hidden_dim=64,
    mc_dropout=0.20
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

train_ds = TensorDataset(
    torch.tensor(X_tab[train_mask], dtype=torch.float32).to(DEVICE),
    torch.tensor(X_comm[train_mask], dtype=torch.float32).to(DEVICE),
    torch.tensor(loc_ids[train_mask], dtype=torch.long).to(DEVICE),
    torch.tensor(y[train_mask], dtype=torch.float32).to(DEVICE)
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)

losses = []
model.train()
for epoch in range(35):
    ep_loss = 0.0
    for bx_tab, bx_comm, b_loc, by in train_loader:
        optimizer.zero_grad()
        mu, _ = model(bx_tab, bx_comm, b_loc, node_feat_tensor, norm_dam_tensor)
        loss = criterion(mu, by)
        loss.backward()
        optimizer.step()
        ep_loss += loss.item() * len(by)
    losses.append(ep_loss / len(train_ds))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:02d}/35: Loss = {losses[-1]:.4f}")

plt.figure(figsize=(7, 3.5))
plt.plot(losses, color='crimson')
plt.title("GDAG-GNN Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 3. Exp 4: Monte Carlo Dropout Uncertainty Quantification"""),

    nbf.v4.new_code_cell("""test_bx_tab = torch.tensor(X_tab[test_mask], dtype=torch.float32).to(DEVICE)
test_bx_comm = torch.tensor(X_comm[test_mask], dtype=torch.float32).to(DEVICE)
test_b_loc = torch.tensor(loc_ids[test_mask], dtype=torch.long).to(DEVICE)

# 30 stochastic forward passes with dropout active
mc_mean, mc_std = model.predict_mc_dropout(
    test_bx_tab, test_bx_comm, test_b_loc, node_feat_tensor, norm_dam_tensor, n_samples=30
)
y_test = y[test_mask]

pt_metrics = compute_point_metrics(y_test, mc_mean)
uq_metrics = compute_uncertainty_metrics(y_test, mc_mean, mc_std)

print("MC Dropout Evaluation:")
print(f"  RMSE:        {pt_metrics['rmse']:.4f}")
print(f"  MAPE:        {pt_metrics['mape_pct']:.2f}%")
print(f"  NLL:         {uq_metrics['nll']:.4f}")
print(f"  90% PICP:    {uq_metrics['picp_90_pct']:.2f}% (Nominal: 90%)")
print(f"  MPIW:        {uq_metrics['mpiw']:.4f}")"""),

    nbf.v4.new_markdown_cell("""## 4. Exp 4: Conformal Prediction (CQR Calibrated Intervals)"""),

    nbf.v4.new_code_cell("""# Calibrate on validation split
model.eval()
with torch.no_grad():
    val_bx_tab = torch.tensor(X_tab[val_mask], dtype=torch.float32).to(DEVICE)
    val_bx_comm = torch.tensor(X_comm[val_mask], dtype=torch.float32).to(DEVICE)
    val_b_loc = torch.tensor(loc_ids[val_mask], dtype=torch.long).to(DEVICE)
    val_mu, _ = model(val_bx_tab, val_bx_comm, val_b_loc, node_feat_tensor, norm_dam_tensor)
    val_mu = val_mu.cpu().numpy()

calibrator = ConformalPredictionCalibrator(alpha=0.10)
q_hat = calibrator.calibrate(y[val_mask], val_mu)
lower_conf, upper_conf = calibrator.predict_interval(mc_mean)

conformal_coverage = np.mean((y_test >= lower_conf) & (y_test <= upper_conf)) * 100.0
print(f"Conformal Prediction Interval Half-Width: +/- {q_hat:.4f}")
print(f"Empirical Coverage on Test Set:           {conformal_coverage:.2f}% (Guaranteed >= 90.0%)")"""),

    nbf.v4.new_markdown_cell("""## 5. Calibration Plot & Uncertainty Intervals"""),

    nbf.v4.new_code_cell("""sample_idx = np.argsort(mc_mean)[:60]
plt.figure(figsize=(14, 5))
plt.plot(range(60), y_test[sample_idx], 'ro', label='True Log-Price', markersize=4)
plt.plot(range(60), mc_mean[sample_idx], 'b-', label='Predicted Mean (mu)')
plt.fill_between(range(60), lower_conf[sample_idx], upper_conf[sample_idx], color='blue', alpha=0.15, label='90% Conformal Interval')
plt.title("Calibrated Conformal Prediction Intervals (Sample of 60 Test Listings)", fontsize=12)
plt.xlabel("Listing Index (Sorted by Price)")
plt.ylabel("log(Price in Lakhs)")
plt.legend()
plt.tight_layout()
plt.show()"""),

    nbf.v4.new_markdown_cell("""## 6. Inferential Commute Elasticity (% Price Drop per 10 Min Commute)"""),

    nbf.v4.new_code_cell("""# Estimate marginal price elasticity per 10 minutes of commute to Whitefield
min_per_10 = 10.0
# Baseline prediction
with torch.no_grad():
    base_mu, _ = model(test_bx_tab[:100], test_bx_comm[:100], test_b_loc[:100], node_feat_tensor, norm_dam_tensor)
    
    # Perturb Whitefield commute by +10 min (scaled)
    wf_idx = commute_features.index('commute_whitefield_peak_min')
    perturbed_comm = test_bx_comm[:100].clone()
    std_wf = np.std(df['commute_whitefield_peak_min'])
    perturbed_comm[:, wf_idx] += (10.0 / std_wf)
    
    pert_mu, _ = model(test_bx_tab[:100], perturbed_comm, test_b_loc[:100], node_feat_tensor, norm_dam_tensor)
    
pct_change = (torch.exp(pert_mu) - torch.exp(base_mu)) / torch.exp(base_mu) * 100.0
mean_pct_change = pct_change.mean().item()
print(f"Marginal Commute Impact: {mean_pct_change:.2f}% price change per +10 min peak commute to Whitefield.")
logger.info("Notebook 04 execution completed.")""")
]

create_notebook("notebooks/04_gdag_gnn_and_bayesian_uncertainty_exp3_exp4.ipynb", cells_nb4)
