# Bangalore Real Estate & Traffic Spatial-GNN Dataset Suite
**Project:** *Does Bangalore Traffic Explain House Prices? Uncertainty-Aware Price Prediction with Road-Network GNNs and Bayesian Heads*  
**Domain:** Urban Computing, Spatial Graph Neural Networks, Bayesian Machine Learning  

---

## 1. Executive Summary & Overview

This dataset suite fulfills the complete multi-modal data pipeline specified in the project proposal and presentation deck. It integrates:
1. **Hedonic Real Estate Listings:** ~13,320 Bangalore housing listings parsed and cleaned.
2. **Geocoded Master Localities:** Canonical neighborhood centroids with OpenStreetMap alignment.
3. **5 Primary Employment / Tech Clusters:** Directional commute destinations (Whitefield, Bellandur/ORR, Electronic City, Manyata Tech Park, and MG Road/CBD).
4. **Traffic Congestion & Travel Time Matrix:** Peak (09:00 AM) vs. off-peak (02:00 PM) commute times and congestion ratios.
5. **Namma Metro Transit Accessibility:** Distance and access times to 64 operational BMRCL metro stations across Purple, Green, and Yellow corridors.
6. **Locality Road-Network Graph Topology:** A spatial graph with 152 nodes and 872 edges formatted for inductive PyTorch Geometric (`GraphSAGE` / `GAT`) modeling without target leakage.
7. **Cross-Validation Split Flags:** Random splits, 5-fold Leave-Locality-Out (LLO), and Spatial Quadrant blocks.

---

## 2. Directory Structure & File Inventory

All generated dataset files are organized in the [`data/`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/) directory:

| File Name | Description | Rows / Shape | Size |
|---|---|---|---|
| [`Bengaluru_House_Data.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/Bengaluru_House_Data.csv) | Raw scraped/curated Kaggle Bengaluru real estate listings. | 13,320 rows | ~903 KB |
| [`bangalore_tech_hubs.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_tech_hubs.csv) | Centroids & corridor metadata for the 5 major tech/employment hubs. | 5 hubs | ~1 KB |
| [`bangalore_metro_stations.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_metro_stations.csv) | Operational BMRCL Namma Metro stations with geographic coordinates. | 64 stations | ~3 KB |
| [`bangalore_localities_master.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_localities_master.csv) | Master catalog of canonical Bangalore localities, zones, and coordinates. | 152 localities | ~7 KB |
| [`bangalore_locality_traffic_matrix.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_locality_traffic_matrix.csv) | Peak/off-peak travel times, congestion ratios, and metro proximity per locality. | 152 rows, 25 cols | ~32 KB |
| [`bangalore_housing_cleaned_tabular.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_housing_cleaned_tabular.csv) | Cleaned, validated listing structural features (BHK, sqft, bath, price, log-price). | 10,009 rows | ~1.2 MB |
| [`bangalore_housing_spatial_commute_dataset.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_housing_spatial_commute_dataset.csv) | **Unified Master Modeling Dataset** combining listings, traffic vectors, metro access, and split markers. | 10,008 rows, 57 cols | ~3.6 MB |
| [`bangalore_locality_graph_edges.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_locality_graph_edges.csv) | Locality road-network graph edges, distances, travel times, and spatial weights. | 872 edges | ~42 KB |
| [`bangalore_locality_graph_nodes.csv`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_locality_graph_nodes.csv) | Inductive GNN node features (strictly non-leaked: commute, metro, degree). | 152 nodes, 14 cols | ~12 KB |
| [`bangalore_graph_pyg.pt`](file:///c:/Users/lalan/Desktop/Deep%20Learning/data/bangalore_graph_pyg.pt) | PyTorch Geometric bundle (`x`, `edge_index`, `edge_attr`, `feature_names`). | PyTorch Tensor Dict | ~31 KB |

---

## 3. Data Dictionary: Unified Modeling Dataset (`bangalore_housing_spatial_commute_dataset.csv`)

### Structural & Listing Features
- `listing_id` *(int)*: Unique identifier for each listing.
- `area_type` *(str)*: `Super built-up Area`, `Built-up Area`, `Plot Area`, `Carpet Area`.
- `availability` *(str)*: Availability status string.
- `is_ready_to_move` *(int: 0/1)*: Binary flag for ready-to-move properties.
- `location` *(str)*: Original raw location text from Kaggle listing.
- `canonical_locality_name` *(str)*: Standardized canonical locality name.
- `locality_match_score` *(float)*: RapidFuzz similarity match score (75.0 - 100.0).
- `bhk` *(int)*: Bedroom, Hall, Kitchen count (parsed integer 1–9).
- `total_sqft_cleaned` *(float)*: Total area in square feet after unit standardization.
- `bath_cleaned` *(int)*: Bathroom count (imputed via BHK median).
- `balcony_cleaned` *(int)*: Balcony count (imputed via BHK median).
- `price_lakhs` *(float)*: Property price in Lakhs INR (1 Lakh = 100,000 INR).
- `price_inr` *(float)*: Property price in Indian Rupees.
- `log_price` *(float)*: $\ln(\text{price\_lakhs})$, primary target for regression and Bayesian models.
- `price_per_sqft` *(float)*: Price per square foot (INR/sqft).

### Spatial & Locality Metadata
- `locality_id` *(int)*: Integer node index (0 to 151) corresponding to the GNN graph.
- `latitude` *(float)*: Locality centroid latitude in WGS84 coordinates.
- `longitude` *(float)*: Locality centroid longitude in WGS84 coordinates.
- `zone` *(str)*: Bangalore geographic zone (e.g., East, South, North, West, Central).
- `spatial_quadrant` *(str)*: Spatial macro-block (East, South, North, West, Central).

### Commute & Traffic Congestion Features
- `dist_<hub>_road_km` *(float)*: Realistic road network distance to tech hub (km).
- `commute_<hub>_offpeak_min` *(float)*: Off-peak (02:00 PM) commute time (minutes).
- `commute_<hub>_peak_min` *(float)*: Peak-hour (09:00 AM) congested commute time (minutes).
- `congestion_ratio_<hub>` *(float)*: Ratio of peak commute time to off-peak time.
- *Hubs covered:*
  - `whitefield` (ITPL, EPIP Corridor)
  - `bellandur_orr` (Outer Ring Road, EcoSpace, Embassy TechVillage)
  - `electronic_city` (Phase 1 & 2, Infosys, Wipro)
  - `manyata_tech_park` (Hebbal / Nagavara, North Corridor)
  - `mg_road_cbd` (Central Business District)
- `min_commute_peak_min` *(float)*: Peak commute time to the closest of the 5 tech hubs.
- `avg_commute_peak_min` *(float)*: Mean peak commute time across all 5 tech hubs.
- `avg_congestion_ratio` *(float)*: Average peak-to-offpeak friction across Bangalore corridors.
- `dist_to_nearest_tech_hub_km` *(float)*: Road distance to the nearest tech cluster.

### Public Transit / Metro Accessibility
- `nearest_metro_station` *(str)*: Name of closest BMRCL operational metro station.
- `nearest_metro_line` *(str)*: Transit corridor (Purple, Green, Yellow).
- `dist_to_metro_km` *(float)*: Road distance to nearest metro station (km).
- `commute_metro_access_min` *(float)*: Feeder / last-mile travel time to metro (min).

### Experimental Splits & Evaluation Markers
- `split_random` *(str: 'train' | 'val' | 'test')*: Standard 80/10/10 split benchmark.
- `split_llo_fold` *(int: 0 to 4)*: 5-fold Leave-Locality-Out cross-validation fold. All listings of a locality are strictly contained within a single fold, ensuring 0 spatial data leakage.
- `spatial_quadrant` *(str)*: For Spatial Block cross-validation (train on 4 quadrants, test out-of-distribution generalization on the held-out quadrant).

---

## 4. Graph Neural Network (GNN) Specifications

### Graph Definition $G = (V, E)$
- **Nodes ($V = 152$):** Canonical Bangalore localities.
- **Edges ($E = 872$):** Directed spatial connectivity based on road network adjacency (K-nearest neighbors with $k=6$, bounded by distance threshold $\le 9$ km).
- **Edge Attributes:**
  - `distance_km`: Inter-locality road distance.
  - `peak_commute_min`: Peak driving time between nodes.
  - `spatial_edge_weight`: Decaying spatial Gaussian/exponential weight $w_{uv} = \exp(-d_{uv} / 4.0)$.

### Inductive Node Feature Matrix $X \in \mathbb{R}^{152 \times 11}$
Strictly non-leaked (contains no listing counts or price statistics):
1. `latitude` (normalized)
2. `longitude` (normalized)
3. `node_degree` (arterial graph connectivity)
4. `commute_whitefield_peak_min` (normalized)
5. `commute_bellandur_peak_min` (normalized)
6. `commute_ecity_peak_min` (normalized)
7. `commute_manyata_peak_min` (normalized)
8. `commute_mgroad_peak_min` (normalized)
9. `avg_congestion_ratio` (normalized)
10. `dist_to_metro_km` (normalized)
11. `commute_metro_access_min` (normalized)

### PyTorch Geometric Loading Example
```python
import torch

data = torch.load("data/bangalore_graph_pyg.pt", weights_only=False)
x = data["x"]  # [152, 11]
edge_index = data["edge_index"]  # [2, 872]
edge_attr = data["edge_attr"]  # [872, 1]

print(f"Loaded GNN graph with {x.size(0)} nodes and {edge_index.size(1)} edges.")
```

---

## 5. Experimental Alignment (How to run Exp 1 – Exp 4)

| Exp ID | Experiment Focus | Input Features | Target & Loss |
|---|---|---|---|
| **Exp 1** | Baseline Non-Spatial | Tabular attributes only (`bhk`, `total_sqft_cleaned`, `bath_cleaned`, `balcony_cleaned`, `area_type`, `is_ready_to_move`) via XGBoost / MLP. | `log_price` (MSE loss) |
| **Exp 2** | Traffic Value-Add | Tabular attributes + explicit commute vectors (`commute_*_peak_min`, `dist_to_metro_km`, `avg_congestion_ratio`). | `log_price` (MSE loss) |
| **Exp 3** | Road-Network GNN Topology | Tabular features + 32-dim locality embeddings $h_v$ extracted from a 2-layer `GraphSAGE` or `GAT` trained on `bangalore_graph_pyg.pt`. | `log_price` (MSE loss) |
| **Exp 4** | Full Uncertainty & Calibration | Fused representations $[h_{tab} \parallel h_v \parallel t_{commute}]$ with: <br>• MC Dropout <br>• Deep Ensemble (5 models) <br>• Conformal Prediction (MAPIE) <br>• Hierarchical Bayes (PyMC / NumPyro) | Heteroscedastic NLL / Interval Coverage (PICP, CRPS) under Leave-Locality-Out splits |

---

## 6. Execution & Verification Scripts

- **Rebuild Dataset:**
  ```powershell
  python scripts/build_bangalore_dataset.py
  ```
- **Run Verification & Quality Assurance:**
  ```powershell
  python scripts/verify_dataset.py
  ```
