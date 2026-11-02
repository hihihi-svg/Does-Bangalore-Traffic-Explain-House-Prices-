"""
Validation & Quality Assurance Script for Bangalore Housing + GNN Dataset
==========================================================================
Verifies:
1. File existence and schemas across all generated assets
2. Absence of data leakage in graph node features (no price targets)
3. Strict locality isolation in Leave-Locality-Out (LLO) splits
4. PyTorch Geometric graph bundle integrity (x, edge_index, edge_attr)
5. Basic statistical summary of house prices and commute metrics
"""

import os
import torch
import numpy as np
import pandas as pd

def verify_all():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE DATASET VERIFICATION & INTEGRITY CHECKS")
    print("=" * 80)

    # 1. Check generated files
    files = [
        "data/Bengaluru_House_Data.csv",
        "data/bangalore_tech_hubs.csv",
        "data/bangalore_metro_stations.csv",
        "data/bangalore_localities_master.csv",
        "data/bangalore_locality_traffic_matrix.csv",
        "data/bangalore_housing_cleaned_tabular.csv",
        "data/bangalore_housing_spatial_commute_dataset.csv",
        "data/bangalore_locality_graph_edges.csv",
        "data/bangalore_locality_graph_nodes.csv",
        "data/bangalore_graph_pyg.pt"
    ]
    for f in files:
        assert os.path.exists(f), f"Missing file: {f}"
        sz_kb = os.path.getsize(f) / 1024.0
        print(f"  [OK] Found {f:<55} ({sz_kb:.1f} KB)")

    # 2. Check Unified Modeling Dataset
    df_unified = pd.read_csv("data/bangalore_housing_spatial_commute_dataset.csv")
    print(f"\n[1] Unified Modeling Dataset Checks:")
    print(f"  Total records: {len(df_unified)}")
    print(f"  Total features: {df_unified.shape[1]}")
    print(f"  Unique canonical localities: {df_unified['canonical_locality_name'].nunique()}")
    
    # Check critical columns
    crit_cols = [
        'listing_id', 'total_sqft_cleaned', 'bhk', 'bath_cleaned', 'price_lakhs', 'log_price',
        'latitude', 'longitude', 'commute_whitefield_peak_min', 'commute_bellandur_orr_peak_min',
        'commute_electronic_city_peak_min', 'dist_to_metro_km', 'split_random', 'split_llo_fold',
        'spatial_quadrant'
    ]
    for c in crit_cols:
        assert c in df_unified.columns, f"Missing column: {c}"
        assert df_unified[c].isna().sum() == 0, f"Nulls found in {c}"
    print("  [OK] All critical columns present with 0 missing values.")

    # 3. Verify Strict Locality Isolation in Leave-Locality-Out (LLO) Splits
    print(f"\n[2] Leave-Locality-Out (LLO) Cross-Validation Verification:")
    for fold in range(5):
        test_locs = set(df_unified[df_unified['split_llo_fold'] == fold]['canonical_locality_name'])
        train_locs = set(df_unified[df_unified['split_llo_fold'] != fold]['canonical_locality_name'])
        overlap = test_locs.intersection(train_locs)
        assert len(overlap) == 0, f"Spatial leakage in fold {fold}: {overlap}"
        test_listings = (df_unified['split_llo_fold'] == fold).sum()
        print(f"  Fold {fold}: {len(test_locs):>2} held-out localities ({test_listings:>4} test listings, 0 overlap with train)")
    print("  [OK] Zero spatial data leakage across all 5 LLO folds.")

    # 4. Verify Zero Target Leakage in Graph Node Features
    print(f"\n[3] Graph Node Inductive Features Leakage Check:")
    df_nodes = pd.read_csv("data/bangalore_locality_graph_nodes.csv")
    leak_keywords = ['price', 'target', 'lakh', 'cost', 'sqft_cleaned', 'sale', 'rent']
    leaked = [c for c in df_nodes.columns if any(k in c.lower() for k in leak_keywords)]
    assert len(leaked) == 0, f"Leaked target features in graph nodes: {leaked}"
    print(f"  Node columns: {list(df_nodes.columns)}")
    print("  [OK] Node feature matrix is strictly inductive (only commute, metro, road topology & coordinates).")

    # 5. Verify PyTorch Geometric Graph Bundle
    print(f"\n[4] PyTorch Geometric Tensor Bundle Verification:")
    pyg_data = torch.load("data/bangalore_graph_pyg.pt", weights_only=False)
    print(f"  x (Node Features): shape={pyg_data['x'].shape}, dtype={pyg_data['x'].dtype}")
    print(f"  edge_index:        shape={pyg_data['edge_index'].shape}, dtype={pyg_data['edge_index'].dtype}")
    print(f"  edge_attr:         shape={pyg_data['edge_attr'].shape}, dtype={pyg_data['edge_attr'].dtype}")
    assert pyg_data['x'].shape[0] == pyg_data['num_nodes']
    assert pyg_data['edge_index'].shape[1] == pyg_data['num_edges']
    assert not torch.isnan(pyg_data['x']).any()
    assert not torch.isnan(pyg_data['edge_attr']).any()
    print("  [OK] PyG graph tensor bundle validated with 0 NaN values.")

    # 6. Basic Descriptive Statistics
    print(f"\n[5] Key Domain Statistics:")
    print(f"  House Price (Lakhs INR): Median = {df_unified['price_lakhs'].median():.1f}, IQR = [{df_unified['price_lakhs'].quantile(0.25):.1f}, {df_unified['price_lakhs'].quantile(0.75):.1f}]")
    print(f"  Square Footage (sqft):   Median = {df_unified['total_sqft_cleaned'].median():.0f}, Range = [{df_unified['total_sqft_cleaned'].min():.0f}, {df_unified['total_sqft_cleaned'].max():.0f}]")
    print(f"  Distance to Metro (km):  Median = {df_unified['dist_to_metro_km'].median():.2f} km, Mean = {df_unified['dist_to_metro_km'].mean():.2f} km")
    print(f"  Min Peak Commute (min):  Median = {df_unified['min_commute_peak_min'].median():.1f} min, Range = [{df_unified['min_commute_peak_min'].min():.1f}, {df_unified['min_commute_peak_min'].max():.1f}] min")
    print(f"  Avg Congestion Ratio:    Mean = {df_unified['avg_congestion_ratio'].mean():.2f}x peak-to-offpeak friction")

    print("\n" + "=" * 80)
    print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (READY FOR EXPERIMENTS 1-4)")
    print("=" * 80)

if __name__ == "__main__":
    verify_all()
