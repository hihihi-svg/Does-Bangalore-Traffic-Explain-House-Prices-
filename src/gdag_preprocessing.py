"""
Algorithm 1: Local-Global Invariant IQR & Min-Max Normalization Preprocessing
=============================================================================
Implements the pre-processing methodology from:
Ravichandran et al. (Nature Scientific Reports 2025, 15:41208)
"Gaussian dual adjacency graph based spatial correlated and temporal time dependent
traffic prediction in Bangalore City"

Key mathematical operations:
1. Local-Global Invariant IQR Series Splitting & Outlier Filtering:
   ResO = X(IQR) = X(Q3) - X(Q1)
   Filter bounds: [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
   Evaluated both globally and within sub-environments (localities & BHK classes).
2. Min-Max Normalization:
   x_norm = (x - x_min) / (x_max - x_min)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
from src.logger import ServerLogger

logger = ServerLogger.get_logger()

class LocalGlobalIQRPreprocessor:
    def __init__(self, iqr_multiplier=1.5):
        self.iqr_multiplier = iqr_multiplier
        self.scalers = {}

    def filter_outliers(self, df: pd.DataFrame, continuous_cols=['total_sqft_cleaned', 'price_per_sqft']) -> pd.DataFrame:
        """
        Applies Local-Global Invariant IQR outlier filtering (Algorithm 1 from Nature 2025 paper).
        Filters anomalies both at global level and conditioned on local sub-environments.
        """
        logger.info(f"Applying Local-Global Invariant IQR filtering on {len(df)} initial records...")
        initial_len = len(df)
        valid_mask = pd.Series(True, index=df.index)

        # 1. Global Invariant IQR Filtering
        for col in continuous_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            col_mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            valid_mask = valid_mask & col_mask
            logger.debug(f"Global IQR [{col}]: Q1={q1:.1f}, Q3={q3:.1f}, IQR={iqr:.1f} -> Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")

        # 2. Local Sub-environment Invariant IQR Filtering (grouped by BHK)
        for bhk_val, group in df.groupby('bhk'):
            if len(group) >= 20:
                q1_local = group['total_sqft_cleaned'].quantile(0.25)
                q3_local = group['total_sqft_cleaned'].quantile(0.75)
                iqr_local = q3_local - q1_local
                l_lower = q1_local - self.iqr_multiplier * iqr_local
                l_upper = q3_local + self.iqr_multiplier * iqr_local
                local_mask = (df['bhk'] == bhk_val) & ((df['total_sqft_cleaned'] < l_lower) | (df['total_sqft_cleaned'] > l_upper))
                valid_mask = valid_mask & (~local_mask)

        filtered_df = df[valid_mask].copy().reset_index(drop=True)
        dropped_count = initial_len - len(filtered_df)
        logger.info(f"Local-Global Invariant IQR filtering complete: Kept {len(filtered_df)} / {initial_len} records ({dropped_count} outliers removed, {len(filtered_df)/initial_len*100:.2f}% retained)")
        return filtered_df

    def min_max_normalize(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        """
        Min-Max Normalization as specified in Eq. (8) of the Nature 2025 paper:
        PD = (x - x_min) / (x_max - x_min)
        """
        df_norm = df.copy()
        for col in feature_cols:
            col_min = df_norm[col].min()
            col_max = df_norm[col].max()
            if col_max - col_min > 1e-6:
                df_norm[f"{col}_minmax"] = (df_norm[col] - col_min) / (col_max - col_min)
                self.scalers[col] = (col_min, col_max)
            else:
                df_norm[f"{col}_minmax"] = 0.0
                self.scalers[col] = (col_min, col_max)
        logger.info(f"Min-Max normalization applied across {len(feature_cols)} continuous features.")
        return df_norm

if __name__ == "__main__":
    df = pd.read_csv("data/bangalore_housing_spatial_commute_dataset.csv")
    preprocessor = LocalGlobalIQRPreprocessor()
    df_clean = preprocessor.filter_outliers(df)
    logger.info(f"Verified preprocessor on master dataset: {len(df_clean)} records ready.")
