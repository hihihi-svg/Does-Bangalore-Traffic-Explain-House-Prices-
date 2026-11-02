"""
Evaluation and Metrics Subsystem
=================================
Calculates point prediction and uncertainty quantification metrics:
1. Point: RMSE, MAPE (%), R2 Score
2. Calibration / UQ:
   - NLL (Negative Log-Likelihood)
   - PICP (Prediction Interval Coverage Probability at 90%)
   - MPIW (Mean Prediction Interval Width)
   - CRPS (Continuous Ranked Probability Score for Gaussian predictions)
3. Price-per-commute-minute elasticity (% price impact per 10 min commute)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from src.logger import ServerLogger

logger = ServerLogger.get_logger()

def compute_point_metrics(y_true, y_pred):
    """
    Computes RMSE, MAPE (in %), and R2 score.
    """
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 2),
        "r2": round(r2, 4)
    }

def compute_uncertainty_metrics(y_true, mu, sigma, lower=None, upper=None):
    """
    Computes NLL, 90% PICP coverage, MPIW, and CRPS.
    """
    sigma = np.clip(sigma, 1e-4, 1e4)
    z = (y_true - mu) / sigma
    
    # 1. Negative Log-Likelihood
    nll = 0.5 * np.log(2 * np.pi) + np.log(sigma) + 0.5 * (z ** 2)
    mean_nll = float(np.mean(nll))
    
    # 2. Prediction Interval Coverage (90% nominal)
    if lower is None or upper is None:
        # Theoretical 90% Gaussian interval: z_0.95 = 1.645
        lower = mu - 1.645 * sigma
        upper = mu + 1.645 * sigma
        
    covered = (y_true >= lower) & (y_true <= upper)
    picp_90 = float(np.mean(covered) * 100.0)
    mpiw = float(np.mean(upper - lower))
    
    # 3. CRPS (Continuous Ranked Probability Score for Gaussian distributions)
    # CRPS(N(mu, sigma^2), y) = sigma * [ z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)
    crps = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1.0 / math.sqrt(math.pi))
    mean_crps = float(np.mean(crps))
    
    return {
        "nll": round(mean_nll, 4),
        "picp_90_pct": round(picp_90, 2),
        "mpiw": round(mpiw, 4),
        "crps": round(mean_crps, 4)
    }

def log_evaluation_summary(experiment_name, split_type, metrics_dict):
    logger.info("=" * 70)
    logger.info(f"EVALUATION RESULTS: [{experiment_name}] on [{split_type}]")
    for k, v in metrics_dict.items():
        logger.info(f"  --> {k:<18}: {v}")
    logger.info("=" * 70)

if __name__ == "__main__":
    y_true = np.array([4.2, 4.5, 5.0, 3.8, 4.9])
    mu = np.array([4.1, 4.6, 4.9, 3.9, 5.0])
    sigma = np.array([0.2, 0.25, 0.18, 0.22, 0.3])
    p_met = compute_point_metrics(y_true, mu)
    u_met = compute_uncertainty_metrics(y_true, mu, sigma)
    log_evaluation_summary("Test Exp", "Random Test", {**p_met, **u_met})
