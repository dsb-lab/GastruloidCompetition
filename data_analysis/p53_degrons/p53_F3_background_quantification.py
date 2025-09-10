

import numpy as np
from scipy import stats

def fit_spillover_and_bg(B_sec, C_sec, robust=True):
    x = B_sec.astype(np.float64).ravel()
    y = C_sec.astype(np.float64).ravel()
    if robust:
        slope, intercept, _, _ = stats.theilslopes(y, x)  # robust to outliers
    else:
        # OLS: y = intercept + slope*x
        x1 = np.vstack([np.ones_like(x), x]).T
        slope, intercept = np.linalg.lstsq(x1, y, rcond=None)[0][1], np.linalg.lstsq(x1, y, rcond=None)[0][0]
    return slope, intercept  # s, b0

def correct_C(C_obs, B_obs, s, b0):
    return C_obs - b0 - s*B_obs

# Example:
# s, b0 = fit_spillover_and_bg(B_secondary_only, C_secondary_only, robust=True)
# C_corr = correct_C(C_obs, B_obs, s, b0)
