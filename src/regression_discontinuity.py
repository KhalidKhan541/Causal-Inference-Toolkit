import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any

class RegressionDiscontinuityDesign:
    """Regression Discontinuity Design (RDD) estimator for causal effect at cutoff."""

    def __init__(self, cutoff: float = 0.0):
        self.cutoff = cutoff
        self.results = None

    def estimate_rdd(self, df: pd.DataFrame, outcome: str,
                     running_var: str, bandwidth: float = None) -> Dict[str, Any]:
        """Estimate RDD effect using local linear regression."""
        rdd_df = df.copy()
        rdd_df['centered'] = rdd_df[running_var] - self.cutoff
        rdd_df['above_cutoff'] = (rdd_df[running_var] >= self.cutoff).astype(int)
        rdd_df['interaction'] = rdd_df['centered'] * rdd_df['above_cutoff']

        # Apply bandwidth if specified
        if bandwidth is not None:
            rdd_df = rdd_df[rdd_df['centered'].abs() <= bandwidth]

        # Local linear regression
        X = sm.add_constant(rdd_df[['centered', 'above_cutoff', 'interaction']])
        y = rdd_df[outcome]

        model = sm.OLS(y, X).fit()

        rdd_effect = model.params['above_cutoff']
        rdd_se = model.bse['above_cutoff']
        rdd_ci = model.conf_int().loc['above_cutoff']

        self.results = {
            'method': 'Regression Discontinuity Design',
            'ate': float(rdd_effect),
            'se': float(rdd_se),
            'ci_lower': float(rdd_ci[0]),
            'ci_upper': float(rdd_ci[1]),
            'p_value': float(model.pvalues['above_cutoff']),
            'r_squared': float(model.rsquared),
            'n_obs': int(model.nobs),
            'bandwidth': bandwidth,
            'bandwidth_used': float(bandwidth) if bandwidth else None,
        }

        return self.results

    def optimal_bandwidth(self, df: pd.DataFrame, outcome: str,
                          running_var: str) -> float:
        """Compute optimal bandwidth using Imbens-Kalyanaraman rule of thumb."""
        rdd_df = df.copy()
        rdd_df['centered'] = rdd_df[running_var] - self.cutoff
        rdd_df['above_cutoff'] = (rdd_df[running_var] >= self.cutoff).astype(int)

        n = len(rdd_df)

        # Variance estimation
        X = sm.add_constant(rdd_df[['centered', 'above_cutoff', 'interaction']])
        y = rdd_df[outcome]
        model = sm.OLS(y, X).fit()
        sigma2 = model.mse_resid

        # Bandwidth rule of thumb
        h = 1.06 * np.std(rdd_df['centered']) * (sigma2 / n) ** (1/5)

        return float(h)

    def sensitivity_to_bandwidth(self, df: pd.DataFrame, outcome: str,
                                 running_var: str,
                                 bandwidth_range: list = None) -> pd.DataFrame:
        """Estimate RDD across different bandwidths."""
        if bandwidth_range is None:
            max_range = abs(df[running_var] - self.cutoff).max()
            bandwidth_range = np.linspace(0.1, max_range * 0.5, 10)

        results = []
        for bw in bandwidth_range:
            est = self.estimate_rdd(df, outcome, running_var, bandwidth=bw)
            results.append({
                'bandwidth': float(bw),
                'ate': est['ate'],
                'se': est['se'],
                'ci_lower': est['ci_lower'],
                'ci_upper': est['ci_upper'],
                'n_obs': est['n_obs'],
            })

        return pd.DataFrame(results)

    def placebo_test(self, df: pd.DataFrame, outcome: str,
                     running_var: str, placebo_cutoffs: list = None) -> pd.DataFrame:
        """Run placebo tests at non-cutoff points."""
        if placebo_cutoffs is None:
            placebo_cutoffs = np.linspace(
                df[running_var].quantile(0.1),
                df[running_var].quantile(0.9),
                5
            )

        results = []
        for cutoff in placebo_cutoffs:
            rdd_temp = RegressionDiscontinuityDesign(cutoff=cutoff)
            est = rdd_temp.estimate_rdd(df, outcome, running_var)
            results.append({
                'cutoff': float(cutoff),
                'ate': est['ate'],
                'se': est['se'],
                'p_value': est['p_value'],
                'is_placebo': True,
            })

        # Add actual cutoff
        actual_est = self.estimate_rdd(df, outcome, running_var)
        results.append({
            'cutoff': self.cutoff,
            'ate': actual_est['ate'],
            'se': actual_est['se'],
            'p_value': actual_est['p_value'],
            'is_placebo': False,
        })

        return pd.DataFrame(results)
