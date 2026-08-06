import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from typing import Dict, Any

class DifferenceInDifferences:
    """Difference-in-Differences (DiD) estimator for causal effect estimation."""

    def __init__(self):
        self.model = None
        self.results = None

    def estimate(self, df: pd.DataFrame, outcome: str, treatment: str,
                 time: str, cluster_var: str = None) -> Dict[str, Any]:
        """Estimate DiD using OLS with interaction term.

        Model: Y = β0 + β1*Treatment + β2*Post + β3*(Treatment×Post) + ε
        β3 is the DiD estimate.
        """
        df = df.copy()
        df['post'] = (df[time] == 1).astype(int)
        df['treat'] = df[treatment].astype(int)
        df['did_interaction'] = df['treat'] * df['post']

        formula = f'{outcome} ~ treat + post + did_interaction'

        if cluster_var:
            self.model = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df[cluster_var]}
            )
        else:
            self.model = smf.ols(formula, data=df).fit()

        did_coef = self.model.params['did_interaction']
        did_se = self.model.bse['did_interaction']
        did_pval = self.model.pvalues['did_interaction']
        did_ci = self.model.conf_int().loc['did_interaction']

        self.results = {
            'method': 'Difference-in-Differences',
            'ate': float(did_coef),
            'se': float(did_se),
            'ci_lower': float(did_ci[0]),
            'ci_upper': float(did_ci[1]),
            'p_value': float(did_pval),
            'r_squared': float(self.model.rsquared),
            'n_obs': int(self.model.nobs),
            'treatment_effect': float(self.model.params['treat']),
            'time_effect': float(self.model.params['post']),
        }

        return self.results

    def parallel_trends_test(self, df: pd.DataFrame, outcome: str,
                             treatment: str, time: str) -> Dict[str, Any]:
        """Test the parallel trends assumption using pre-treatment periods."""
        df = df.copy()
        df['post'] = (df[time] == 1).astype(int)
        df['treat'] = df[treatment].astype(int)

        # Pre-treatment only
        pre_df = df[df[time] == 0].copy()

        # Interaction with pre-treatment time (if multiple pre-periods)
        formula = f'{outcome} ~ treat'
        model_pre = smf.ols(formula, data=pre_df).fit()

        treated_pre = pre_df[pre_df[treatment] == 1][outcome]
        control_pre = pre_df[pre_df[treatment] == 0][outcome]

        # F-test for parallel trends
        treated_mean = treated_pre.mean()
        control_mean = control_pre.mean()
        treated_var = treated_pre.var()
        control_var = control_pre.var()

        diff = treated_mean - control_mean
        se_diff = np.sqrt(treated_var / len(treated_pre) + control_var / len(control_pre))

        return {
            'method': 'Parallel Trends Test',
            'treated_pre_mean': float(treated_mean),
            'control_pre_mean': float(control_mean),
            'difference': float(diff),
            'se': float(se_diff),
            't_stat': float(diff / se_diff),
            'p_value': float(2 * (1 - abs(diff / se_diff))),  # approximation
            'assumption_plausible': abs(diff / se_diff) < 1.96,
        }

    def sensitivity_to_pre_trends(self, df: pd.DataFrame, outcome: str,
                                  treatment: str, time: str,
                                  deviation_range: list = None) -> pd.DataFrame:
        """Assess sensitivity to pre-trend violations."""
        if deviation_range is None:
            deviation_range = np.arange(-2, 2.5, 0.5)

        base_result = self.estimate(df, outcome, treatment, time)
        base_ate = base_result['ate']

        results = []
        for delta in deviation_range:
            df_mod = df.copy()
            df_mod.loc[df_mod[treatment] == 1, outcome] += delta * df_mod[time]

            mod_result = self.estimate(df_mod, outcome, treatment, time)

            results.append({
                'pre_trend_deviation': float(delta),
                'estimated_ate': mod_result['ate'],
                'bias': mod_result['ate'] - base_ate,
                'bias_pct': (mod_result['ate'] - base_ate) / abs(base_ate) * 100 if base_ate != 0 else 0,
            })

        return pd.DataFrame(results)

    def summary(self) -> pd.DataFrame:
        """Return summary of DiD results."""
        if self.results is None:
            raise ValueError("Must call estimate() first")

        return pd.DataFrame([{
            'Estimate': self.results['ate'],
            'Std Error': self.results['se'],
            '95% CI': f"[{self.results['ci_lower']:.3f}, {self.results['ci_upper']:.3f}]",
            'p-value': f"{self.results['p_value']:.4f}",
            'R²': f"{self.results['r_squared']:.4f}",
        }])
