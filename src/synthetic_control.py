import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Any

class SyntheticControl:
    """Synthetic Control method for causal effect estimation with treated unit and donor pool."""

    def __init__(self):
        self.weights = None
        self.results = None

    def fit(self, treated_ts: np.ndarray, donor_ts: np.ndarray,
            treatment_start: int) -> Dict[str, Any]:
        """Fit synthetic control by optimizing pre-treatment fit.

        Parameters:
            treated_ts: Time series for treated unit (length T)
            donor_ts: Matrix of donor unit time series (J x T)
            treatment_start: Index when treatment begins
        """
        n_donors = donor_ts.shape[0]

        def objective(w):
            synthetic = w @ donor_ts
            pre_treatment = treated_ts[:treatment_start]
            synthetic_pre = synthetic[:treatment_start]
            return np.sum((pre_treatment - synthetic_pre) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_donors)]
        w0 = np.ones(n_donors) / n_donors

        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        self.weights = result.x
        synthetic_full = self.weights @ donor_ts
        pre_fit = np.sqrt(np.mean((treated_ts[:treatment_start] - synthetic_full[:treatment_start]) ** 2))
        treatment_effect = treated_ts[treatment_start:] - synthetic_full[treatment_start:]

        self.results = {
            'method': 'Synthetic Control',
            'weights': self.weights.tolist(),
            'pre_treatment_rmse': float(pre_fit),
            'treatment_effect': treatment_effect.tolist(),
            'mean_effect': float(np.mean(treatment_effect)),
            'cumulative_effect': float(np.sum(treatment_effect)),
        }
        return self.results

    def placebo_test(self, treated_ts: np.ndarray, donor_ts: np.ndarray,
                     treatment_start: int, n_permutations: int = 100) -> Dict[str, Any]:
        """Permutation placebo test: iteratively assign treatment to each donor."""
        n_donors = donor_ts.shape[0]
        original_effect = self.results['mean_effect']
        placebo_effects = []

        for i in range(n_permutations):
            placebo_idx = np.random.randint(n_donors)
            placebo_treated = donor_ts[placebo_idx].copy()
            placebo_donors = donor_ts.copy()
            placebo_donors[placebo_idx] = treated_ts.copy()

            sc_placebo = SyntheticControl()
            sc_placebo.fit(placebo_treated, placebo_donors, treatment_start)
            placebo_effects.append(sc_placebo.results['mean_effect'])

        p_value = np.mean(np.abs(placebo_effects) >= np.abs(original_effect))

        return {
            'method': 'Synthetic Control (Placebo)',
            'original_effect': original_effect,
            'placebo_effects': placebo_effects,
            'p_value': float(p_value),
            'n_permutations': n_permutations,
            'significant': p_value < 0.05,
        }

    def inference(self, treated_ts: np.ndarray, donor_ts: np.ndarray,
                  treatment_start: int, n_bootstraps: int = 200) -> Dict[str, Any]:
        """Bootstrap inference for treatment effect."""
        n_donors = donor_ts.shape[0]
        T = len(treated_ts)
        bootstrap_effects = []

        for b in range(n_bootstraps):
            bootstrap_idx = np.random.choice(n_donors, size=n_donors, replace=True)
            bootstrap_donors = donor_ts[bootstrap_idx]
            treated_noisy = treated_ts + np.random.normal(0, 0.1, T)

            sc_boot = SyntheticControl()
            sc_boot.fit(treated_noisy, bootstrap_donors, treatment_start)
            bootstrap_effects.append(sc_boot.results['mean_effect'])

        ci_lower = np.percentile(bootstrap_effects, 2.5)
        ci_upper = np.percentile(bootstrap_effects, 97.5)

        return {
            'method': 'Synthetic Control (Bootstrap)',
            'ate': self.results['mean_effect'],
            'se': float(np.std(bootstrap_effects)),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_bootstraps': n_bootstraps,
        }

    def summary(self) -> pd.DataFrame:
        """Return summary of synthetic control results."""
        if self.results is None:
            raise ValueError("Must call fit() first")

        return pd.DataFrame([{
            'Estimate': self.results['mean_effect'],
            'Pre-RMSE': self.results['pre_treatment_rmse'],
            'N Donors': len(self.weights),
            'Effective Donors': int(np.sum(self.weights > 0.01)),
        }])
