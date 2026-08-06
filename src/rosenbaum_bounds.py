import numpy as np
import pandas as pd
from typing import Dict, Any

class RosenbaumBounds:
    """Rosenbaum sensitivity analysis for unmeasured confounding in matched studies."""
    
    def __init__(self, matched_pairs: list, df: pd.DataFrame, outcome: str):
        self.matched_pairs = matched_pairs
        self.df = df
        self.outcome = outcome
        self.treated_outcomes = np.array([df.loc[t, outcome] for t, c in matched_pairs])
        self.control_outcomes = np.array([df.loc[c, outcome] for t, c in matched_pairs])
    
    def compute_gamma_range(self, gamma_min: float = 1.0, gamma_max: float = 3.0,
                           step: float = 0.1) -> pd.DataFrame:
        """Compute bounds for a range of gamma (treatment assignment odds)."""
        gammas = np.arange(gamma_min, gamma_max + step, step)
        results = []
        
        for gamma in gammas:
            bounds = self.compute_bounds(gamma)
            results.append(bounds)
        
        return pd.DataFrame(results)
    
    def compute_bounds(self, gamma: float) -> Dict[str, Any]:
        """Compute Rosenbaum bounds for a given gamma.
        
        Gamma = odds ratio of treatment assignment.
        Gamma=1 means random assignment (no hidden bias).
        Gamma=2 means a treated unit could be 2x more likely to be treated
        than a control unit with same observed covariates.
        """
        diffs = self.treated_outcomes - self.control_outcomes
        
        # Wilcoxon signed-rank statistic
        abs_diffs = np.abs(diffs)
        ranks = np.argsort(np.argsort(abs_diffs)) + 1
        signed_ranks = ranks * np.sign(diffs)
        
        W_plus = np.sum(signed_ranks[signed_ranks > 0])
        W_minus = np.sum(np.abs(signed_ranks[signed_ranks < 0]))
        
        n = len(self.matched_pairs)
        
        # Expected value and variance under gamma
        E_W = n * (n + 1) / 4
        p_upper = gamma / (1 + gamma)
        p_lower = 1 / (1 + gamma)
        
        var_W = 0
        for i in range(1, n + 1):
            var_W += i**2 * (p_upper * (1 - p_upper) + p_lower * (1 - p_lower))
        
        # Upper bound p-value
        z_upper = (W_plus - E_W) / np.sqrt(var_W) if var_W > 0 else 0
        p_value_upper = 1 - self._normal_cdf(z_upper)
        
        # Lower bound p-value
        z_lower = (W_minus - E_W) / np.sqrt(var_W) if var_W > 0 else 0
        p_value_lower = 1 - self._normal_cdf(z_lower)
        
        return {
            'gamma': gamma,
            'W_plus': float(W_plus),
            'W_minus': float(W_minus),
            'E_W': float(E_W),
            'Var_W': float(var_W),
            'z_upper': float(z_upper),
            'z_lower': float(z_lower),
            'p_value_upper': float(p_value_upper),
            'p_value_lower': float(p_value_lower),
            'significant_upper': p_value_upper < 0.05,
            'significant_lower': p_value_lower < 0.05,
        }
    
    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + np.tanh(0.7071067811865475 * x))
    
    def critical_gamma(self, alpha: float = 0.05) -> Dict[str, Any]:
        """Find critical gamma where result becomes non-significant."""
        gamma = 1.0
        while gamma < 10:
            bounds = self.compute_bounds(gamma)
            if bounds['p_value_upper'] > alpha:
                return {
                    'critical_gamma': gamma,
                    'alpha': alpha,
                    'interpretation': f'Result becomes non-significant at gamma={gamma:.1f}',
                    'hidden_bias_needed': f'Hidden bias of {gamma:.1f}x would negate the effect',
                }
            gamma += 0.1
        
        return {
            'critical_gamma': float('inf'),
            'alpha': alpha,
            'interpretation': 'Result robust even with extreme hidden bias',
        }
    
    def summary(self) -> pd.DataFrame:
        """Summary of bounds at common gamma values."""
        gammas = [1.0, 1.5, 2.0, 2.5, 3.0]
        results = []
        for g in gammas:
            bounds = self.compute_bounds(g)
            results.append({
                'Gamma': g,
                'W+': f"{bounds['W_plus']:.0f}",
                'W-': f"{bounds['W_minus']:.0f}",
                'p-value (upper)': f"{bounds['p_value_upper']:.4f}",
                'p-value (lower)': f"{bounds['p_value_lower']:.4f}",
                'Sig?': 'Yes' if bounds['significant_upper'] else 'No',
            })
        return pd.DataFrame(results)
