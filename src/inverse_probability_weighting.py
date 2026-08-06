import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

class InverseProbabilityWeighting:
    """Inverse Probability Weighting (IPW) for causal effect estimation."""
    
    def __init__(self):
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.propensity_scores = None
        self.weights = None
    
    def estimate_propensity_scores(self, df: pd.DataFrame,
                                   covariates: list, treatment: str) -> np.ndarray:
        """Estimate propensity scores."""
        X = self.scaler.fit_transform(df[covariates])
        y = df[treatment].values
        
        self.propensity_model.fit(X, y)
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        return self.propensity_scores
    
    def compute_weights(self, df: pd.DataFrame, treatment: str,
                        stabilize: bool = True, trim: float = 0.01) -> np.ndarray:
        """Compute IPW weights with optional stabilization and trimming."""
        ps = self.propensity_scores.copy()
        
        # Trim extreme propensity scores
        ps = np.clip(ps, trim, 1 - trim)
        
        treatment_vals = df[treatment].values
        
        if stabilize:
            p_treat = np.mean(treatment_vals)
            p_control = 1 - p_treat
            weights = np.where(
                treatment_vals == 1,
                p_treat / ps,
                p_control / (1 - ps)
            )
        else:
            weights = np.where(
                treatment_vals == 1,
                1.0 / ps,
                1.0 / (1 - ps)
            )
        
        self.weights = weights
        return weights
    
    def estimate_ate(self, df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
        """Estimate ATE using IPW."""
        if self.weights is None:
            raise ValueError("Must call compute_weights() before estimate_ate()")
        
        treatment_vals = df[treatment].values
        outcome_vals = df[outcome].values
        
        # ATE = E[Y*T/w] - E[Y*(1-T)/w]
        weighted-treated = np.sum(outcome_vals * treatment_vals * self.weights) / np.sum(treatment_vals * self.weights)
        weighted_control = np.sum(outcome_vals * (1 - treatment_vals) * self.weights) / np.sum((1 - treatment_vals) * self.weights)
        
        ate = weighted_treated - weighted_control
        
        # Robust standard error
        n = len(df)
        treated_mask = treatment_vals == 1
        
        var_treated = np.sum(((outcome_vals[treated_mask] - weighted_treated) * self.weights[treated_mask])**2) / (np.sum(treated_mask) ** 2)
        var_control = np.sum(((outcome_vals[~treated_mask] - weighted_control) * self.weights[~treated_mask])**2) / (np.sum(~treated_mask) ** 2)
        
        se = np.sqrt(var_treated + var_control)
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'method': 'Inverse Probability Weighting',
            'ate': float(ate),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'mean_weight': float(np.mean(self.weights)),
            'max_weight': float(np.max(self.weights)),
            'effectiveness': float(np.var(self.weights)),
        }
    
    def estimate_ate_robust(self, df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
        """IPW with sandwich (robust) standard errors."""
        if self.weights is None:
            raise ValueError("Must call compute_weights() before estimate_ate_robust()")
        
        treatment_vals = df[treatment].values
        outcome_vals = df[outcome].values
        
        # ATE
        wt = self.weights * treatment_vals
        wc = self.weights * (1 - treatment_vals)
        
        yt = np.sum(outcome_vals * wt) / np.sum(wt)
        yc = np.sum(outcome_vals * wc) / np.sum(wc)
        
        ate = yt - yc
        
        # Sandwich SE
        n = len(df)
        treated_mask = treatment_vals == 1
        
        psi_treated = (outcome_vals[treated_mask] - yt) * self.weights[treated_mask]
        psi_control = (outcome_vals[~treated_mask] - yc) * self.weights[~treated_mask]
        
        var_treated = np.mean(psi_treated**2) / (np.mean(self.weights[treated_mask])**2)
        var_control = np.mean(psi_control**2) / (np.mean(self.weights[~treated_mask])**2)
        
        se = np.sqrt(var_treated + var_control)
        
        return {
            'method': 'IPW (Robust SE)',
            'ate': float(ate),
            'se': float(se),
            'ci_lower': float(ate - 1.96 * se),
            'ci_upper': float(ate + 1.96 * se),
        }
    
    def sensitivity_analysis(self, df: pd.DataFrame, outcome: str,
                             gamma_range: list = None) -> pd.DataFrame:
        """Conduct sensitivity analysis by varying treatment effect on outcomes."""
        if gamma_range is None:
            gamma_range = np.arange(-5, 6, 1)
        
        results = []
        base_estimate = self.estimate_ate(df, outcome)['ate']
        
        for gamma in gamma_range:
            adjusted_outcome = df[outcome] + gamma * df['treatment']
            adj_estimate = self.estimate_ate_adjusted(df, adjusted_outcome)
            
            results.append({
                'gamma': gamma,
                'adjusted_ate': adj_estimate,
                'bias': adj_estimate - base_estimate,
            })
        
        return pd.DataFrame(results)
    
    def estimate_ate_adjusted(self, df: pd.DataFrame, adjusted_outcome) -> float:
        """Estimate ATE with pre-computed adjusted outcome."""
        treatment_vals = df['treatment'].values
        
        wt = self.weights * treatment_vals
        wc = self.weights * (1 - treatment_vals)
        
        yt = np.sum(adjusted_outcome * wt) / np.sum(wt)
        yc = np.sum(adjusted_outcome * wc) / np.sum(wc)
        
        return yt - yc
