import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any

class PropensityScoreMatching:
    """Propensity Score Matching for causal effect estimation."""
    
    def __init__(self):
        self.propensity_model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.propensity_scores = None
        self.matched_pairs = None
    
    def estimate_propensity_scores(self, df: pd.DataFrame, 
                                   covariates: list, treatment: str) -> np.ndarray:
        """Estimate propensity scores using logistic regression."""
        X = self.scaler.fit_transform(df[covariates])
        y = df[treatment].values
        
        self.propensity_model.fit(X, y)
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        return self.propensity_scores
    
    def match(self, df: pd.DataFrame, treatment: str, 
              caliper: float = 0.05) -> pd.DataFrame:
        """Perform nearest-neighbor matching with caliper."""
        ps = self.propensity_scores
        treated_idx = df[df[treatment] == 1].index.tolist()
        control_idx = df[df[treatment] == 0].index.tolist()
        
        matched_treated = []
        matched_control = []
        used_controls = set()
        
        for t_idx in treated_idx:
            t_score = ps[t_idx]
            best_dist = float('inf')
            best_c_idx = None
            
            for c_idx in control_idx:
                if c_idx in used_controls:
                    continue
                dist = abs(t_score - ps[c_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_c_idx = c_idx
            
            if best_c_idx is not None and best_dist <= caliper:
                matched_treated.append(t_idx)
                matched_control.append(best_c_idx)
                used_controls.add(best_c_idx)
        
        self.matched_pairs = list(zip(matched_treated, matched_control))
        
        matched_df = pd.concat([
            df.loc[matched_treated].assign(match_role='treated'),
            df.loc[matched_control].assign(match_role='control')
        ]).sort_index()
        
        return matched_df
    
    def estimate_ate(self, df: pd.DataFrame, outcome: str) -> Dict[str, Any]:
        """Estimate Average Treatment Effect from matched pairs."""
        if self.matched_pairs is None:
            raise ValueError("Must call match() before estimate_ate()")
        
        treated_outcomes = [df.loc[t, outcome] for t, c in self.matched_pairs]
        control_outcomes = [df.loc[c, outcome] for t, c in self.matched_pairs]
        
        ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
        
        se = np.sqrt(
            np.var(treated_outcomes) / len(treated_outcomes) +
            np.var(control_outcomes) / len(control_outcomes)
        )
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        return {
            'method': 'Propensity Score Matching',
            'ate': float(ate),
            'se': float(se),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_matched': len(self.matched_pairs),
            'n_treated': len([p for p in self.matched_pairs]),
        }
    
    def check_balance(self, df: pd.DataFrame, covariates: list, treatment: str) -> pd.DataFrame:
        """Check covariate balance before and after matching."""
        results = []
        
        for cov in covariates:
            treated_all = df[df[treatment] == 1][cov]
            control_all = df[df[treatment] == 0][cov]
            
            std_mean_diff_before = (
                (treated_all.mean() - control_all.mean()) / 
                np.sqrt((treated_all.var() + control_all.var()) / 2)
            )
            
            if self.matched_pairs:
                treated_matched = [df.loc[t, cov] for t, c in self.matched_pairs]
                control_matched = [df.loc[c, cov] for t, c in self.matched_pairs]
                
                std_mean_diff_after = (
                    (np.mean(treated_matched) - np.mean(control_matched)) /
                    np.sqrt((np.var(treated_matched) + np.var(control_matched)) / 2)
                )
            else:
                std_mean_diff_after = np.nan
            
            results.append({
                'covariate': cov,
                'std_mean_diff_before': float(std_mean_diff_before),
                'std_mean_diff_after': float(std_mean_diff_after),
                'balanced_after': abs(std_mean_diff_after) < 0.1 if not np.isnan(std_mean_diff_after) else False,
            })
        
        return pd.DataFrame(results)
