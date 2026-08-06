import numpy as np
import pandas as pd
from typing import Tuple, Dict

class ObservationalDataGenerator:
    """Generate synthetic observational data with known treatment effects for causal method comparison."""
    
    def __init__(self, n: int = 5000, seed: int = 42):
        self.n = n
        self.rng = np.random.default_rng(seed)
    
    def generate(self) -> pd.DataFrame:
        """Generate dataset with treatment assignment mechanism and outcome."""
        # Confounders (affect both treatment and outcome)
        age = self.rng.normal(45, 12, self.n)
        income = self.rng.lognormal(10.5, 0.8, self.n)
        education_years = self.rng.poisson(14, self.n).astype(float)
        pre_score = self.rng.normal(50, 15, self.n)  # pre-treatment outcome
        
        # Time variable (for DiD)
        time_period = self.rng.choice([0, 1], self.n)
        
        # Treatment assignment (depends on confounders - non-random)
        logit_treatment = (
            -2.0
            + 0.03 * (age - 45)
            + 0.0002 * (income - 30000)
            + 0.15 * (education_years - 14)
            + 0.05 * (pre_score - 50)
        )
        prob_treatment = 1 / (1 + np.exp(-logit_treatment))
        treatment = self.rng.binomial(1, prob_treatment)
        
        # True individual treatment effect (heterogeneous)
        true_ite = 5.0 + 2.0 * (age - 45) / 12 + 0.5 * (education_years - 14)
        
        # Outcome (depends on confounders + treatment)
        epsilon = self.rng.normal(0, 5, self.n)
        outcome = (
            100
            + 0.5 * pre_score
            + 0.1 * (age - 45)
            + 0.0005 * (income - 30000)
            + 2.0 * (education_years - 14)
            + true_ite * treatment
            + epsilon
        )
        
        # DiD-specific: potential outcomes by period
        outcome_0_period_0 = outcome
        outcome_1_period_0 = outcome + 3.0  # pre-period level shift
        outcome_0_period_1 = outcome + 1.5  # time trend
        outcome_1_period_1 = outcome + 1.5 + 3.0 + true_ite  # full DiD effect
        
        # RDD-specific: running variable centered at cutoff
        running_variable = self.rng.normal(0, 1, self.n)
        treatment_rdd = (running_variable >= 0).astype(int)
        
        # RDD outcome: sharp discontinuity at cutoff
        rdd_outcome = (
            20
            + 5 * running_variable
            + 8 * treatment_rdd  # true RDD effect = 8
            + 3 * running_variable * treatment_rdd
            + self.rng.normal(0, 2, self.n)
        )
        
        # Synthetic control: treated unit + donor pool
        n_periods = 20
        treatment_start = 15
        n_donors = 20
        
        # Treated unit time series
        treated_ts = np.cumsum(self.rng.normal(0.1, 0.5, n_periods)) + 100
        treated_ts[treatment_start:] += 8.0  # true effect = 8
        
        # Donor pool time series
        donor_ts = np.zeros((n_donors, n_periods))
        for i in range(n_donors):
            donor_ts[i] = np.cumsum(self.rng.normal(0.1, 0.5, n_periods)) + 100 + self.rng.normal(0, 10)
        
        df = pd.DataFrame({
            'age': age,
            'income': income,
            'education_years': education_years,
            'pre_score': pre_score,
            'time_period': time_period,
            'treatment': treatment,
            'outcome': outcome,
            'true_ite': true_ite,
            'running_variable': running_variable,
            'treatment_rdd': treatment_rdd,
            'rdd_outcome': rdd_outcome,
        })
        
        # Store ground truth
        df.attrs['true_ate'] = np.mean(true_ite)
        df.attrs['true_rdd_effect'] = 8.0
        df.attrs['true_ite'] = true_ite
        df.attrs['treated_ts'] = treated_ts
        df.attrs['donor_ts'] = donor_ts
        df.attrs['treatment_start'] = treatment_start
        
        return df
    
    def get_true_ate(self, df: pd.DataFrame) -> float:
        return df.attrs.get('true_ate', np.mean(df.attrs.get('true_ite', [0])))
    
    def get_true_rdd_effect(self, df: pd.DataFrame) -> float:
        return df.attrs.get('true_rdd_effect', 8.0)
