import pandas as pd
import numpy as np
import os
import logging
import json
from typing import Dict, Any

from .generate_data import ObservationalDataGenerator
from .propensity_score_matching import PropensityScoreMatching
from .inverse_probability_weighting import InverseProbabilityWeighting
from .difference_in_differences import DifferenceInDifferences
from .regression_discontinuity import RegressionDiscontinuityDesign
from .synthetic_control import SyntheticControl
from .rosenbaum_bounds import RosenbaumBounds
from .plotting import CausalPlotter

class CausalInferencePipeline:
    """Orchestrate all causal inference method comparisons."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.results = {}
        self.estimates = {}
        self.plotter = CausalPlotter()
    
    def run_psm(self, df: pd.DataFrame, covariates: list,
                treatment: str = 'treatment', outcome: str = 'outcome') -> Dict[str, Any]:
        """Run Propensity Score Matching analysis."""
        self.logger.info("=" * 60)
        self.logger.info("PROPENSITY SCORE MATCHING")
        self.logger.info("=" * 60)
        
        psm = PropensityScoreMatching()
        psm.estimate_propensity_scores(df, covariates, treatment)
        matched_df = psm.match(df, treatment)
        
        ate_result = psm.estimate_ate(matched_df, outcome)
        balance = psm.check_balance(df, covariates, treatment)
        
        # Rosenbaum bounds
        rb = RosenbaumBounds(psm.matched_pairs, matched_df, outcome)
        rb_results = rb.compute_gamma_range()
        critical = rb.critical_gamma()
        
        self.estimates['PSM'] = ate_result
        
        self.results['psm'] = {
            **ate_result,
            'balance': balance.to_dict(),
            'rosenbaum_bounds': rb_results.to_dict(),
            'critical_gamma': critical,
        }
        
        return self.results['psm']
    
    def run_ipw(self, df: pd.DataFrame, covariates: list,
                treatment: str = 'treatment', outcome: str = 'outcome') -> Dict[str, Any]:
        """Run Inverse Probability Weighting analysis."""
        self.logger.info("=" * 60)
        self.logger.info("INVERSE PROBABILITY WEIGHTING")
        self.logger.info("=" * 60)
        
        ipw = InverseProbabilityWeighting()
        ipw.estimate_propensity_scores(df, covariates, treatment)
        ipw.compute_weights(df, treatment, stabilize=True, trim=0.01)
        
        ate_result = ipw.estimate_ate(df, outcome)
        robust_result = ipw.estimate_ate_robust(df, outcome)
        
        self.estimates['IPW'] = ate_result
        self.estimates['IPW (Robust)'] = robust_result
        
        self.results['ipw'] = {
            **ate_result,
            'robust': robust_result,
        }
        
        return self.results['ipw']
    
    def run_did(self, df: pd.DataFrame, outcome: str = 'outcome',
                treatment: str = 'treatment', time: str = 'time_period') -> Dict[str, Any]:
        """Run Difference-in-Differences analysis."""
        self.logger.info("=" * 60)
        self.logger.info("DIFFERENCE-IN-DIFFERENCES")
        self.logger.info("=" * 60)
        
        did = DifferenceInDifferences()
        did_result = did.estimate(df, outcome, treatment, time)
        parallel_trends = did.parallel_trends_test(df, outcome, treatment, time)
        sensitivity = did.sensitivity_to_pre_trends(df, outcome, treatment, time)
        
        self.estimates['DiD'] = did_result
        
        self.results['did'] = {
            **did_result,
            'parallel_trends': parallel_trends,
            'sensitivity': sensitivity.to_dict(),
        }
        
        return self.results['did']
    
    def run_rdd(self, df: pd.DataFrame, outcome: str = 'rdd_outcome',
                running_var: str = 'running_variable',
                cutoff: float = 0.0) -> Dict[str, Any]:
        """Run Regression Discontinuity Design analysis."""
        self.logger.info("=" * 60)
        self.logger.info("REGRESSION DISCONTINUITY DESIGN")
        self.logger.info("=" * 60)
        
        rdd = RegressionDiscontinuityDesign(cutoff=cutoff)
        rdd_result = rdd.estimate_rdd(df, outcome, running_var)
        
        # Bandwidth sensitivity
        bandwidth_sensitivity = rdd.sensitivity_to_bandwidth(df, outcome, running_var)
        
        # Placebo tests
        placebo_tests = rdd.placebo_test(df, outcome, running_var)
        
        self.estimates['RDD'] = rdd_result
        
        self.results['rdd'] = {
            **rdd_result,
            'bandwidth_sensitivity': bandwidth_sensitivity.to_dict(),
            'placebo_tests': placebo_tests.to_dict(),
        }
        
        return self.results['rdd']
    
    def run_synthetic_control(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run Synthetic Control analysis."""
        self.logger.info("=" * 60)
        self.logger.info("SYNTHETIC CONTROL")
        self.logger.info("=" * 60)
        
        treated_ts = df.attrs['treated_ts']
        donor_ts = df.attrs['donor_ts']
        treatment_start = df.attrs['treatment_start']
        
        sc = SyntheticControl()
        sc_result = sc.fit(treated_ts, donor_ts, treatment_start)
        
        # Inference
        inference = sc.inference(treated_ts, donor_ts, treatment_start)
        
        self.estimates['Synthetic Control'] = {
            'ate': sc_result['mean_effect'],
            'se': inference['se'],
            'ci_lower': inference['ci_lower'],
            'ci_upper': inference['ci_upper'],
        }
        
        self.results['synthetic_control'] = {
            **sc_result,
            'inference': inference,
        }
        
        return self.results['synthetic_control']
    
    def run_all_methods(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run all causal inference methods."""
        self.logger.info("=" * 60)
        self.logger.info("CAUSAL INFERENCE METHODS COMPARISON")
        self.logger.info("=" * 60)
        
        covariates = ['age', 'income', 'education_years', 'pre_score']
        
        self.run_psm(df, covariates)
        self.run_ipw(df, covariates)
        self.run_did(df)
        self.run_rdd(df)
        self.run_synthetic_control(df)
        
        self.export_results()
        
        return self.results
    
    def export_results(self, output_dir: str = None):
        """Export all results to files."""
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'outputs')
        
        os.makedirs(output_dir, exist_ok=True)
        
        summary = {
            'method': list(self.estimates.keys()),
            'ate': [self.estimates[m]['ate'] for m in self.estimates],
            'se': [self.estimates[m]['se'] for m in self.estimates],
            'ci_lower': [self.estimates[m]['ci_lower'] for m in self.estimates],
            'ci_upper': [self.estimates[m]['ci_upper'] for m in self.estimates],
        }
        
        with open(os.path.join(output_dir, 'method_comparison.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {output_dir}")
    
    def generate_plots(self, df: pd.DataFrame, output_dir: str = None):
        """Generate all comparison plots."""
        if output_dir is None:
            output_dir = self.config.get('output_dir', 'outputs')
        
        plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        true_ate = df.attrs.get('true_ate', 0)
        
        # ATE comparison
        self.plotter.plot_ate_comparison(
            self.estimates, true_ate,
            os.path.join(plot_dir, 'ate_comparison.png'))
        
        # Covariate balance
        if 'psm' in self.results:
            balance_df = pd.DataFrame(self.results['psm']['balance'])
            self.plotter.plot_covariate_balance(
                balance_df, os.path.join(plot_dir, 'covariate_balance.png'))
        
        # Parallel trends
        self.plotter.plot_parallel_trends(
            df, 'outcome', 'treatment', 'time_period',
            os.path.join(plot_dir, 'parallel_trends.png'))
        
        # RDD
        self.plotter.plot_rdd(
            df, 'rdd_outcome', 'running_variable', 0.0,
            save_path=os.path.join(plot_dir, 'rdd_plot.png'))
        
        # Synthetic control
        sc_results = self.results.get('synthetic_control', {})
        if sc_results:
            treated_ts = df.attrs['treated_ts']
            synthetic_ts = np.array(treated_ts)
            for i, w in enumerate(sc_results.get('weights', [])):
                if i == 0:
                    synthetic_ts = w * df.attrs['donor_ts'][i]
                else:
                    synthetic_ts += w * df.attrs['donor_ts'][i]
            
            self.plotter.plot_synthetic_control(
                treated_ts, synthetic_ts, df.attrs['treatment_start'],
                os.path.join(plot_dir, 'synthetic_control.png'))
        
        # Rosenbaum bounds
        if 'psm' in self.results:
            rb_df = pd.DataFrame(self.results['psm']['rosenbaum_bounds'])
            self.plotter.plot_rosenbaum_bounds(
                rb_df, os.path.join(plot_dir, 'rosenbaum_bounds.png'))
        
        # Dashboard
        self.plotter.create_summary_dashboard(
            self.estimates, true_ate,
            os.path.join(plot_dir, 'dashboard.png'))
        
        self.logger.info(f"Plots saved to {plot_dir}")
