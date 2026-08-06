import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

class CausalPlotter:
    """Visualization utilities for causal inference methods."""
    
    def __init__(self, figsize: tuple = (14, 10)):
        self.figsize = figsize
    
    def plot_ate_comparison(self, estimates: Dict[str, Dict], true_ate: float = None,
                           save_path: str = None):
        """Compare ATE estimates across all methods."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = list(estimates.keys())
        ate_values = [estimates[m]['ate'] for m in methods]
        ci_lowers = [estimates[m]['ci_lower'] for m in methods]
        ci_uppers = [estimates[m]['ci_upper'] for m in methods]
        
        y_pos = range(len(methods))
        
        errors = [[a - l for a, l in zip(ate_values, ci_lowers)],
                  [u - a for a, u in zip(ate_values, ci_uppers)]]
        
        ax.errorbar(ate_values, y_pos, xerr=errors, fmt='o', capsize=8,
                    capthick=2, markersize=10, color='#2196F3', ecolor='#666')
        
        if true_ate is not None:
            ax.axvline(x=true_ate, color='red', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'True ATE = {true_ate:.2f}')
        
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(methods, fontsize=11)
        ax.set_xlabel('Average Treatment Effect (ATE)', fontsize=12)
        ax.set_title('ATE Estimates: Method Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_propensity_scores(self, propensity_scores: np.ndarray,
                              treatment: np.ndarray, save_path: str = None):
        """Plot propensity score distributions by treatment group."""
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(propensity_scores[treatment == 0], bins=50, alpha=0.6,
                label='Control', color='#2196F3', density=True)
        ax.hist(propensity_scores[treatment == 1], bins=50, alpha=0.6,
                label='Treated', color='#F44336', density=True)
        
        ax.set_xlabel('Propensity Score', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Propensity Score Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_parallel_trends(self, df: pd.DataFrame, outcome: str,
                            treatment: str, time: str, save_path: str = None):
        """Plot parallel trends assumption for DiD."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        treated = df[df[treatment] == 1].groupby(time)[outcome].mean()
        control = df[df[treatment] == 0].groupby(time)[outcome].mean()
        
        ax.plot(treated.index, treated.values, 'o-', color='#F44336',
                label='Treated', linewidth=2, markersize=8)
        ax.plot(control.index, control.values, 's--', color='#2196F3',
                label='Control', linewidth=2, markersize=8)
        
        # Extrapolate counterfactual
        if len(control) > 1:
            slope = control.values[-1] - control.values[0]
            counterfactual = control.values + slope
            ax.plot(treated.index[1:], counterfactual[1:], ':', color='#2196F3',
                    alpha=0.5, label='Counterfactual')
        
        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Outcome', fontsize=12)
        ax.set_title('Parallel Trends Test', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_rdd(self, df: pd.DataFrame, outcome: str, running_var: str,
                cutoff: float = 0.0, bandwidth: float = None, save_path: str = None):
        """Plot RDD with fitted regression lines."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if bandwidth:
            plot_df = df[df[running_var].abs() <= bandwidth].copy()
        else:
            plot_df = df.copy()
        
        ax.scatter(plot_df[running_var], plot_df[outcome], alpha=0.3, s=15,
                  color='#666', label='Observations')
        
        # Fit separate regressions
        above = plot_df[plot_df[running_var] >= cutoff]
        below = plot_df[plot_df[running_var] < cutoff]
        
        if len(above) > 2:
            z = np.polyfit(above[running_var], above[outcome], 1)
            x_line = np.linspace(cutoff, above[running_var].max(), 100)
            ax.plot(x_line, np.polyval(z, x_line), '-', color='#F44336',
                   linewidth=2, label='Right of cutoff')
        
        if len(below) > 2:
            z = np.polyfit(below[running_var], below[outcome], 1)
            x_line = np.linspace(below[running_var].min(), cutoff, 100)
            ax.plot(x_line, np.polyval(z, x_line), '-', color='#2196F3',
                   linewidth=2, label='Left of cutoff')
        
        ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Running Variable', fontsize=12)
        ax.set_ylabel('Outcome', fontsize=12)
        ax.set_title('Regression Discontinuity Design', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_synthetic_control(self, treated_ts: np.ndarray, synthetic_ts: np.ndarray,
                              treatment_start: int, save_path: str = None):
        """Plot synthetic control time series."""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        T = len(treated_ts)
        time = np.arange(T)
        
        ax.plot(time[:treatment_start], treated_ts[:treatment_start], 'o-',
               color='#2196F3', label='Treated Unit', linewidth=2)
        ax.plot(time[:treatment_start], synthetic_ts[:treatment_start], 's--',
               color='#F44336', label='Synthetic Control', linewidth=2)
        
        ax.plot(time[treatment_start:], treated_ts[treatment_start:], 'o-',
               color='#2196F3', linewidth=2)
        ax.plot(time[treatment_start:], synthetic_ts[treatment_start:], 's--',
               color='#F44336', linewidth=2)
        
        ax.axvline(x=treatment_start, color='black', linestyle='--', linewidth=1.5)
        ax.text(treatment_start + 0.5, ax.get_ylim()[0], 'Treatment', fontsize=11,
               rotation=90, verticalalignment='bottom')
        
        # Shade treatment effect
        ax.fill_between(time[treatment_start:], 
                       treated_ts[treatment_start:],
                       synthetic_ts[treatment_start:],
                       alpha=0.2, color='green', label='Treatment Effect')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Outcome', fontsize=12)
        ax.set_title('Synthetic Control Method', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_rosenbaum_bounds(self, bounds_df: pd.DataFrame, save_path: str = None):
        """Plot Rosenbaum bounds sensitivity analysis."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(bounds_df['gamma'], bounds_df['p_value_upper'], 'o-',
               color='#F44336', linewidth=2, label='Upper bound p-value')
        ax.plot(bounds_df['gamma'], bounds_df['p_value_lower'], 's-',
               color='#2196F3', linewidth=2, label='Lower bound p-value')
        
        ax.axhline(y=0.05, color='black', linestyle='--', linewidth=1, alpha=0.7,
                   label='α = 0.05')
        
        ax.fill_between(bounds_df['gamma'], bounds_df['p_value_upper'],
                       where=bounds_df['p_value_upper'] > 0.05,
                       alpha=0.2, color='gray', label='Non-significant region')
        
        ax.set_xlabel('Gamma (Treatment Assignment Odds)', fontsize=12)
        ax.set_ylabel('p-value', fontsize=12)
        ax.set_title('Rosenbaum Bounds Sensitivity Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_covariate_balance(self, balance_df: pd.DataFrame, save_path: str = None):
        """Plot covariate balance before and after matching."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = range(len(balance_df))
        width = 0.35
        
        ax.barh([i - width/2 for i in x], balance_df['std_mean_diff_before'],
               width, label='Before Matching', color='#F44336', alpha=0.7)
        ax.barh([i + width/2 for i in x], balance_df['std_mean_diff_after'],
               width, label='After Matching', color='#2196F3', alpha=0.7)
        
        ax.axvline(x=0.1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=-0.1, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_yticks(list(x))
        ax.set_yticklabels(balance_df['covariate'], fontsize=10)
        ax.set_xlabel('Standardized Mean Difference', fontsize=12)
        ax.set_title('Covariate Balance', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_sensitivity_heatmap(self, sensitivity_df: pd.DataFrame,
                                 save_path: str = None):
        """Plot sensitivity analysis heatmap."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sensitivity_df['gamma'], sensitivity_df['bias'], 'o-',
               color='#9C27B0', linewidth=2)
        ax.fill_between(sensitivity_df['gamma'],
                       sensitivity_df['bias'],
                       alpha=0.3, color='#9C27B0')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Gamma', fontsize=12)
        ax.set_ylabel('Bias', fontsize=12)
        ax.set_title('Sensitivity to Unmeasured Confounding', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_summary_dashboard(self, estimates: Dict[str, Dict],
                                 true_ate: float = None,
                                 save_path: str = None):
        """Create comprehensive summary dashboard."""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. ATE Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        methods = list(estimates.keys())
        ate_vals = [estimates[m]['ate'] for m in methods]
        ci_lowers = [estimates[m]['ci_lower'] for m in methods]
        ci_uppers = [estimates[m]['ci_upper'] for m in methods]
        
        errors = [[a - l for a, l in zip(ate_vals, ci_lowers)],
                  [u - a for a, u in zip(ate_vals, ci_uppers)]]
        
        ax1.errorbar(ate_vals, range(len(methods)), xerr=errors, fmt='o',
                    capsize=6, markersize=8, color='#2196F3')
        if true_ate is not None:
            ax1.axvline(x=true_ate, color='red', linestyle='--', linewidth=2)
        ax1.set_yticks(range(len(methods)))
        ax1.set_yticklabels(methods, fontsize=9)
        ax1.set_title('ATE Comparison', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. CI Width Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ci_widths = [u - l for l, u in zip(ci_lowers, ci_uppers)]
        bars = ax2.barh(range(len(methods)), ci_widths, color='#4CAF50', alpha=0.7)
        ax2.set_yticks(range(len(methods)))
        ax2.set_yticklabels(methods, fontsize=9)
        ax2.set_title('95% CI Width', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Precision Ranking
        ax3 = fig.add_subplot(gs[1, 0])
        se_vals = [estimates[m]['se'] for m in methods]
        sorted_idx = np.argsort(se_vals)
        sorted_methods = [methods[i] for i in sorted_idx]
        sorted_se = [se_vals[i] for i in sorted_idx]
        ax3.barh(range(len(sorted_methods)), sorted_se, color='#FF9800', alpha=0.7)
        ax3.set_yticks(range(len(sorted_methods)))
        ax3.set_yticklabels(sorted_methods, fontsize=9)
        ax3.set_title('Precision (SE) Ranking', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Table Summary
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        table_data = []
        for m in methods:
            e = estimates[m]
            table_data.append([
                m[:20],
                f"{e['ate']:.3f}",
                f"[{e['ci_lower']:.3f}, {e['ci_upper']:.3f}]",
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Method', 'ATE', '95% CI'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary Table', fontweight='bold', pad=20)
        
        plt.suptitle('Causal Inference Methods Comparison Dashboard',
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
