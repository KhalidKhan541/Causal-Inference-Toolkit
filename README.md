# Causal Inference Methods Comparison Toolkit

Implementation of 5 causal inference methods on the same observational dataset with DoWhy, EconML, and statsmodels.

## Methods

| Method | Use Case | Key Assumption |
|--------|----------|----------------|
| **PSM** | Observational studies | Conditional independence |
| **IPW** | Observational studies | Correctly specified propensity model |
| **DiD** | Policy evaluation | Parallel trends |
| **RDD** | Threshold-based treatment | Continuity at cutoff |
| **Synthetic Control** | Single unit treatment | Donor pool contains good controls |

## Key Features

- **Same dataset**: All methods applied to identical data for fair comparison
- **Ground truth**: Known true ATE for evaluating method accuracy
- **Sensitivity analysis**: Rosenbaum bounds for PSM, bandwidth sensitivity for RDD
- **Parallel trends test**: Formal test for DiD assumption
- **Placebo tests**: Synthetic control and RDD placebo tests
- **Covariate balance**: Before/after matching comparison

## Quick Start

```bash
pip install -r requirements.txt

# Run all methods
python -m analysis.run --methods all

# Run specific method
python -m analysis.run --methods psm

# Without plots
python -m analysis.run --methods all --no-plots
```

## Architecture

```
Causal-Inference-Toolkit/
├── src/
│   ├── generate_data.py              # Synthetic data generator
│   ├── propensity_score_matching.py  # PSM with Rosenbaum bounds
│   ├── inverse_probability_weighting.py  # IPW with robust SE
│   ├── difference_in_differences.py  # DiD with parallel trends
│   ├── regression_discontinuity.py   # RDD with bandwidth sensitivity
│   ├── synthetic_control.py          # Synthetic Control with placebo
│   ├── rosenbaum_bounds.py           # Sensitivity analysis
│   ├── plotting.py                   # Visualization
│   └── pipeline.py                   # Orchestration
├── analysis/
│   ├── configs/
│   │   └── default.yaml
│   └── run.py                        # CLI
├── requirements.txt
└── README.md
```

## Dependencies

- `dowhy` - Causal inference framework
- `econml` - EconML causal ML
- `statsmodels` - OLS/IV regression
- `scikit-learn` - Propensity score estimation
- `scipy` - Optimization for synthetic control
