import argparse
import yaml
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description='Causal Inference Methods Comparison')
    parser.add_argument('--config', default='configs/default.yaml', help='Config file')
    parser.add_argument('--methods', nargs='+',
                       choices=['psm', 'ipw', 'did', 'rdd', 'synthetic', 'all'],
                       default=['all'], help='Methods to run')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--output', default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    config_path = args.config
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    config['output_dir'] = args.output
    
    from src.pipeline import CausalInferencePipeline
    from src.generate_data import ObservationalDataGenerator
    
    # Generate data
    data_config = config.get('data', {})
    generator = ObservationalDataGenerator(
        n=data_config.get('n', 5000),
        seed=config.get('seed', 42)
    )
    df = generator.generate()
    
    pipeline = CausalInferencePipeline(config)
    
    if 'all' in args.methods:
        pipeline.run_all_methods(df)
    else:
        covariates = ['age', 'income', 'education_years', 'pre_score']
        
        if 'psm' in args.methods:
            pipeline.run_psm(df, covariates)
        if 'ipw' in args.methods:
            pipeline.run_ipw(df, covariates)
        if 'did' in args.methods:
            pipeline.run_did(df)
        if 'rdd' in args.methods:
            pipeline.run_rdd(df)
        if 'synthetic' in args.methods:
            pipeline.run_synthetic_control(df)
        
        pipeline.export_results()
    
    if not args.no_plots:
        pipeline.generate_plots(df)
    
    print("\nAnalysis complete! Check outputs/ for results and plots.")

if __name__ == '__main__':
    main()
