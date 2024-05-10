import argparse
import pandas as pd
import numpy as np

from income.util import *


def present_results(cfg):
    """ Estimate the causal effect of interventions and evaluate the results
    """

    # Parse estimators and set up parameter grids
    estimators = {}
    est = cfg.estimators.__dict__

    results_dir = os.path.join(cfg.results.base_path, cfg.experiment.label)

    df = pd.DataFrame({})
    for e in est.keys():
        cv_path = os.path.join(results_dir, '%s.%s.cv_results.csv' % (cfg.experiment.label, e))
        R = pd.DataFrame({})
        if os.path.isfile(cv_path):
            R = pd.read_csv(cv_path, index_col=0)
            R = R.groupby(['experiment', 'estimator', 'best_params'], as_index=False).mean().drop(columns=['fold', 'best_params'])

        ope_path = os.path.join(results_dir, '%s.%s.ope_results.csv' % (cfg.experiment.label, e))
        if os.path.isfile(cv_path):
            Rope = pd.read_csv(ope_path, index_col=0)
            R = pd.merge(R, Rope, on=['experiment', 'estimator'])
        df = pd.concat([df, R], axis=0)
    
    r_path = os.path.join(results_dir, '%s.results.csv' % (cfg.experiment.label))
    df.to_csv(r_path)

    TABLE_LABELS = {
        's-ridge': 'S-learner (Ridge)',
        's-xgbr':  'S-learner (XGB)',
        's-rfr': 'S-learner (RF)',
        't-ridge': 'T-learner (Ridge)',
        't-xgbr': 'T-learner (XGB)',
        't-rfr': 'T-learner (RF)'
    }
    
    for e, l in TABLE_LABELS.items(): 
        if (df['estimator']==e).sum()>0:
            r = df[df['estimator']==e].iloc[0]
            print('%s & %.2f & (%.2f, %.2f) & %.0f & (%.0f, %.0f) & %.2f \\\\' % (l, r['CATE_R2_r'], r['CATE_R2_r_l'], r['CATE_R2_r_u'], r['ATE_AE_r'], r['ATE_AE_r_l'], r['ATE_AE_r_u'], r['test_R2']))

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Present reuslts from IncomeSim runs')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='estimation.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    present_results(cfg)
    

    