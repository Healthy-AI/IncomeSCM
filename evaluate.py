import argparse
import sklearn
import pandas as pd
import numpy as np
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error, accuracy_score

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *
from income.estimators import *
from income.evaluation import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

TABLE_LABELS = {
        'ipw-lr': 'IPW (LR)',
        'ipw-rfc': 'IPW (RF)',
        'ipww-lr': 'IPW-W (LR)',
        'ipww-rfc': 'IPW-W (RF)',
        's-ridge': 'S-learner (Ridge)',
        's-xgbr':  'S-learner (XGB)',
        's-rfr': 'S-learner (RF)',
        't-ridge': 'T-learner (Ridge)',
        't-xgbr': 'T-learner (XGB)',
        't-rfr': 'T-learner (RF)'
    }

def evaluate_estimators(cfg):
    """ Estimate the causal effect of interventions and evaluate the results
    """

    # Set random seed
    np.random.seed(cfg.experiment.seed)

    # Load data
    print('Loading data ...')
    df_obs = pd.read_pickle(os.path.join(cfg.data.path, cfg.data.observational))
    df0 = pd.read_pickle(os.path.join(cfg.data.path, cfg.data.control))
    df1 = pd.read_pickle(os.path.join(cfg.data.path, cfg.data.target))

    # Fetch variables
    c_cov = cfg.experiment.covariates
    c_int = cfg.experiment.intervention
    c_out = cfg.experiment.outcome

    # Parse estimators
    estimators = {}
    est = cfg.estimators.__dict__
    for k,v in est.items():
        estimators[k] = {'label': v.label, 'estimator': v.estimator}
    
    # Fetch results dir
    results_dir = os.path.join(cfg.results.base_path, cfg.experiment.label)

    # Results
    ope_results = {}
    hpw_results = {} # Results for hours-per-week subsets
    hpw_cates = {}

    # Evaluate all estimators 
    for i, v in estimators.items(): 
        label = v['label']
        e = v['estimator']
        estimator_type = get_estimator(e)._effect_estimator_type

        print('Evaluating estimator %s...' % label)

        # Save model
        clf = load_model(results_dir, '%s.%s.best' % (cfg.experiment.label, i))
        if clf is None:
            print('Couldn\'t read model file for estimator %s. Skipping.' % e )
            continue            
        
        # Perform CATE evaluation
        ope_result, cate_est = cate_evaluation(clf, df0, df1, c_cov, c_int, c_out, return_cate_estimate=True)
        ope_result['experiment'] = cfg.experiment.label
        ope_result['estimator'] = i
        ope_result = ope_result[['experiment', 'estimator'] + [c for c in ope_result.columns if not c in ['experiment', 'estimator']]]
        r_path = os.path.join(results_dir, '%s.%s.ope_results.csv' % (cfg.experiment.label, i))
        ope_result.to_csv(r_path) 

        # Evaluate CATE for stratified by hours-per-week
        hpw_result, hpw_cate = cate_hpw_evaluation(cate_est, df0, df1, c_out)
        hpw_cate['estimator'] = i
        hpw_result['experiment'] = cfg.experiment.label
        hpw_result['estimator'] = i
        hpw_result = hpw_result[['experiment', 'estimator'] + [c for c in hpw_result.columns if not c in ['experiment', 'estimator']]]
        r_path = os.path.join(results_dir, '%s.%s.hpw_results.csv' % (cfg.experiment.label, i))
        hpw_result.to_csv(r_path) 

        # Store results for overview
        ope_results[i] = ope_result
        hpw_results[i] = hpw_result
        hpw_cates[i] = hpw_cate
        
    # Create overview and store results
    df_ope_all = pd.concat(ope_results.values(), axis=0)
    df_hpw_all = pd.concat(hpw_results.values(), axis=0)
    hpw_cates = pd.concat(hpw_cates.values(), axis=0)

    r_path = os.path.join(results_dir, '%s.ope_results.csv' % (cfg.experiment.label))
    df_ope_all.to_csv(r_path)

    r_path = os.path.join(results_dir, '%s.hpw_results.csv' % (cfg.experiment.label))
    df_hpw_all.to_csv(r_path)


    # Visualize CATE v hpw
    plt.rc('font', family='serif', size=18)
    colors = ['C0']
    markers = ['o','v','^','<','>','p','s','*','d','P']
    plt.figure(figsize=(9,6))
    plt.grid(alpha=0.2)
    divisor = 1000

    # Sample CATE
    plt.fill_between(hpw_cate['hpw'], hpw_cate['cate_sample_l']/divisor, hpw_cate['cate_sample_u']/divisor, alpha=0.1, color='k', lw=0)
    plt.plot(hpw_cate['hpw'], hpw_cate['cate_sample_r']/divisor, '--', color='k', marker=markers[0], label='CATE (Sample)', lw=3)

    # All the estimated cates
    i = 1
    estimators_sel = ['s-rfr', 's-xgbr', 't-ridge', 't-rfr', 't-xgbr']
    for e in estimators_sel:
        dfe = hpw_cates[hpw_cates['estimator']==e]
        plt.fill_between(dfe['hpw'], dfe['cate_est_l']/divisor, dfe['cate_est_u']/divisor, alpha=0.1, color='C%d' % i, lw=0)
        plt.plot(dfe['hpw'], dfe['cate_est_r']/divisor, color='C%d' % i, marker=markers[i], label=TABLE_LABELS[e], lw=2)
        i += 1

    plt.xlabel('Hours per week')
    plt.ylabel('CATE (\$%d)' % divisor)
    plt.legend()

    path = os.path.join(results_dir, '%s.hpw_results.pdf' % (cfg.experiment.label))
    plt.tight_layout()
    plt.savefig(path)
    os.system('pdfcrop %s %s' % (path, path))
    plt.close()


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate estimated causal effects of fitted simulators')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='configs/estimation.yml')
    args = parser.parse_args()


    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    evaluate_estimators(cfg)
    

    