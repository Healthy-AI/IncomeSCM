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
    strat_results = {} # Results for stratified CATE
    strat_cates = {}

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
        strat_result, strat_cate = cate_strat_evaluation(cate_est, df0, df1, c_out, c_strat='education-num', bins=16)
        strat_cate['estimator'] = i
        strat_result['experiment'] = cfg.experiment.label
        strat_result['estimator'] = i
        strat_result = strat_result[['experiment', 'estimator'] + [c for c in strat_result.columns if not c in ['experiment', 'estimator']]]
        r_path = os.path.join(results_dir, '%s.%s.strat_results.csv' % (cfg.experiment.label, i))
        strat_result.to_csv(r_path) 

        # Store results for overview
        ope_results[i] = ope_result
        strat_results[i] = strat_result
        strat_cates[i] = strat_cate
        
    # Create overview and store results
    df_ope_all = pd.concat(ope_results.values(), axis=0)
    df_strat_all = pd.concat(strat_results.values(), axis=0)
    strat_cates = pd.concat(strat_cates.values(), axis=0)

    r_path = os.path.join(results_dir, '%s.ope_results.csv' % (cfg.experiment.label))
    df_ope_all.to_csv(r_path)

    r_path = os.path.join(results_dir, '%s.strat_results.csv' % (cfg.experiment.label))
    df_strat_all.to_csv(r_path)


    # Visualize CATE v strata
    plt.rc('font', family='serif', size=18)
    colors = ['C0']
    markers = ['o','v','^','<','>','p','s','*','d','P']
    plt.figure(figsize=(9,6))
    plt.grid(alpha=0.2)
    divisor = 1000

    # Sample CATE
    plt.fill_between(strat_cate['strata'], strat_cate['cate_sample_l']/divisor, strat_cate['cate_sample_u']/divisor, alpha=0.1, color='k', lw=0)
    plt.plot(strat_cate['strata'], strat_cate['cate_sample_r']/divisor, '--', color='k', marker=markers[0], label='CATE (Sample)', lw=3)

    # All the estimated cates
    i = 1
    estimators_sel = ['s-rfr', 's-xgbr', 't-ridge', 't-rfr', 't-xgbr']
    for e in estimators_sel:
        dfe = strat_cates[strat_cates['estimator']==e]
        plt.fill_between(dfe['strata'], dfe['cate_est_l']/divisor, dfe['cate_est_u']/divisor, alpha=0.1, color='C%d' % i, lw=0)
        plt.plot(dfe['strata'], dfe['cate_est_r']/divisor, color='C%d' % i, marker=markers[i], label=TABLE_LABELS[e], lw=2)
        i += 1

    plt.xlabel('Education (num)')
    plt.ylabel('CATE (\$%d)' % divisor)
    plt.legend()

    path = os.path.join(results_dir, '%s.strat_results.pdf' % (cfg.experiment.label))
    plt.tight_layout()
    plt.savefig(path)
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
    

    