import argparse
import sklearn
import pandas as pd
import numpy as np
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *
from income.estimators import *
from income.evaluation import *

def run_experiment(cfg):
    """ Estimate the causal effect of interventions and evaluate the results
    """

    # Set random seed
    np.random.seed(cfg.experiment.seed)

    # Load data
    print('Loading data ...')
    obs_path = os.path.join(cfg.data.path, cfg.data.observational)
    df_obs = pd.read_pickle(obs_path)

    # Fetch variables
    c_cov = cfg.experiment.covariates
    c_int = cfg.experiment.intervention
    c_out = cfg.experiment.outcome

    # Remove rows that are neither of the main interventions
    df_obs = df_obs[df_obs[c_int].isin([cfg.experiment.intervention0, cfg.experiment.intervention1])]

    # Fetch numeric features
    c_num = [k for k in c_cov if df_obs[k].dtype != 'category']
    c_cat = [k for k in c_cov if df_obs[k].dtype == 'category']

    # Parse estimators and set up parameter grids
    estimators = {}
    est = cfg.estimators.__dict__
    for k,v in est.items():
        param_grid = {('estimator__'+p):a for p,a in v.__dict__.items() if not p in ['label', 'estimator']}
        param_grid['estimator__c_int'] = [c_int]
        param_grid['estimator__c_out'] = [c_out]
        param_grid['estimator__c_adj'] = [c_cov]
        param_grid['estimator__v_int0'] = [cfg.experiment.intervention0]
        param_grid['estimator__v_int1'] = [cfg.experiment.intervention1]
        
        estimators[k] = {'label': v.label, 'estimator': v.estimator, 'param_grid': param_grid}
    
    # Create results dir
    results_dir = os.path.join(cfg.results.base_path, cfg.experiment.label)
    os.makedirs(results_dir, exist_ok=True)

    # Scoring metrics
    scoring = {"R2": make_scorer(r2_score), 
               "RMSE": make_scorer(root_mean_squared_error), 
               "MSE": make_scorer(mean_squared_error)}

    # Fit estimators
    cv_results = {}
    fit_estimators = {}
    ope_results = {}

    for i, v in estimators.items(): 
        label = v['label']
        e = v['estimator']
        param_grid = v['param_grid']

        # Create pipeline, with transformation, including the intervention variable
        pipe = get_pipeline(e, c_num, c_cat+[c_int])

        # Perform cross-validation
        if cfg.selection.type == 'grid':
            cv = GridSearchCV(pipe, param_grid, cv=cfg.selection.folds, refit='R2', scoring=scoring, return_train_score=True)
        elif cfg.selection.type == 'random':
            cv = RandomizedSearchCV(pipe, param_grid, cv=cfg.selection.folds, refit='R2', scoring=scoring, return_train_score=True, n_iter=cfg.selection.n_iter)
        else: 
            raise Exception('Unknown selection type %s' % cfg.selection.type)

        print('Performing cross-validation')
        cv.fit(df_obs[c_cov + [c_int]], df_obs[c_out])

        # Create results data frame
        rows = []
        best_params_ = {k[len('estimator__'):]:v for k,v in cv.best_params_.items() if k.startswith('estimator__')}
        for f in range(cfg.selection.folds):
            
            row = {'experiment': cfg.experiment.label, 'estimator': i, 'fold': f, 'best_params': str(best_params_)}
            for s in scoring.keys():
                for h in ['test', 'train']:
                    k = 'split%d_%s_%s' % (f, h, s)
                    score = cv.cv_results_[k][cv.best_index_]
                    row['%s_%s' % (h, s)] = score
            rows.append(row)
        df_cv = pd.DataFrame(rows)

        # Save results
        r_path = os.path.join(results_dir, '%s.%s.cv_results.csv' % (cfg.experiment.label, i))
        df_cv.to_csv(r_path) 
        
        # Save model
        clf = cv.best_estimator_
        save_model(cv, results_dir, '%s.%s.cv' % (cfg.experiment.label, i))
        save_model(clf, results_dir, '%s.%s.cv' % (cfg.experiment.label, i))
        
        # Do OPE evaluation
        df0 = pd.read_pickle(os.path.join(cfg.data.path, cfg.data.control))
        df1 = pd.read_pickle(os.path.join(cfg.data.path, cfg.data.target))

        ope_result = cate_evaluation(clf, df0, df1, c_cov, c_int, c_out) #pd.DataFrame({}) #off_policy_evaluation(clf, cfg)

        # Store results for overview
        fit_estimators[i] = clf
        cv_results[i] = df_cv
        ope_results[i] = ope_result
        
    # Create overview and store results
    df_cv_all = pd.concat(cv_results.values(), axis=0)
    df_ope_all = pd.concat(ope_results.values(), axis=0)
    
    r_path = os.path.join(results_dir, '%s.cv_results.csv' % (cfg.experiment.label))
    df_cv_all.to_csv(r_path)

    r_path = os.path.join(results_dir, '%s.ope_results.csv' % (cfg.experiment.label))
    df_ope_all.to_csv(r_path)


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Estimate causal effects from IncomeSim samples')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='estimation.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    run_experiment(cfg)
    

    