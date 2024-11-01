import argparse
import pandas as pd
import numpy as np

from income.util import *
from income.data import *

TABLE_LABELS = {
    'ipw-lr': 'IPW (LR)',
    'ipw-rfc': 'IPW (RF)',
    'ipww-lr': 'IPW-W (LR)',
    'ipww-rfc': 'IPW-W (RF)',
    'match-nn-eu': 'Match (EU-NN)',
    's-ridge': 'S-learner (Ridge)',
    's-xgbr':  'S-learner (XGB)',
    's-rfr': 'S-learner (RF)',
    't-ridge': 'T-learner (Ridge)',
    't-xgbr': 'T-learner (XGB)',
    't-rfr': 'T-learner (RF)',
    'dml-lin': 'DML (Linear)',
    'dml-rf': 'DML (RF)',
    'dml-xgb': 'DML (XGB)',
    'dml-mix': 'DML (Mix)',
}

NON_CATE = ['ipw-lr', 'ipw-rfc', 'ipww-lr', 'ipww-rfc', 'match-nn-eu']
STRAT_MODELS = ['s-xgbr', 's-rfr', 't-ridge', 't-xgbr', 't-rfr', 'dml-lin', 'dml-rf', 'dml-xgb', 'dml-mix']

def present_results(sim_cfg, est_cfg):
    """ Estimate the causal effect of interventions and evaluate the results
    """

    # Parse estimators and set up parameter grids
    estimators = {}
    est = est_cfg.estimators.__dict__

    results_dir = os.path.join(est_cfg.results.base_path, est_cfg.experiment.label)

    df = pd.DataFrame({})
    for e in est.keys():
        cv_path = os.path.join(results_dir, '%s.%s.cv_results.csv' % (est_cfg.experiment.label, e))
        R = pd.DataFrame({})
        if os.path.isfile(cv_path):
            R = pd.read_csv(cv_path, index_col=0)
            R = R.groupby(['experiment', 'estimator', 'best_params'], as_index=False).mean().drop(columns=['fold', 'best_params'])

        ope_path = os.path.join(results_dir, '%s.%s.ope_results.csv' % (est_cfg.experiment.label, e))
        if os.path.isfile(ope_path):
            Rope = pd.read_csv(ope_path, index_col=0)
            R = pd.merge(R, Rope, on=['experiment', 'estimator'])

        strat_path = os.path.join(results_dir, '%s.%s.strat_results.csv' % (est_cfg.experiment.label, e))
        if os.path.isfile(strat_path):
            Rstrat = pd.read_csv(strat_path, index_col=0)
            R = pd.merge(R, Rstrat, on=['experiment', 'estimator'])

        df = pd.concat([df, R], axis=0)
    
    r_path = os.path.join(results_dir, '%s.results.csv' % (est_cfg.experiment.label))
    df.to_csv(r_path)

    f = open(os.path.join(results_dir, '%s.table_results.tex' % est_cfg.experiment.label), 'w')
    
    log_n_print(f, '# CATE AND FITTING RESULTS')
    for e, l in TABLE_LABELS.items(): 
        if (df['estimator']==e).sum()>0:
            r = df[df['estimator']==e].iloc[0]
            if e.startswith('ipw'):
                if e in NON_CATE:
                    log_n_print(f, '%s & & & %.0f & (%.0f, %.0f) & %.2f \\\\' % (l, r['ATE_AE_r'], r['ATE_AE_r_l'], r['ATE_AE_r_u'], r['test_AUC']))
                else:
                    log_n_print(f, '%s & %.2f & (%.2f, %.2f) & %.0f & (%.0f, %.0f) & %.2f \\\\' % (l, r['CATE_R2_r'], r['CATE_R2_r_l'], r['CATE_R2_r_u'], r['ATE_AE_r'], r['ATE_AE_r_l'], r['ATE_AE_r_u'], r['test_AUC']))
            else:
                if e in NON_CATE:
                    log_n_print(f, '%s &  &  & %.0f & (%.0f, %.0f) & %.2f \\\\' % (l, r['ATE_AE_r'], r['ATE_AE_r_l'], r['ATE_AE_r_u'], r['test_R2']))
                else:
                    log_n_print(f, '%s & %.2f & (%.2f, %.2f) & %.0f & (%.0f, %.0f) & %.2f \\\\' % (l, r['CATE_R2_r'], r['CATE_R2_r_l'], r['CATE_R2_r_u'], r['ATE_AE_r'], r['ATE_AE_r_l'], r['ATE_AE_r_u'], r['test_R2']))


    log_n_print(f, '\n\n# CATE STRATIFICATION RESULTS')
    for e in STRAT_MODELS:
        l = TABLE_LABELS[e]
        if (df['estimator']==e).sum()>0:
            r = df[df['estimator']==e].iloc[0]
            log_n_print(f, '%s & %.2f & (%.2f, %.2f)  \\\\' % (l, r['CATE_strat_R2_r'], r['CATE_strat_R2_r_l'], r['CATE_strat_R2_r_u']))


    # TABLE 1
    # Load data
    D_tr, c_cat, c_num, c_out, c_features = load_income_data(sim_cfg.data.path, download=False)
    D_tr = D_tr.drop(columns=['income', 'studies'])

    D_s = pd.read_pickle(os.path.join(est_cfg.data.path, est_cfg.data.observational))
    D_s['income>50k'] = ((D_s['income_prev']>50000).astype(str)).astype('category')
    D_tr['income>50k'] = (D_tr['income>50k']>0).astype(str).astype('category')

    log_n_print(f, '\n\n# TABLE 1')
    log_n_print(f,' & Simulated ($n=$%d) & Adult ($n=$%d) \\\\' % (D_s.shape[0], D_tr.shape[0]))
    for c in D_s.columns: 
        if D_s[c].dtype == 'category':
            log_n_print(f, '%s \\\\' % c)
            
            for v in D_s[c].unique():
                if c in D_tr.columns: 
                    log_n_print(f, '\;\;\;\;%s & %d (%.1f) & %d (%.1f) \\\\' % (v.replace('&','\\&'), 
                        (D_s[c]==v).sum(), 100*(D_s[c]==v).mean(), 
                        (D_tr[c]==v).sum(), 100*(D_tr[c]==v).mean()))
                else:
                    log_n_print(f, '\;\;\;\;%s & %d (%.1f) & -- \\\\' % (v.replace('&','\\&'), 
                        (D_s[c]==v).sum(), 100*(D_s[c]==v).mean()))
                
        else: 
            lb = np.percentile(D_s[c], 25)
            ub = np.percentile(D_s[c], 75)
            
            if c in D_tr.columns: 
                lb_tr = np.percentile(D_tr[c], 25)
                ub_tr = np.percentile(D_tr[c], 75)
                log_n_print(f, '%s & %.1f (%.1f, %.1f) & %.1f (%.1f, %.1f) \\\\' % (c, D_s[c].mean(), 
                    lb, ub, D_tr[c].mean(), lb_tr, ub_tr))
            else:
                log_n_print(f, '%s & %.1f (%.1f, %.1f) & -- \\\\' % (c, D_s[c].mean(), lb, ub))
            




    f.close()

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Present reuslts from IncomeSim runs')
    parser.add_argument('-sc', '--sim_config', type=str, dest='sim_config', help='Path to estimation config file', default='configs/simulator.yml')
    parser.add_argument('-ec', '--est_config', type=str, dest='est_config', help='Path to simulator config file', default='configs/estimation.yml')
    args = parser.parse_args()

    # Load config file
    sim_cfg = load_config(args.sim_config)
    est_cfg = load_config(args.est_config)

    # Fit simulator
    present_results(sim_cfg, est_cfg)
    

    