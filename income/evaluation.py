import numpy as np
import pandas as pd

from sklearn.utils import resample
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error


EFFECT_SCORING = {"R2": r2_score, 
            "RMSE": root_mean_squared_error, 
            "MSE": mean_squared_error}

def cate_evaluation(clf, df0, df1, c_cov, c_int, c_out, n_bootstrap=1000, alpha=0.05, return_cate_estimate=False):

    n = df0.shape[0]

    # Check that df0 and df1 are aligned by sampling random rows and check that they are the same
    for i in np.random.choice(n, 10):
        if np.any(df0.iloc[i][c_cov] != df1.iloc[i][c_cov]):
            print(df0.iloc[i][c_cov] != df1.iloc[i][c_cov])
            raise Exception('Dataframes for control and treated counterfactuals not matched')
    
    # Transform data for estimator
    dft0 = clf[:-1].transform(df0)
    dft1 = clf[:-1].transform(df1)

    # Predict outcomes using estimator
    y0 = clf[-1].predict_outcomes(dft0)
    y1 = clf[-1].predict_outcomes(dft1)

    # Estimate conditional and average effects
    cate_est = y1-y0
    cate_sample = df1[c_out]-df0[c_out]
    ate_est = cate_est.mean()
    ate_sample = cate_sample.mean()

    df = pd.DataFrame({'cate_est': cate_est, 'cate_sample': cate_sample})
    
    rows = []
    for i in range(n_bootstrap):
        dfr = resample(df, n_samples=df.shape[0], random_state=i)

        ate_est_r    = dfr['cate_est'].mean()
        ate_sample_r = dfr['cate_sample'].mean()

        # Cate measures
        row = { ('CATE_%s_r' % k):s(dfr['cate_sample'], dfr['cate_est']) for k, s in EFFECT_SCORING.items() }
        row['ATE_AE_r'] = np.abs(ate_est_r - ate_sample_r)
        row['ATE_SE_r'] = np.abs(ate_est_r - ate_sample_r)**2

        rows.append(row)

    Ra = pd.DataFrame(rows)
    R = Ra.mean()
    for c in Ra.columns: 
        R[c+'_l'] = np.percentile(Ra[c], alpha/2*100)
        R[c+'_u'] = np.percentile(Ra[c], (1-alpha/2) * 100)
        
    for k, s in EFFECT_SCORING.items():
        R['CATE_'+k] = s(cate_sample, cate_est)
        
    R['ATE_AE'] = np.abs(ate_est-ate_sample)
    R['ATE_SE'] = np.abs(ate_est-ate_sample)**2
    R['ATE_True'] = ate_sample

    R = R.to_frame().T
    R = R[np.sort(R.columns)]

    if return_cate_estimate: 
        return R, cate_est
    else:
        return R

def cate_hpw_evaluation(cate_est, df0, df1, c_out, n_bootstrap=1000, alpha=0.05):
    """ Evaluate CATE stratified by hours-per-week
    """

    df = pd.DataFrame({'hpw': df0['hours-per-week'], 'cate_sample': df1[c_out] - df0[c_out], 'cate_est': cate_est})
    
    bins = [i for i in np.linspace(0, df['hpw'].max(),11)] # max hpw is 100
    cut = pd.cut(df['hpw'], bins=bins, include_lowest=True)

    # Group by bins of hpw
    dfm = df.groupby(cut, as_index=False).mean()
    dfm['hpw'] = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    cate_est = dfm['cate_est']
    cate_sample = dfm['cate_sample']

    rows = []
    cates = []
    for i in range(n_bootstrap):
        dfr = resample(df, n_samples=df.shape[0], random_state=i)
        dfmr = dfr.groupby(pd.cut(dfr['hpw'], bins=bins, include_lowest=True), as_index=False).mean()
        dfmr['hpw'] = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]

        # Cate measures
        row = { ('CATE_hpw_%s_r' % k):s(dfmr['cate_sample'], dfmr['cate_est']) for k, s in EFFECT_SCORING.items() }
        rows.append(row)

        # Cate estimates and "truth"
        cates.append(dfmr)

    # Scoring results
    Ra = pd.DataFrame(rows)
    R = Ra.mean()
    for c in Ra.columns: 
        R[c+'_l'] = np.percentile(Ra[c], alpha/2*100)
        R[c+'_u'] = np.percentile(Ra[c], (1-alpha/2) * 100)
        
    for k, s in EFFECT_SCORING.items():
        R['CATE_'+k] = s(cate_sample, cate_est)

    # Bootstrapped cate estimates
    cates = pd.concat(cates, axis=0)
    cut = pd.cut(cates['hpw'], bins=bins, include_lowest=True)
    
    cates_l = cates.groupby(cut, as_index=False).quantile(q=alpha/2)
    cates_r = cates.groupby(cut, as_index=False).mean()
    cates_u = cates.groupby(cut, as_index=False).quantile(q=1-alpha/2)

    cates = pd.DataFrame({
        'hpw': dfm['hpw'],
        'cate_sample_l': cates_l['cate_sample'],
        'cate_sample_r': cates_r['cate_sample'],
        'cate_sample_u': cates_u['cate_sample'],
        'cate_est_l': cates_l['cate_est'],
        'cate_est_r': cates_r['cate_est'],
        'cate_est_u': cates_u['cate_est'],
    })
    return R.to_frame().T, cates

    



