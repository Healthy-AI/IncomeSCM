import argparse
import sklearn
import pandas as pd
import numpy as np
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning
import xgboost as xgb

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer, roc_auc_score, mean_squared_error, r2_score, root_mean_squared_error, accuracy_score

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *
from income.estimators import *
from income.evaluation import *

# Load config file
cfg = load_config('configs/estimation_test.yml')

np.random.seed(0)

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

# Fetch numeric features. Other variables (intervention, outcome) will be passed through unchanged
c_num = [k for k in c_cov if df_obs[k].dtype != 'category']
c_cat = [k for k in c_cov if df_obs[k].dtype == 'category']

# Create estimator
e = xgb.XGBRegressor(tree_method='hist', eta=0.1, max_depth=3, random_state=0)
e = S_learner(base_estimator=e, c_int=c_int, c_out=c_out, c_adj=c_cov, v_int0=cfg.experiment.intervention0, v_int1=cfg.experiment.intervention1)

# Create pipeline, with transformation, including the intervention variable
pipe = get_pipeline(e, c_num, c_cat)

# Fit estimator
pipe.fit(df_obs, np.random.rand(df_obs.shape[0])) # @TODO: Don't want to pass around this dummy outcome!

print(pipe.predict(df_obs.iloc[:10]))