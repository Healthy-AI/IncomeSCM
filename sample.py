import argparse
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.income import *

def sample(cfg):
    """ Samples from a stored model with a given set of target policies
    """

    policies = cfg.samples.policy
    if type(policies) == str:
        policies = [policies]

    # Sample from all target policies _with the same starting state (due to the seed)_
    for pol in policies: 
    
        # Load simulator
        print('Loading model ...')
        A = load_model(cfg.simulator.path, cfg.simulator.label)
        
        # Set propensity model
        if pol in ['no', 'full']:
            if pol == 'no':
                policy = 'No studies'
            elif pol == 'full':
                policy = 'Full-time studies'
            s = StudiesSampler()
            A.replace_variable('studies', ['age', 'sex', 'education','education-num', 'relationship'], StudiesSampler(), transform_input=False, 
                        seq_sampler=StudiesTransition(s, intervention={'T': 1, 'action': policy}), 
                        seq_parents_curr=['age', 'sex', 'education', 'education-num', 'relationship', 'time'], 
                        seq_parents_prev=['studies','income'], 
                        seq_transform_input=False)
                        #ConstantSampler(policy) --- replaces first time step. Need for income
        elif pol == 'default':
            pass
        else:
            raise Exception('Unknown sampling policy \'%s\'. Aborting.' % pol)

        # Sample observations with the same starting seed for all policies (counterfactuals)
        np.random.seed(cfg.samples.seed)
        print('Sampling observations ...')
        S = A.sample(cfg.samples.n_samples, T=(cfg.samples.horizon+1)) # Adding 1 since throwing away first time step

        # Prep data
        df0 = S[S['time']==0] # To generate income without studies
        df1 = S[S['time']==1] # To generate all the other variables, and the studies indicator
        df = df1.copy().rename(columns={'income': 'income_prev'})
        df['income_prev'] = df0['income'].values
        df['studies_prev'] = df0['studies'].values
        
        # Possible solutions to studies_prev being a confounder. Trying #4 first
        #
        # 1. Go back to not having current income among the covariates
        # 2. Make StudiesTransition not depend on previous studies at the time of intervention
        # 3. Create a separate income variable for the first time step that doesn't depend on studies
        # 4. Add studies_prev to the adjustment set
        
        # Get the income from the last time point as the outcome variable
        Tend = cfg.samples.horizon-1
        df['income'] = S[S['time']==Tend]['income'].values
        
        # Make categorical columns have the right type
        c_cols = ['native-country', 'sex', 'race', 'education', 
                'studies', 'workclass', 'occupation', 'marital-status', 'relationship', 'studies_prev']
        df[c_cols] = df[c_cols].astype('category')

        # Drop index columns
        df = df.drop(columns=['time','id'])
        
        # Reorder columns
        special_cols = ['studies', 'income']
        df = df[[c for c in df.columns if c not in special_cols] + special_cols]
        
        # Save data to file
        fname = '%s_%s_n%d_T%d_s%d.pkl' % (cfg.samples.label, pol, cfg.samples.n_samples, cfg.samples.horizon, cfg.samples.seed)
        fpath = os.path.join(cfg.samples.path, fname)
        df.to_pickle(fpath)
        print('Saved result to: %s' % fpath)


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Sample from IncomeSCM simulator')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='config_v1.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    sample(cfg)
    

    