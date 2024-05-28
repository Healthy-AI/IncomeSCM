import argparse
import sklearn
sklearn.set_config(transform_output="pandas")
from sklearn.exceptions import ConvergenceWarning

from income.data import *
from income.util import *
from income.samplers import *
from income.income_samplers import *
from income.arm import *
from income.simulator import *

def fit_simulator(cfg):

    # Load data
    D_tr, c_cat, c_num, c_out, c_features = load_income_data(cfg.data.path, download=cfg.data.download)

    # Set random seed
    np.random.seed(cfg.simulator.seed)
    
    # Fit column transformer
    print('Fitting column transformer ...')
    T = SubsetTransformer()
    D_prev = D_tr.copy().rename(columns=dict([(c,c+'#prev') for c in D_tr.columns if not c == 'id']))
    D_tran = pd.merge(D_tr, D_prev, on='id')    
    D_tran['income#prev'] = np.random.randn(D_tran.shape[0])    # Ensure that we avoid transforming income#prev
    T.fit(D_tran)
    
    # Create ARM model
    print('Initializing ARM model ...')
    A = create_simulator(cfg, T)

    # Fit to observed data 
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    print('Fitting ARM model (suppressing warnings) ...')
    A.fit(D_tr)

    # Save the simulator
    save_model(A, cfg.simulator.path, cfg.simulator.label)

if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Fit IncomeSCM simulator')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='configs/simulator.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    fit_simulator(cfg)
    

    