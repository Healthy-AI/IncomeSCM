import pandas as pd
import time 

from sklearn.base import BaseEstimator

from .util import *
from .samplers import *
from .income_samplers import *

def sampler_score(estimator):
    if type(estimator) in [LogisticSampler, RandomForestClassifierSampler, IncomeSampler]:
        return 'AUC', estimator.auc_
    elif type(estimator) in [LinearGaussianSampler, GaussianRegressionSampler, GaussianRandomForestSampler, ZeroOrGaussianRegressionSampler]:
        return 'R2', estimator.r2_
    else:
        return None, None

class MarkovARM():
    """
    Markov autoregressive model
    
    """
    
    def __init__(self, transformation=None, sequential_data=False):
        """
        Constructor
        
        Args 
            transformation: Specifies method transforming input at fitting and prediction
            sequential_data (default=False). If set to true, samplers are fit to sequence data. 
                If set to false, samplers are fit to cross-sectional data. 
        """
        self.vars_ = {}
        self.fitted = False
        self.samplers_ = {}
        self.seq_samplers_ = {}
        self.seq_data = sequential_data
        
        self.transf_ = transformation
    
    def add_variable(self, name, parents, sampler, label=None, transform_input=True, seq_transform_input=True,
                     seq_parents_curr=[], seq_parents_prev=[], seq_sampler=None, seq_fit=True):
        """
        Adds a variable to the ARM
        
        Args:
            ... @TODO
        """
        var = {
            'name': name,
            'parents': parents, 
            'sampler': sampler, 
            'transform_input': transform_input, 
            'seq_transform_input': seq_transform_input, 
            'seq_parents_curr': seq_parents_curr, 
            'seq_parents_prev': seq_parents_prev,
            'seq_sampler': seq_sampler,
            'seq_fit': seq_fit, 
            'label': label
        }
        var = Namespace(**var)
        self.vars_[name] = var

        # Check parents
        for p in var.parents:
            if p not in self.vars_:
                warnings.warn('Variable "%s" added without parent "%s"' % (var.name, p))
                
    def replace_variable(self, name, parents, sampler, label=None, transform_input=True, seq_transform_input=True,
                     seq_parents_curr=[], seq_parents_prev=[], seq_sampler=None, seq_fit=True, require_refit=False):
        """
        Replace a variable in the ARM
        
        Args:
            ... @TODO
        """

        if name not in self.vars_:
            raise Exception('No variable with name %s found' % name)
            
        del self.vars_[name]
        
        self.add_variable(name, parents, sampler, label=label, transform_input=transform_input, 
                          seq_transform_input=seq_transform_input, seq_parents_curr=seq_parents_curr, 
                          seq_parents_prev=seq_parents_prev, seq_sampler=seq_sampler, seq_fit=seq_fit)
        
        if require_refit:
            self.fitted = False
        else:
            self.samplers_[name] = sampler
            self.seq_samplers_[name] = seq_sampler
        
    def get_sampler(self, e):
        if isinstance(e, str):
            if e == 'gaussian':
                return GaussianVariable()
            elif e == 'multinomial':
                return MultinomialVariable()
        elif isinstance(e, BaseEstimator):
            return e
        elif isinstance(e, Sampler):
            return e
        elif issubclass(type(e), Sampler):
            return e
        elif isclass(e) and issubclass(e, BaseEstimator):
            return e()
        elif isclass(e) and issubclass(e, Sampler):
            return e()
            
    def compute_order_(self):
        """ Kahn's algorithm """
        parents = dict([(k,v.parents.copy()) for k,v in self.vars_.items()])
        S = [k for k,v in parents.items() if len(v) == 0]
        L = []
        
        while len(S) > 0:
            s = S.pop()
            L.append(s)
            cs = [c for c,v in parents.items() if s in v]
            
            for c in cs:
                parents[c].remove(s)
                if len(parents[c])==0:
                    S.append(c)
        

        return L
    
    def compute_order_seq_(self):
        """ Kahn's algorithm """
        def ps(P, C): 
            return [p + '#prev' for p in P] + C
        parents = dict([(k,ps(v.seq_parents_prev, v.seq_parents_curr)) for k,v in self.vars_.items()]
                      +[(k+'#prev', []) for k in self.vars_.keys()])

        S = [k for k,v in parents.items() if len(v) == 0]
        L = []
        while len(S) > 0:
            s = S.pop()
            L.append(s)
            cs = [c for c,v in parents.items() if s in v]
            for c in cs:
                parents[c].remove(s)
                if len(parents[c])==0:
                    S.append(c)

        L = [l for l in L if not l[-5:] == '#prev']
        return L
        
        
    def fit(self, df, log_file=None):
        if len(self.vars_) == 0:
            raise Exception('At least one variable must be added before fitting')    

        f = None
        if not log_file is None: f = open(log_file, 'w')
            
        for k, v in self.vars_.items():
            Mv = self.get_sampler(v.sampler)

            if isinstance(Mv, Sampler):
                z = df[[v.name]+v.parents].dropna()

                log_n_print(f, 'Fitting variable %s' % k)
                if len(v.parents) < 1:
                    Mv.fit(z[v.name])                    
                else:
                    if v.transform_input:
                        Mv.fit(self.transf_.transform(z[v.parents]), z[v.name])
                    else:
                        Mv.fit(z[v.parents], z[v.name])
                    _sfun, _score = sampler_score(Mv)
                    if not _score is None:
                        log_n_print(f, '  In-sample fitting score: %.4f (%s)' % (_score, _sfun))

            self.samplers_[k] = Mv 
            
            # Fit sequential samplers
            if self.seq_data:
                if not f is None: f.close()
                raise Exception('Fitting to sequential data not implemented yet')
            else:
                Mv_seq = self.get_sampler(v.seq_sampler)
                
                if isinstance(Mv_seq, Sampler) and v.seq_fit:
                    seq_parents = list(set(v.seq_parents_prev + v.seq_parents_curr)) # Only keep one version
                    v_vars = list(set([v.name]+seq_parents))
                                  
                    z = df[v_vars].dropna()
                    
                    if v.seq_transform_input:
                        Mv_seq.fit(self.transf_.transform(z[seq_parents]), z[v.name])
                    else:
                        Mv_seq.fit(z[seq_parents], z[v.name])
                    
                self.seq_samplers_[k] = Mv_seq

        if not f is None: f.close()
            
        self.fitted = True
                
    def sample(self, n_samples, T=1):
        if not self.fitted: 
            raise Exception('Estimator is not yet fitted')
         
        if T<1:
            raise Exception('Can\'t sample less than 1 time point')
        
        # Initialize output
        df = pd.DataFrame({})
        
        # Sampling order 
        O = self.compute_order_()
        
        # Sample first time point
        self.sample_times_ = {}
        for k in O:
            v = self.vars_[k] 
            
            t0 = time.time()
            if len(v.parents) < 1:
                y = self.samplers_[k].sample(n_samples)
            else:
                if v.transform_input:
                    y = self.samplers_[k].sample(self.transf_.transform(df[v.parents]))
                else:
                    y = self.samplers_[k].sample(df[v.parents])
            df[k] = y
            self.sample_times_[k] = time.time() - t0
            
        df['id'] = range(n_samples)
       
        # Sample multiple time points if T>1
        if T>1:
            OS = self.compute_order_seq_()
            
            for t in range(1,T):
                dft = pd.DataFrame({})
                dft['id'] = range(n_samples)
    
                # Sample variables
                for k in OS:
                    v = self.vars_[k]
                                        
                    df_prev = df[df['time']==t-1] # Assume that time is a variable in the sampler
                    
                    df_prev = df_prev.rename(columns=dict([(c,c+'#prev') \
                                   for c in df_prev.columns if c in v.seq_parents_prev + v.seq_parents_curr + [k]]))
                    
                    df_m = pd.merge(df_prev, dft, on='id')

                    parents = [c + '#prev' for c in v.seq_parents_prev] \
                            + [c for c in v.seq_parents_curr]
                    

                    if self.seq_samplers_[k] is None:
                        y = df_prev[k+'#prev'].values.ravel().copy()
                    else:
                        if len(parents) < 1:
                            y = self.seq_samplers_[k].sample(n_samples)
                        else:
                            if v.seq_transform_input:
                                y = self.seq_samplers_[k].sample(self.transf_.transform(df_m[parents]))
                            else:
                                y = self.seq_samplers_[k].sample(df_m[parents])
                                
                    dft[k] = y
                    
                    
                df = pd.concat([df, dft], axis=0)
                
        df = df[['id', 'time']+[c for c in df.columns if not c in ['id','time']]].reset_index(drop=True)
                
        
            
        return df

    def copy(self):
        """
        """
        copy = MarkovARM(transformation=self.transf_, sequential_data=self.seq_data)
        copy.vars_ = self.vars_.copy()
        copy.fitted = self.fitted
        copy.samplers_ = self.samplers_.copy()
        copy.seq_samplers_ = self.seq_samplers_.copy()

        return copy
        