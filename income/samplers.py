import pandas as pd
import numpy as np
import inspect 

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.base import BaseEstimator

from inspect import isclass
from argparse import Namespace
import warnings

class Sampler():
    def __init__(self, **kwargs):
        """
        Initializes the sampler
        """
        pass
    
    def fit(self, x, y=None, **kwargs):
        """
        Fits sampler parameters.
        Does not need implementation if not necessary
        """
        self.fitted=True
        return self
    
    def sample(self):
        """
        Samples observations
        """
        raise Exception('Sample mechanism not implemented')

class ConstantSampler(Sampler):
    def __init__(self, value, **kwargs):
        """
        Initializes the sampler
        """
        self.value = value

    def sample(self, n_samples):
        """
        Samples constant observations
        """
        y = np.array([self.value]*n_samples).ravel()
        return y
        
    
class GaussianVariable(BaseEstimator, Sampler):
    
    def __init__(self, round_result=False, bounds=None):
        self.fitted = False
        self.round_result = round_result
        self.bounds = bounds
    
    def fit(self, x):
        self.mean_ = x.mean()
        self.std_ = x.std()
        
        self.fitted = True
        
        return self
        
    def sample(self, n_samples=1, random_state=None):
        if self.fitted:
            
            if self.bounds is None:
                y = self.mean_ + np.random.randn(n_samples)*self.std_
                if self.round_result: 
                    y = np.round(y)
            else:
                y = []
                i = 0
                while len(y) < n_samples:
                    yp = self.mean_ + np.random.randn(n_samples)*self.std_
                    if self.round_result: 
                        yp = np.round(yp)
                        
                    yp = yp[(yp>=self.bounds[0])&(yp<=self.bounds[1])]
                    
                    y = np.concatenate([y, yp])
                    i += 1
                    
                    if i>100:
                        break
                        
                y = y[:n_samples]

            return y
        else:
            raise Exception('Estimator is not yet fitted')
            
class MultinomialVariable(BaseEstimator, Sampler):
    def __init__(self):
        self.fitted = False
    
    def fit(self, x):
        self.labels_ = sorted(x.unique())
        self.p = [(x==self.labels_[i]).sum()/x.shape[0] for i in range(len(self.labels_))]
        
        self.fitted = True
    
        return self
    
    def sample(self, n_samples=1, random_state=None):
        if self.fitted:
            return np.random.choice(self.labels_, size=(n_samples), p=self.p)
        else:
            raise Exception('Estimator is not yet fitted')          
            
class BernoulliVariable(BaseEstimator, Sampler):
    def __init__(self):
        self.fitted = False
    
    def fit(self, x):
        self.mean_ = x.mean()
        
        self.fitted = True
        
        return self
        
    def sample(self, n_samples=1, random_state=None):
        if self.fitted:
            return 1*(np.random.rand(n_samples) < self.mean_)
        else:
            raise Exception('Estimator is not yet fitted')
            
class LogisticSampler(LogisticRegression, Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted = False
        
    def sample(self, x):
        p = self.predict_proba(x)
        y = np.array([np.random.choice(self.classes_, p=p[i,:]) for i in range(p.shape[0])])
        return y
    
    def fit(self, x, y, **kwargs):
        super().fit(x, y, **kwargs)

        self.auc_ = roc_auc_score(y, self.predict_proba(x), multi_class='ovr')

        self.fitted = True
        
        return self
    
class RandomForestClassifierSampler(RandomForestClassifier, Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitted = False
        
    def sample(self, x):
        p = self.predict_proba(x)
        y = np.array([np.random.choice(self.classes_, p=p[i,:]) for i in range(p.shape[0])])
        return y
    
    def fit(self, x, y, **kwargs):
        super().fit(x, y, **kwargs)
        
        self.auc_ = roc_auc_score(y, self.predict_proba(x), multi_class='ovr')

        self.fitted = True
        
        return self
        
class LinearGaussianSampler(LinearRegression, Sampler):
    
    def __init__(self, round_result=False, bounds=None, **kwargs):
        super().__init__(**kwargs)
        self.fitted = False
        self.round_result = round_result
        self.bounds = bounds
        
    def sample(self, x):
        y = self.predict(x) + np.random.randn(x.shape[0])*np.sqrt(self.mse_)
        
        if not self.bounds is None:
            y = np.clip(y, self.bounds[0], self.bounds[1])

        if self.round_result:
            y = np.round(y)
        
        return y
    
    def fit(self, x, y, **kwargs):
        super().fit(x, y, **kwargs)
        
        self.mse_ = mean_squared_error(y, self.predict(x))
        self.r2_ = mean_squared_error(y, self.predict(x))
        
        self.fitted = True
        
        return self
    
    
class GaussianRegressionSampler(BaseEstimator, Sampler):
    
    def __init__(self, estimator, round_result=False, bounds=None, std_mod=1., **kwargs):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.fitted = False
        self.round_result = round_result
        self.bounds = bounds
        self.std_mod = std_mod
        
    def sample(self, x):
        y = self.estimator.predict(x) + np.random.randn(x.shape[0])*np.sqrt(self.mse_)*self.std_mod
        
        if not self.bounds is None:
            y = np.clip(y, self.bounds[0], self.bounds[1])

        if self.round_result:
            y = np.round(y)
        
        return y
    
    def fit(self, x, y, **kwargs):
        self.estimator.fit(x, y, **kwargs)
        
        self.mse_ = mean_squared_error(y, self.estimator.predict(x))
        self.r2_ = r2_score(y, self.estimator.predict(x))
        
        self.fitted = True
        
        return self


class GaussianRandomForestSampler(GaussianRegressionSampler):
    
    def __init__(self, round_result=False, bounds=None, std_mod=1., **kwargs):

        rf_args = inspect.signature(RandomForestRegressor.__init__).parameters.keys()
        args = {k:v for k,v in kwargs.items() if k in rf_args}
        estimator = RandomForestRegressor(**args)

        super().__init__(estimator=estimator, round_result=round_result, bounds=bounds, std_mod=std_mod, **kwargs)

    
class ZeroOrGaussianRegressionSampler(BaseEstimator, Sampler):
    
    def __init__(self, zero_estimator, estimator, round_result=False, bounds=None, std_mod=1., **kwargs):
        super().__init__(**kwargs)

        if type(zero_estimator) == str: 
            if zero_estimator == 'LogisticRegression':
                zero_estimator = LogisticRegression()
            else: 
                raise Exception('Unknowon classifier \'%s\'. Aborting' % zero_estimator)
        if type(estimator) == str: 
            if estimator == 'Ridge':
                estimator = Ridge()
            else: 
                raise Exception('Unknowon classifier \'%s\'. Aborting' % estimator)

        self.zero_estimator = zero_estimator
        self.estimator = estimator
        self.fitted = False
        self.round_result = round_result
        self.bounds = bounds
        self.std_mod = std_mod
        
    def sample(self, x, zero_prob_=None):
        n = x.shape[0]

        z = np.random.rand(n) < self.zero_estimator.predict_proba(x)[:,1]
        y = self.estimator.predict(x) + np.random.randn(n)*np.sqrt(self.mse_)*self.std_mod
        
        y = (1-z)*y
        
        if not self.bounds is None:
            y = np.clip(y, self.bounds[0], self.bounds[1])

        if self.round_result:
            y = np.round(y)
        
        return y
    
    def fit(self, x, y, **kwargs):
        self.zero_estimator.fit(x, y==0, **kwargs)
        self.estimator.fit(x[y != 0], y[y != 0], **kwargs)
        
        self.mse_ = mean_squared_error(y[y != 0], self.estimator.predict(x[y != 0]))
        self.r2_ = r2_score(y[y != 0], self.estimator.predict(x[y != 0]))
        
        self.fitted = True
        
        return self

    
class UniformRegressionSampler(BaseEstimator, Sampler):
    
    def __init__(self, estimator, round_result=False, bounds=None, **kwargs):
        super().__init__(**kwargs)

        self.estimator = estimator
        self.fitted = False
        self.round_result = round_result
        self.bounds = bounds
        
    def sample(self, x):
        y = self.estimator.predict(x) + np.random.rand(x.shape[0])*self.w - self.w/2
        
        if not self.bounds is None:
            y = np.clip(y, self.bounds[0], self.bounds[1])

        if self.round_result:
            y = np.round(y)
        
        return y
    
    def fit(self, x, y, **kwargs):
        self.estimator.fit(x, y, **kwargs)
        
        v = mean_squared_error(y, self.estimator.predict(x))
        self.w = np.sqrt(v*12)
        
        self.fitted = True
        
        return self
    
class IncrementSampler(Sampler):
    def __init__(self, increment=1, **kwargs):
        super().__init__(**kwargs)
        self.increment = increment
        
    def sample(self, x):
        y = x.values.ravel() + self.increment
        
        return y
    
class LogisticTransitionSampler(Sampler):
    def __init__(self, c_prev, p_stay, **kwargs):
        super().__init__(**kwargs)
        self.fitted = False
        self.c_prev = c_prev
        self.p_stay = p_stay
        self.model_ = LogisticRegression(**kwargs)
        
    def sample(self, x):
       
        c_prev = self.c_prev + '_prev'
        
        if c_prev in x.columns: # If not dummy input for target var
            prev = x[c_prev]
            c_feat = [c for c in x.columns if not c == c_prev]        
        else: # If dummy
            cs_prev = [c for c in x.columns if (c_prev + '_') in c]
            c_feat = [c for c in x.columns if not c in cs_prev]        
            prev = x[cs_prev].idxmax(axis=1)
            prev = np.array([p.split(c_prev + '_')[1] for p in prev])

        n = x.shape[0]
        stay = 1*(np.random.rand(n) < self.p_stay)
        
        p = self.model_.predict_proba(x[c_feat])
        
        y = np.array([np.random.choice(self.model_.classes_, p=p[i,:]) if not stay[i] else prev[i] for i in range(p.shape[0])])
        
        return y
    
    def fit(self, x, y, **kwargs):
        
        if self.c_prev in x.columns: # If not dummy input for target var
            c_feat = [c for c in x.columns if not c == self.c_prev]        
        else: # If dummy
            cs_prev = [c for c in x.columns if (self.c_prev + '_') in c]
            c_feat = [c for c in x.columns if not c in cs_prev] 

        self.model_.fit(x[c_feat], y)
        self.fitted = True
        
        return self