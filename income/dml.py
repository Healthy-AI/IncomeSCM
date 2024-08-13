from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .estimators import *

class DMLCATEEstimator(CausalEffectEstimator):
    def __init__(self, y_estimator, t_estimator, e_estimator, c_int='intervention', c_out='outcome', c_adj=[], v_int0=0, v_int1=1):

        self.c_int = c_int
        self.c_out = c_out
        self.c_adj = c_adj
        self.v_int0 = v_int0
        self.v_int1 = v_int1
        self.c_int_bin = '%s__%s' % (c_int, v_int1)

        self.y_estimator = get_estimator(y_estimator)
        self.t_estimator = get_estimator(t_estimator)
        self.e_estimator = get_estimator(e_estimator)
        
    def fit(self, x, y=None, sample_weight=None):

        t = 1*(x[self.c_int] == self.v_int1)
        y = x[self.c_out]

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj] 

        self.y_estimator.fit(x[c_adjs], y)
        self.t_estimator.fit(x[c_adjs], t)

        ry = y - self.y_estimator.predict(x[c_adjs])
        rt = t - self.t_estimator.predict_proba(x[c_adjs])[:,1]

        w = rt**2
        z = ry/rt

        self.e_estimator.fit(x[c_adjs], z, sample_weight=w)

        return self

        
    def predict(self, x):
        """ Returns the predicted CATE value for the given inputs """

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj] 
        return self.e_estimator.predict(x[c_adjs])

    def predict_proba(self, x):
        pass

    def predict_outcomes(self, x):
        """ Returns predictions of both potential outcomes """

        tb = 1*(x[self.c_int]==self.v_int1)

        c_adjs = [c for c in x.columns if c in self.c_adj or c.partition('__')[0] in self.c_adj] 
        
        tp = self.t_estimator.predict_proba(x[c_adjs])[:,1]
        yp = self.y_estimator.predict(x[c_adjs])
        ep = self.e_estimator.predict(x[c_adjs])

        y0p = yp - tp*ep
        y1p = yp + (1-tp)*ep

        yp = tb*y1p + (1-tb)*y0p
        
        return yp.values
    
        

        
