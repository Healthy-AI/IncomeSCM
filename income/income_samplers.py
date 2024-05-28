import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .util import *
from .samplers import *

EDUC_MAP = {'Preschool': 1.0, '1st-4th': 2.0, '5th-6th': 3.0, '7th-8th': 4.0, '9th': 5.0, '10th': 6.0, '11th': 7.0, '12th': 8.0, 'HS-grad': 9.0, 
            'Some-college': 10.0, 'Assoc-voc': 11.0, 'Assoc-acdm': 12.0, 'Bachelors': 13.0, 'Masters': 14.0, 'Prof-school': 15.0, 'Doctorate': 16.0}

STUDIES_MAP = {'No studies': 0, 'Evening course': 1, 'Day course': 2, 'Full-time studies': 3}

class StudiesSampler(Sampler):
    """
    
    Sampler of studies (action)
    
    Relies on access to columns in sampling:
    
        @TODO: T.B.D.
        
    """
    
    classes_ = ['No studies', 'Evening course', 'Day course', 'Full-time studies']

    def __init__(self, intervention=None, **kwargs):
        super().__init__(**kwargs)

        self.intervention=intervention
        
    def fit(self, x, y):     
        """ No data, so nothing to fit """
        self.fitted = True
        
        return self
    
    def sample_proba(self, x):
        n = x.shape[0]
        
        # Order: 'No studies', 'Evening course', 'Day course', 'Full-time studies'

        # Default policy
        # Overall, no studies and full-time studies most common. Evening courses less
        if self.intervention is None: 

            c_const = np.array([[7., -5., -3.5,  2.]])
            c_age   = np.array([[2., 1., 0., -.3]])
            
            c_own_child  = np.array([[3., 0., 0., -10]])
            
            lg_age = np.dot(x[['age']]-25, c_age)
            lg_oc = np.dot((x[['relationship']]=='Own-child').astype(int), c_own_child)
            
            #Preschool': 1.0, '1st-4th': 2.0, '5th-6th': 3.0, '7th-8th': 4.0, '9th': 5.0, '10th': 6.0, '11th': 7.0, '12th': 8.0, 'HS-grad': 9.0, 'Some-college': 10.0, 'Assoc-voc': 11.0, 'Assoc-acdm': 12.0, 'Bachelors': 13.0, 'Masters': 14.0, 'Prof-school': 15.0, 'Doctorate': 16.0

            # More likely to do full-time studies if mid-degree
            c_mid = np.array([[0., 0., 0., 5.]])
            mid_studies = x[['education-num']].isin([1,2,3,4,5,6,7,8,10,11,12]).astype(int)
            lg_mid = np.dot(mid_studies, c_mid)
            
            # Less likely to study if just finished degree
            c_bach = np.array([[0., -4, -4, -4]])
            bach_studies = (x[['education']]=='Bachelors').astype(int)
            lg_bach = np.dot(bach_studies, c_bach)
            
            c_hsg = np.array([[0., -2, -2, -2]])
            hs_studies = (x[['education']]=='HS-grad').astype(int)
            lg_hsg = np.dot(hs_studies, c_hsg)
            
            lg = lg_age + lg_oc + lg_mid + lg_bach + lg_hsg + c_const
            
            t = 1.2
            l = lambda z, t : 1./(1+np.exp(-z/t))
            p = l(lg,t)/l(lg,t).sum(axis=1,keepdims=True)
        else: 
            p = np.zeros(len(STUDIES_MAP))
            if self.intervention in STUDIES_MAP.keys():
                p[STUDIES_MAP[self.intervention]] = 1
            else: 
                raise Exception('Unknown intervention %s' % self.intervention)
            p = np.repeat(p.reshape(1,-1), n, axis=0)
        
        return p

    def sample(self, x):
    
        n = x.shape[0]
        p = self.sample_proba(x)
        
        y = np.array([np.random.choice(self.classes_, 1, p=p[i]) for i in range(n)]).ravel()
        
        return y
    
class StudiesTransition(Sampler):
    """
    Relies on access to columns in sampling:
        ...
    """
    def __init__(self, sampler, intervention=None, **kwargs):
        super().__init__(**kwargs)

        self.sampler = sampler
        self.intervention = intervention

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, x):
        n = x.shape[0]
        classes = self.sampler.classes_
        time = x['time'].values[0]

        # Set a constant intervention
        if (not self.intervention is None) and (type(self.intervention) == str):

            p = np.zeros(len(STUDIES_MAP))
            if self.intervention in STUDIES_MAP.keys():
                p[STUDIES_MAP[self.intervention]] = 1
            else: 
                raise Exception('Unknown intervention %s' % self.intervention)
            
            p = np.repeat(p.reshape(1,-1), n, axis=0)
            
        # Set a constant intervention at a specific time
        elif (not self.intervention is None) and (type(self.intervention) == dict) and (self.intervention['T'] == time):

            p = np.zeros(len(STUDIES_MAP))
            if self.intervention['action'] in STUDIES_MAP.keys():
                p[STUDIES_MAP[self.intervention['action']]] = 1
            else: 
                raise Exception('Unknown intervention %s' % self.intervention['action'])
            
            p = np.repeat(p.reshape(1,-1), n, axis=0)
            
        else: 
            # Probability of the initial state
            p = self.sampler.sample_proba(x) 
            
            # Make it more likely to continue if in ongoing studies already
            ongoing = ['11th','9th','Some-college','Assoc-acdm','7th-8th',
                    'Assoc-voc','5th-6th','10th','Preschool','12th','1st-4th']
            
            p = p + 2*((x['studies#prev'] == 'Full-time studies')&(x['education'].isin(ongoing))) \
                        .values.reshape([-1, 1])*np.array([[0, 0, 0, 1]])*p
            
            # Make it unlikely to start full-time studies from nothing
            p = p - 0.7*(x['studies#prev'] != 'Full-time studies').values.reshape([-1, 1])*np.array([[0, 0, 0, 1]])*p

            # Make it unlikely to start full-time studies or day course if income is already reasonably high
            p = p - 0.7*(x['income#prev'] > 50000).values.reshape([-1, 1])*np.array([[0, 0, 0.7, 1]])*p
            
            # Make full-time studies impossible if doctorate
            p = p - 1.*(x['education'] == 'Doctorate').values.reshape([-1, 1])*np.array([[0, 0, 0, 1]])*p
            
            # Normalize
            p = p/p.sum(axis=1,keepdims=True)

        y = np.array([np.random.choice(classes, 1, p=p[i]) for i in range(n)]).ravel()

        return y
    

class StudiesSamplerPolicy1(Sampler):
    """
    Relies on access to columns in sampling:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, n_samples):
        y = pd.DataFrame({'studies': np.array(['No studies']*n_samples)})
        
        return y['studies'].values.ravel()
    
class StudiesTransitionPolicy1(Sampler):
    """
    Relies on access to columns in sampling:
        Age
        Education
        Income#prev
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, x):
        n = x.shape[0]
        
        c = (x['age'] < 40)&(x['income#prev']<20000)&(x['education-num']<13)
        y = pd.DataFrame({'studies': np.array(['No studies']*n)})
        y.loc[c,'studies'] = 'Full-time studies'

        return y.values.ravel()
    
class StudiesSamplerPolicy2(Sampler):
    """
    Relies on access to columns in sampling:
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, x):
        n = x.shape[0]
        
        c1 = (x['age'] < 22)&(x['age'] >= 18)
        #c2 always false in first time step
        #c3 always false in first time step
        c4 = (1-c1)&(x['education-num']<13)
        
        y = pd.DataFrame({'studies': np.array(['No studies']*n)})
        
        y.loc[c1,'studies'] = 'Full-time studies'
        y.loc[c4,'studies'] = 'Evening course'
        
        return y['studies'].values.ravel()
    
class StudiesTransitionPolicy2(Sampler):
    """
    Relies on access to columns in sampling:
        Age
        Education
        Income#prev
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, x):
        n = x.shape[0]
        
        c1 = (x['age'] < 22)&(x['age'] >= 18)
        c2 = (1-c1)&(x['education-num']<9)&(x['income#prev']<10000)
        c3 = (1-c2)&(x['income#prev']<20000)
        c4 = (1-c3)&(x['education-num']<13)
        
        y = pd.DataFrame({'studies': np.array(['No studies']*n)})
        
        y.loc[c1,'studies'] = 'Full-time studies'
        y.loc[c2,'studies'] = 'Full-time studies'
        y.loc[c3,'studies'] = 'Day course'
        y.loc[c4,'studies'] = 'Evening course'

        return y['studies'].values.ravel()

class IncomeSampler(Sampler):
    """
    
    Sampler of income when previous income value is not available
    
    Relies on access to columns in sampling:
    
        [columns supplied to fit(.) in the first argument]
        workclass_Without-pay
        
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.brier_ = None
        self.fitted = False
        self.a = None
        self.b = None
        
    def fit(self, x, y):     

        cols = [c for c in x.columns if 'studies_' not in c]
        self.model_ = RandomForestRegressor(n_estimators=10, min_samples_leaf=50).fit(x[cols],y)
        
        yp = self.model_.predict(x[cols])
        self.brier_ = mean_squared_error(y, yp)
        
        self.sample(x, update_params=True)
        
        self.fitted = True
        
        return self

    def sample(self, x, update_params=False):    
        
        n = x.shape[0]
        cols = [c for c in x.columns if 'studies_' not in c]
        
        yp = self.model_.predict(x[cols])
        yp = yp + np.random.randn(n)*0.2+0.3 # To control unemployment rate
        yp = yp*(yp>0)
        yp = np.power(yp*(yp>0), 2) # To control median
        if update_params:
            self.a = yp.mean()
        yp = (yp*70000. / self.a).astype(np.int32) # To make mean 70000 before noise
        if update_params:
            self.b = yp.max()
        yp = yp*(1-np.sqrt(self.brier_)) + (yp>0)*np.random.beta(1.2, 20.5, n)*self.b*np.sqrt(self.brier_) # Add noise
        
        yp[x['studies_Full-time studies']==1] = 0
        yp[x['studies_Day course']==1] = 4*yp[x['studies_Day course']==1]/5
        
        yp[x['workclass_Without-pay']==1] = 0
        
        
        yp = np.round(yp)
        
        return yp
    
class IncomeTransition(Sampler):
    """
    Relies on access to columns in sampling:
        ...
    """
    def __init__(self, sampler, prev_weight=0.95, max_raise_frac=0.04, **kwargs):
        super().__init__(**kwargs)
    
        self.prev_weight = prev_weight
        self.sampler = sampler
        self.max_raise_frac = max_raise_frac

    def fit(self, x, y):

        self.fitted = True

        return self

    def sample(self, x):

        n = x.shape[0]
        c_curr = [c for c in x.columns if '#prev' not in c]
        x_curr = x[c_curr]
        
        # This column is assumed not to be affected by transformers
        y_prev = x['income#prev']
        
        # If switching between full-time studies and not, don't look at previous income
        prev_weight = self.prev_weight*(x['studies#prev_Full-time studies'] == x['studies_Full-time studies'])
            
        y_new = self.sampler.sample(x_curr)
        #y_new += x['studies#prev_Full-time studies']*(np.random.rand(n)*2000) # No bonus for moving from full-time studies?
        y_new += x['studies#prev_Day course']*(np.random.rand(n)*1000)
        y_new += x['studies#prev_Evening course']*(np.random.rand(n)*100)

        # Add a random yearly raise to the previous year's salary
        raise_factor = 1+(np.random.rand(n)*self.max_raise_frac)
        
        y = prev_weight*(y_prev*raise_factor) + (1-prev_weight)*y_new
        
        y[x['studies_Full-time studies']==1] = 0
        y[x['studies_Day course']==1] = 4*y[x['studies_Day course']==1]/5
        y = np.round(y)

        return y

class CapitalTransition(Sampler):
    """
    Relies on access to columns in sampling:
        ...
    """
    def __init__(self, sampler, p_stay=0.8, **kwargs):
        super().__init__(**kwargs)

        self.p_stay = p_stay
        self.sampler = sampler

    def fit(self, x, y):
        self.fitted = True

        return self

    def sample(self, x):

        n = x.shape[0]
        c_curr = [c for c in x.columns if '#prev' not in c]
        x_curr = x[c_curr]
        
        y_prev = x['capital-net#prev']
        
        
        stay = 1*(np.random.rand(n) < self.p_stay)
        stay_on = stay*(y_prev != 0)
        z = np.random.rand(n) < self.sampler.zero_estimator.predict_proba(x_curr)[:,1]
        z = 1-stay_on + (1-stay)*z
        
        y = self.sampler.estimator.predict(x_curr) + np.random.randn(n)*np.sqrt(self.sampler.mse_)*self.sampler.std_mod
        
        y = (1-z)*y
        
        if not self.sampler.bounds is None:
            y = np.clip(y, self.sampler.bounds[0], self.sampler.bounds[1])

        if self.sampler.round_result:
            y = np.round(y)
            
        y = np.round(y)

        return y

class EducationNumSampler(Sampler):
    """
    
    Relies on access to columns in sampling:
    
        education
        
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def sample(self, x):
        y = np.array([EDUC_MAP[v] for v in list(x.values.ravel())]).ravel()
        
        return y
    
class EducationTransition(Sampler):
    """
    
    Relies on access to columns in sampling:
    
        education#prev
        studies#prev
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
        
    def sample(self, x):
        
        educ_prev = inv_dummies(x, columns='education#prev')['education#prev']
        
        educ_num = np.array([EDUC_MAP[v] for v in list(educ_prev.values)]).ravel()
        rev_map = dict([(v,k) for k,v in EDUC_MAP.items()])
       
        # Transitions based on studies
        p = 0.9*(x['studies#prev_Full-time studies']==1) + 0.05*(x['studies#prev_Evening course']==1)
        trans = np.random.rand(p.shape[0]) < p
        
        educ_new = np.clip(educ_num + trans, 1, 16)
        y = np.array([rev_map[n] for n in educ_new.values]).ravel()
        
        return y
    
    
class WorkclassTransition(Sampler):
    """
    
    Relies on access to columns in sampling:
    
        workclass#prev
    """
    def __init__(self,  p_stay=0.95, **kwargs):
        super().__init__(**kwargs)
        
        self.p_stay = p_stay
        
    def fit(self, x, y):
        x = x['workclass']
        
        L = sorted(x.unique())
        p = [(x==L[i]).sum()/x.shape[0] for i in range(len(L))]
        self.p_ = p
        self.labels_ = L
        
        self.fitted=True
        
        return self
        
    def sample(self, x):
        
        x = x['workclass#prev']
        n = x.shape[0]
        stay = np.random.rand(n) < self.p_stay
        x_move = pd.Series(np.random.choice(self.labels_, size=n, p=self.p_))
        y = stay*x.values + (~stay)*x_move.values
        
        return y
    
class MaritalStatusTransition(Sampler):
    """
    
    Relies on access to columns in fitting: 
        age
        marital-status
    
    Relies on access to columns in sampling:
        age
        marital-status#prev
        studies#prev
        
    """
    def __init__(self,  a_stay=0, **kwargs):
        super().__init__(**kwargs)
        
        self.a_stay = a_stay
        
        self.labels = ['Never-married','Married','Divorced','Separated','Widowed']
        self.possible_transitions = np.array([
            [1, 1, 0, 0, 0], 
            [0, 1, 1, 1, 1],
            [0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1],
            [0, 1, 0, 0, 1],
        ])
        
    def fit(self, x, y):
        M = RandomForestClassifier().fit(x[['age']], y)
        self.model_ = M
        
        self.fitted=True
        
        return self
        
    def sample(self, x):
        
        n = x.shape[0]
        
        M = self.model_
        I = [list(M.classes_).index(l) for l in self.labels]
        
        p_age = M.predict_proba(x[['age']])[:,I]
        
        MS = pd.get_dummies(x['marital-status#prev'], prefix=None)
        for l in self.labels:
            if l not in MS.columns: 
                MS[l] = 0
        MS = MS[self.labels].values
        
        study = 1*((x['studies#prev']=='Full-time studies')&(x['marital-status#prev']=='Never-married')).values.reshape(-1,1)
        study_factor = np.ones((n,len(M.classes_))) + study*np.array([[2, 0.5, 1, 1, 1]]*n)
                               
        p_age_study = p_age*study_factor
        p_age_study = p_age_study/p_age_study.sum(axis=1, keepdims=True)              
        
        possible_statuses = np.dot(MS, self.possible_transitions)

        p = (p_age*possible_statuses + self.a_stay*MS)/ \
            (p_age*possible_statuses + self.a_stay*MS).sum(axis=1, keepdims=True)
        
        y = [np.random.choice(self.labels, size=1, p=p[i,:].astype(np.float32))[0] for i in range(n)]
        
        return y
    
    
class RelationshipTransition(Sampler):
    """
    Relies on access to columns in fitting: 
        age, education, workclass, marital-status, race, sex
    
    Relies on access to columns in sampling:
        age, education, workclass, marital-status, race, sex
        relationship#prev
        
    """
    def __init__(self, sampler, p_stay=0.9, **kwargs):
        super().__init__(**kwargs)

        self.p_stay = p_stay
        self.c_feat = ['age', 'education', 'workclass', 'marital-status', 'race', 'sex']
        self.sampler = sampler

    def fit(self, x, y):

        self.fitted = True

        return self

    def sample(self, x):

        n = x.shape[0]
        p_stay = np.ones(n)*self.p_stay
        
        stay = 1*(np.random.rand(n) < p_stay)
        
        c_curr = [c for c in x.columns if '#prev' not in c]
        cs_prev = [c for c in x.columns if ('relationship#prev_') in c]   
        prev = x[cs_prev].idxmax(axis=1)
        prev = np.array([p.split('relationship#prev_')[1] for p in prev])
            
        p = self.sampler.predict_proba(x[c_curr])
        
        y = np.array([np.random.choice(self.sampler.classes_, p=p[i,:]) if not stay[i] else prev[i] for i in range(p.shape[0])])
        
        return y
    
class HoursPerWeekTransition(Sampler):
    """
    Relies on access to columns in fitting: 
        age, education, workclass, marital-status, race, relationship, sex
        hours-per-week
    
    Relies on access to columns in sampling:
        age, education, workclass, marital-status, race, relationship, sex
        hours-per-week#prev
        
    """
    def __init__(self, transformation, prev_weight=0.9, sampler=None, **kwargs):
        super().__init__(**kwargs)

        self.transf_ = transformation
        self.prev_weight = prev_weight
        self.c_feat = ['age', 'education', 'workclass', 'occupation', 'marital-status', 'race', 'relationship', 'sex']
        self.sampler = sampler

    def fit(self, x, y):

        if self.sampler is None:
            M = GaussianRegressionSampler(RandomForestRegressor(min_samples_leaf=50), round_result=True, bounds=(0,100), std_mod=2.5)
            M.fit(self.transf_.transform(x[self.c_feat]), y)
            self.model_ = M
        else:
            self.model_ = self.sampler 

        return self

    def sample(self, x):

        p = self.model_.sample(self.transf_.transform(x[self.c_feat]))
        y = x['hours-per-week#prev']*(self.prev_weight) + (1-self.prev_weight)*p
        y = np.round(y).astype(np.int32)
        y = np.clip(y, 0, 7*24)

        return y
    
    
class OccupationTransition(Sampler):
    
    def __init__(self, sampler, p_stay, **kwargs):
        super().__init__(**kwargs)
        self.fitted = False
        self.p_stay = p_stay
        self.sampler = sampler
        
    def sample(self, x):

        n = x.shape[0]
        p_stay = np.ones(n)*self.p_stay
        p_stay[x['studies#prev_Full-time studies'].astype(int)] /= 4
        
        stay = 1*(np.random.rand(n) < p_stay)
        
        c_curr = [c for c in x.columns if '#prev' not in c]
        cs_prev = [c for c in x.columns if ('occupation#prev_') in c]   
        prev = x[cs_prev].idxmax(axis=1)
        prev = np.array([p.split('occupation#prev_')[1] for p in prev])
            
        p = self.sampler.predict_proba(x[c_curr])
        
        y = np.array([np.random.choice(self.sampler.classes_, p=p[i,:]) if not stay[i] else prev[i] for i in range(p.shape[0])])
        
        return y
    
    def fit(self, x, y, **kwargs):
        self.fitted = True
        
        return self
    
    