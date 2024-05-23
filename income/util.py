import os
import numpy as np
import pandas as pd
import argparse
import yaml
import time
import pickle as pkl

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector

def dict_to_namespace(d):
    """ Converts a dictionary (recursively) to an argparse namespace
    """
    if type(d) is dict:
        for k,v in d.items():
            d[k] = dict_to_namespace(v)
        return argparse.Namespace(**d)
    else:
        return d


def load_config(config_file):
    """ Loads a Yaml configuration file and converts it to namespace form
    """
    with open(config_file, 'r') as file:
       cfg = dict_to_namespace(yaml.safe_load(file))

    return cfg

def save_model(clf, path, label):
    """ Saves a simulator model
    """
    os.makedirs(path, exist_ok=True)
    model_file = os.path.join(path, label+'.pkl')
    pkl.dump(clf, open(model_file, 'wb'))

def load_model(path, label):
    """ Loads a simulator model
    """
    model_file = os.path.join(path, label+'.pkl')
    if os.path.isfile(model_file):
        clf = pkl.load(open(model_file, 'rb'))
        return clf 
    else: 
        return None


class SubsetTransformer(TransformerMixin):
    def __init__(self):
        super().__init__() #@TODO: Variable exclusions not implemented yet

        transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), selector(dtype_exclude="category")),
                ("cat", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), selector(dtype_include="category")),
            ], verbose_feature_names_out=False
        )
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        self.feature_names_in = X.columns.tolist()
        self.default_in = X.iloc[0]
        self.default_out = self.transformer.transform(X.iloc[0:1]).iloc[0]

        return self

    def transform(self, X):

        Xp = self.pad_input_(X)
        Xt = self.transformer.transform(Xp)

        out_columns = [c for c in Xt.columns if (c in X.columns)
                       or np.any([c.rsplit('_',maxsplit=1)[0] == c_ for c_ in X.columns])]

        return Xt[out_columns]

    def pad_input_(self, X):
        Xn = X.copy()
        if Xn.shape[1] < len(self.feature_names_in):
            cmis = [c for c in self.feature_names_in if c not in X.columns]
            df = pd.DataFrame([self.default_in[cmis]]*X.shape[0], index=X.index)
            Xn = pd.concat([Xn, df], axis=1)

        Xn = Xn[self.feature_names_in]

        return Xn

    def pad_output_(self, X):
        cout = self.transformer.get_feature_names_out()

        Xn = X.copy()
        if Xn.shape[1] < len(cout):
            cmis = [c for c in cout if c not in X.columns]
            df = pd.DataFrame([self.default_out[cmis]]*X.shape[0], index=X.index)
            Xn = pd.concat([Xn, df], axis=1)

        Xn = Xn[cout]

        return Xn


    def inverse_transform(self, X):
        Xp = self.pad_output_(X)
        #@TODO: Doesn't work because of pipeline. Need to match transformed feature names too
        Xt = self.transformer.inverse_transform(Xp)

        #@TODO: Doesn't work because of pipeline. Need to match transformed feature names too
        return Xt[X.columns]


class Standardizer(StandardScaler):
    """
    Standardizes a subset of columns using the scikit-learn StandardScaler
    """
    def __init__(self, copy=True, with_mean=True, with_std=True, columns=None, ignore_missing=False):
        StandardScaler.__init__(self, copy=copy, with_mean=with_mean, with_std=with_std)
        self.columns = columns
        self.ignore_missing = ignore_missing

    def fit(self, X, y=None):
        columns = X.columns if self.columns == None else self.columns

        StandardScaler.fit(self, X[columns], y)

        return self

    def transform(self, X, copy=None):
        columns = X.columns if self.columns == None else self.columns

        Xn = X.copy()
        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            columns_mis = [c for c in columns if c not in X.columns]

            if len(columns_sub)==0:
                return X

            Xt = X.copy()
            Xt[columns_mis] = 0
            try:
                Xt = StandardScaler.transform(self, Xt[columns_sub + columns_mis], copy=copy)
            except:
                print(columns_sub + columns_mis)
                print(Xt[columns_sub + columns_mis])

            Xt = Xt[:,:len(columns_sub)]
            Xn.loc[:,columns_sub] = Xt
        else:
            Xt = StandardScaler.transform(self, X[columns], copy=copy)
            Xn.loc[:,columns] = Xt

        return Xn

    def inverse_transform(self, X, copy=None):
        columns = X.columns if self.columns == None else self.columns

        if self.ignore_missing:
            columns_sub = [c for c in columns if c in X.columns]
            Xn = self.inverse_transform_single(X, columns_sub, copy=copy)
        else:
            Xt = StandardScaler.inverse_transform(self, X[columns], copy=copy)
            Xn = X.copy()
            Xn.loc[:,columns] = Xt

        return Xn

    def inverse_transform_single(self, Xs, columns, copy=None):
        X = pd.DataFrame(np.zeros((Xs.shape[0], len(self.columns))), columns=self.columns)
        X[columns] = Xs[columns]

        Xt = StandardScaler.inverse_transform(self, X[self.columns], copy=copy)
        X.loc[:,self.columns] = Xt

        return X[columns]

    def fit_transform(self, X, y=None, **fit_params):
        Xt = StandardScaler.fit_transform(self, X, y, **fit_params)
        return Xt


def inv_dummies(df, columns, separator='_'):
    """
    Inverts the Pandas get_dummies function, converting a dataframe of dummy variables to categorical columns

    args
       columns (list or str): Specifies the original names of columns that have been converted to dummies.
           The function will look for columns with names fitting the pattern <column name><separator>* for each
           column in columns, and treat them as dummy columns of the variable with name <column name>
       separator (str, default: '_'): Specifies which separator string that separates column names from their
           values in the dummy format.
    """

    if not isinstance(columns, list):
        columns = [columns]

    out = pd.DataFrame({})
    dummy_cs = []
    for col in columns:
        col_ = col+separator
        cs = [c for c in df.columns if col_ in c]
        if len(cs) == 0:
            continue

        inv = df[cs].idxmax(axis=1)
        inv = inv.apply(lambda x : x.split(col_)[1])
        out[col] = inv
        dummy_cs += cs

    other_cs = [c for c in df.columns if c not in dummy_cs]

    if len(other_cs)==0 and len(columns) == 1:
        return out[columns[0]]
    else:
        out[other_cs] = df[other_cs]
        out = out[other_cs + columns]
        return out

class Transformation():
    """
    @TODO: Change to sklearn pipeline?
    """

    def __init__(self):
        self.c_dummies = {}
        self.standardizer = None
        self.time_ = 0

    def fit(self, df, c_cat, c_num, c_cat_suffixes=[]):
        self.c_cat = c_cat
        self.c_num = c_num

        adds = []
        for c in c_cat:
            D = pd.get_dummies(df[[c]], columns=[c])
            self.c_dummies[c] = D.columns.values

            def h_(v, c, f):
                ps = v.split(c)
                return c+f+ps[1]

            for f in c_cat_suffixes:
                adds.append(c+f)
                self.c_dummies[c+f] = [h_(v, c, f) for v in D.columns.values]

        self.c_cat = self.c_cat + adds

        # Fit standardizer to numerical columns
        self.standardizer = Standardizer(copy=True, columns=c_num, ignore_missing=True)
        self.standardizer.fit(df)

    def transform(self, df, copy=None):

        t0 = time.time()

        # Transform categorical features
        D = pd.get_dummies(df, columns=[c for c in self.c_cat if c in df.columns])

        for c in self.c_cat:
            if c in df.columns:
                D[[cd for cd in self.c_dummies[c] if cd not in D.columns]] = 0

        # Transform numerical features
        D = self.standardizer.transform(D, copy=copy)

        # Ensure persistant feature order
        D = D[sorted(D.columns.values)]

        self.time_ += time.time()-t0

        return D

    def inverse_transform(self, df):
        raise Exception('Not yet implemented')
