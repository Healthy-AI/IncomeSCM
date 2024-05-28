import os, sys
import urllib.request
import pandas as pd
import numpy as np
import zipfile
import subprocess

from .income_samplers import *

ADULT_URL = 'https://archive.ics.uci.edu/static/public/2/adult.zip'

def download_adult_data(folder):
    """ Downloads the Adult data set, extracts it and removes the file """
    fname = 'adult.tmp.zip'

    os.makedirs(folder, exist_ok=True)
    fpath = os.path.join(folder, fname)
    
    subprocess.run(["curl", ADULT_URL, "-o", fpath]) 

    with zipfile.ZipFile(fpath, 'r') as zip_ref:
        zip_ref.extractall(folder)

    os.remove(fpath)

def load_income_data(folder='data/adult/', data_only=False, download=True):
    """
    Loads the "Adult" income dataset and returns

    args
        folder: The folder where the data files are
        data_only: If true, returns only the dataframe for the training portion
        download: If true, downloads the data set from UCI if it doesn't exist

    returns
        D_tr: Training dataframe
        c_cat: Categorical columns
        c_num: Numerical columns
        c_out: Outcome column
        c_features: Feature columns
    """

    train_file = os.path.join(folder, 'adult.data')
    col_file = os.path.join(folder, 'adult.names')
    test_file = os.path.join(folder, 'adult.test')

    # Check if files exist
    if download and (not os.path.isfile(train_file) or not os.path.isfile(col_file) or not os.path.isfile(test_file)):
        print('Donwloading Adult data set...')
        download_adult_data(folder)

    # Load data
    D_tr = pd.read_csv(train_file, header=None, delimiter=', ', engine='python')

    # Column names
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income>50k']
    c_cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    c_bin = ['income>50k']
    c_out = 'income>50k'

    D_tr.columns = columns

    # Remove redundant features
    D_tr['capital-net'] = D_tr['capital-gain'] - D_tr['capital-loss']
    D_tr = D_tr.drop(columns=['fnlwgt', 'capital-gain', 'capital-loss'])
    c_num = ['age', 'education-num', 'capital-net', 'hours-per-week']

    D_tr = D_tr.replace('<=50K', 0).replace('>50K', 1).replace('?', np.nan).replace(' ', '')
    D_tr = D_tr.replace('Married-civ-spouse', 'Married').replace('Married-AF-spouse', 'Married').replace('Married-spouse-absent', 'Married')

    # Drop nan values
    D_tr = D_tr.dropna()

    # Add index columns
    D_tr['id'] = range(D_tr.shape[0])
    D_tr['time'] = 0

    # Feature columns
    c_features = [c for c in list(D_tr.columns) if c not in ['income>50k', c_out]]

    # Add studies variable
    c_cat += ['studies']
    L_studies = StudiesSampler.classes_
    D_tr['studies'] = (L_studies*int(np.ceil(D_tr.shape[0]/len(L_studies))))[:D_tr.shape[0]]

    D_tr['income'] = D_tr['income>50k'].copy()

    # Set data type to category for categorical features
    D_tr[c_cat] = D_tr[c_cat].astype('category')

    if data_only:
        return D_tr
    else:
        return D_tr, c_cat, c_num, c_out, c_features
