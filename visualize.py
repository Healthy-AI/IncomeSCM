import os 
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')

from income.util import *

def save_crop_close(path):
    plt.tight_layout()
    plt.savefig(path)
    os.system('pdfcrop %s %s' % (path, path))
    plt.close()


def visualize(cfg):
    """ Visualize the samples from the simulator """

    fname0 = '%s_%s_n%d_T%d_s%d.pkl' % (cfg.samples.label, 'no', cfg.samples.n_samples, cfg.samples.horizon, cfg.samples.seed)
    s_path0 = os.path.join(cfg.samples.path, fname0)
    
    fname1 = '%s_%s_n%d_T%d_s%d.pkl' % (cfg.samples.label, 'full', cfg.samples.n_samples, cfg.samples.horizon, cfg.samples.seed)
    s_path1 = os.path.join(cfg.samples.path, fname1)

    df0 = pd.read_pickle(s_path0)
    df1 = pd.read_pickle(s_path1)

    plt.rc('font', family='serif', size=16)

    # Create output directory
    fdir = cfg.figures.path
    os.makedirs(fdir, exist_ok=True)

    # Compute CATE and plot
    cate = df1['income'] - df0['income']
    plt.hist(cate, bins=30)
    plt.xlabel('CATE (USD)')
    plt.ylabel('Number')
    save_crop_close(os.path.join(fdir, 'cate_histogram.pdf'))

    dfc = df0.copy()
    dfc['cate'] = cate

    # Compute CATE by Age
    gb = dfc[['age', 'cate']].groupby('age', as_index=False).mean()
    plt.plot(gb['age'], gb['cate'])
    plt.xlabel('Age')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_age.pdf'))

    # Compute CATE by Age
    bins = [(0,30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 100)]
    x = []
    y = []
    for i,j in bins:
        x.append((i+j)/2)
        y.append(dfc[(dfc['age']>=i) & (dfc['age']<j)]['cate'].mean())
    plt.plot(x, y)
    plt.xlabel('Age')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_age_disc.pdf'))

    # Compute CATE by Education
    gb = dfc[['education-num', 'cate']].groupby('education-num', as_index=False).mean()
    plt.plot(gb['education-num'], gb['cate'])
    plt.xlabel('Education')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_education.pdf'))

    # Compute CATE by Sex
    gb = dfc[['sex', 'cate']].groupby('sex', as_index=False).mean()
    plt.bar(gb['sex'], gb['cate'])
    plt.xlabel('Sex')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_sex.pdf'))

    # Compute Work hours histogram
    bins = [(i,i+19) for i in range(0,130,20)]
    x = []
    y = []
    for i,j in bins:
        x.append((i+j)/2)
        y.append(dfc[(dfc['hours-per-week']>=i) & (dfc['hours-per-week']<j)]['cate'].mean())
    plt.plot(x, y)
    plt.xlabel('Hours-per-week')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_hours-per-week_disc.pdf'))
    print(bins)

    # Compute CATE by hours-per-week
    gb = dfc[['hours-per-week', 'cate']].groupby('hours-per-week', as_index=False).mean()
    plt.bar(gb['hours-per-week'], gb['cate'])
    plt.xlabel('hours-per-week')
    plt.ylabel('CATE')
    save_crop_close(os.path.join(fdir, 'cate_vs_hours-per-week.pdf'))

    """
    dfc = df0.copy()
    dfc['cate'] = cate

    gb = dfc[['age', 'cate']].groupby('age', as_index=False).mean()
    plt.plot(gb['age'], gb['cate'])
    plt.show()

    gb = dfc[['education-num', 'cate']].groupby('education-num', as_index=False).mean()
    plt.plot(gb['education-num'], gb['cate'])
    plt.show()
    """


if __name__ == "__main__":
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Visualize IncomeSim samples')
    parser.add_argument('-c', '--config', type=str, dest='config', help='Path to config file', default='configs/config_v1.yml')
    args = parser.parse_args()

    # Load config file
    cfg = load_config(args.config)

    # Fit simulator
    visualize(cfg)
    