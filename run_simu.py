"""
This script is used to run simulations for causal analysis.

Command line arguments:
    sys.argv[1]: Index of data models, 0 for mean shift and 1 for median shift.
    sys.argv[2]: The number of folds K for sample spliting. 1 for no spliting.

Example usage:
    python run_simu.py 0 1
"""
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

from tqdm import tqdm

from causarray import *
from generate_simu_data import generate_data


n_list = [100, 200, 300, 400, 500]
p_list = [8000]

m = 100 # number of cells per individuals
ps = 200 # number of non-nulls
d = 5 # number of covariates (the intercept not included)

# parameters for the FDP control
# P(FDP > c) < alpha
c = 0.1
alpha = 0.05
# E[FDP] < q_alpha
q_alpha = 0.1

shift_list = ['mean', 'median']
shift = shift_list[int(sys.argv[1])]
signal = 1. if shift=='mean' else signal = 10.

K = int(sys.argv[2])

path_result = 'results/simu/'
os.makedirs(path_result, exist_ok=True)
for n in n_list:
    for p in p_list:
        df_res = pd.DataFrame()
        for seed in range(50):

            W, A, X, Y, theta, _ = generate_data(
                n, p, d-1, m, ps, intercept=1.,
                shift=shift, psi=0.,
                scale_beta=0.5 if shift=='median' else 2.,
                signal=signal, seed=seed)
            X = np.transpose(X, (1,0,2))
            Yg = np.sum(X, 1)
            result = []
            for func in [ATE, SATE, QTE, SQTE]:
                df = func(Yg.copy(), W.copy(), A.copy(), B=1000, alpha=alpha, c=c, family='poisson', K=K)
                true = (theta!=0.).astype(int)
                pred = (df['rej']==1).astype(int)
                res = comp_stat(true, pred, c)
                res.insert(0, func.__name__)
                result.append(res)

                pred = (df['qvals'] <= q_alpha).astype(int)
                res = comp_stat(true, pred, c)
                res.insert(0, func.__name__ + '-BH')
                result.append(res)

            _df = pd.DataFrame(result, columns=['Method','typeI_err', 'FDP', 'power', 'FDPex', 'num_dis'])
            _df['seed'] = seed
            print(_df)
            df_res = pd.concat([df_res, _df], axis=0)
            df_res.to_csv(path_result+'n_{}_p_{}_signal_{:.02f}_shift_{}_K_{}_seed_{}.csv'.format(n,p,signal,shift,K,seed))
