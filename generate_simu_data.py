import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm
from causarray import *


def generate_data(
    n_samples, n_features, d, m, s,
    signal=1., intercept=0., psi=0., seed=0, shift='mean',
    scale_alpha=1., scale_beta=2.
    ):
    '''
    Parameters
    ----------
    n_samples: int
        Number of samples.
    n_features: int
        Number of features.
    d: int
        Number of latent variables.
    m: int
        Number of replications.
    s: int
        Number of active features.
    signal: float
        Signal strength.
    intercept: float
        Intercept.
    psi: float
        Zero-inflation rate.
    seed: int
        Random seed.
    shift: str
        Shift type.
    scale_alpha: float
        Scale parameter for the beta distribution.
    scale_beta: float
        Scale parameter for the beta distribution.
    
    Return
    ----------
    W: 2d-array
        The covariates.
    A: 1d-array
        The treatment.
    X: 3d-array
        The gene expression of cells, with dimension (m,n_samples,n_features).
    Y: 1d-array
        The gene expression of individuals.
    theta: 1d-array
        The active features.
    signal: float
        Signal strength.
    '''
    reset_random_seeds(seed)

    tmp = np.random.beta(scale_alpha, scale_beta, size=(1,n_features))

    W = np.random.normal(size=(n_samples, d))
    W = np.c_[np.ones((n_samples,1)), W]
    B = np.random.normal(size=(d, n_features))/2
    B = np.r_[intercept * np.ones((1,n_features)), B]

    beta = np.ones(d+1)/(d+1)
    A = np.random.binomial(1, 1 / (1 + np.exp(-W @ beta)), size=(n_samples,))
    
    # Generate effects
    Theta = W @ B
    
    if shift=='mean':
        prob = sp.special.softmax(np.log(np.std(np.exp(Theta), axis=0)))
        theta = np.zeros(n_features)
        theta[np.random.choice(np.arange(n_features), size=s, replace=False, p=prob)] = 1.
    else:
        theta = np.random.binomial(1, s/n_features, size=(n_features,))
    signal = tmp * signal
    
    if shift=='mean':
        signal = signal * (np.random.binomial(1, 0.5, size=(1,n_features)) * 2 - 1)    
    
    expTheta = np.exp(Theta)
    Y = np.random.poisson(expTheta)
    
    mu, sigma2 = Theta, signal
    
    if shift=='median':
        expTheta = np.tile(np.exp(Theta), (m,1,1))
        X = np.random.poisson(expTheta)
        eps = np.random.lognormal(mu[None,:,:] - signal[None,:,:]**2/2, signal[None,:,:], size=(m,n_samples,n_features))    
        eps = np.round(eps)
        X = np.where(((A[:,None] @ theta[None,:])==1)[None,:,:], eps, X)

    elif shift=='mean':
        Theta = np.where(
            (A[:,None] @ theta[None,:])==1, Theta + signal, Theta
                    )
        X = np.random.poisson(np.tile(np.exp(Theta), (m,1,1)))
    else:
        pass

    zero_inflation = np.random.binomial(1, 1-psi, X.shape)
    X = X * zero_inflation
    
    return W, A, X, Y, theta, signal
