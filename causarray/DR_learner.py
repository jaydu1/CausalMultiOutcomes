import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
from scipy.stats import norm
from causarray.estimation import fit_qr, AIPW_mean, AIPW_quantile, cross_fitting
from causarray.glm_test import fit_glm
from causarray.inference import multiplier_bootstrap, step_down, augmentation
from causarray.utils import reset_random_seeds
from statsmodels.stats.multitest import multipletests



def ATE(
    Yg, W, D, W_D=None, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the average treatment effects (ATEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)

    n = W.shape[0]
    p = Yg.shape[1]

    pi, Y_hat_0, Y_hat_1 = cross_fitting(Yg, W, D, W_D, family=family, **kwargs)
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Yg, 1-D, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Yg, D, Y_hat_1, pi, pseudo_outcome=True)

    tau_estimate = tau_1 - tau_0
    eta = eta_1 - eta_0  - tau_estimate[None, :]

    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        'pvals': pvals, 
        'qvals': qvals
        })

    return df_res



def SATE(
    Yg, W, D, W_D=None, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the standardized average treatment effects (SATEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)

    n = W.shape[0]
    p = Yg.shape[1]    

    pi, Y_hat_0, Y_hat_1 = cross_fitting(Yg, W, D, W_D, family=family, **kwargs)
    Y2_hat_0 = Y_hat_0**2
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Yg, 1-D, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Yg, D, Y_hat_1, pi, pseudo_outcome=True)
    _, eta_2 = AIPW_mean(Yg**2, 1-D, Y2_hat_0, 1-pi, pseudo_outcome=True)

    idx = np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2 <= 0.
    sd = np.sqrt(np.maximum(
        np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2, 1e-6))[None,:]
    tau_estimate = np.mean((eta_1 - eta_0) / sd, axis=0)
    tau_estimate[idx] = 0.
    eta = (eta_1 - eta_0) / sd - tau_estimate[None,:] * (
        eta_2 + np.mean(eta_2, axis=0, keepdims=True) - 
        2 * np.mean(eta_0, axis=0, keepdims=True) * eta_0)/ (2 * sd**2)

    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        'pvals': pvals, 
        'qvals': qvals
        })

    return df_res




def QTE(
    Yg, W, D, W_D=None, B=1000, alpha=0.05, c=0.01, family='poisson', 
    varrho=0.5, **kwargs):
    '''
    Estimate the quantile treatment effects (QTEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    varrho : float
        The quantile level.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.

    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)
    
    n = W.shape[0]
    p = Yg.shape[1]

    id_g = np.where(np.quantile(Yg, varrho, axis=0)>np.min(Yg, axis=0))[0]

    pi, tau_arr, density_arr, nu_arr = cross_fitting(Yg[:,id_g], W, D, W_D, family=family, 
        estimand='quantile', **kwargs)

    # point estimation of the treatment effect
    # tau_0, eta_0 = AIPW_quantile(Yg[:,id_g], 1-D, np.mean(tau_arr[0], axis=0), np.mean(density_arr[0], axis=0), nu_arr[0], 1-pi, varrho, pseudo_outcome=True)
    # tau_1, eta_1 = AIPW_quantile(Yg[:,id_g], D, np.mean(tau_arr[1], axis=0), np.mean(density_arr[1], axis=0), nu_arr[1], pi, varrho, pseudo_outcome=True)
    tau_0, eta_0 = AIPW_quantile(Yg[:,id_g], 1-D, tau_arr[0], density_arr[0], nu_arr[0], 1-pi, varrho, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_quantile(Yg[:,id_g], D, tau_arr[1], density_arr[1], nu_arr[1], pi, varrho, pseudo_outcome=True)

    tau_estimate = (tau_1 - tau_0)
    idx = (np.abs(tau_estimate) > 1e6)
    tau_estimate[idx] = 0.
    eta = (eta_1 - eta_0) - tau_estimate[None, :]
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)
    
    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,

        'tvalues': tvalues,
        'rej': V,

        'pvals': pvals, 
        'qvals': qvals,
        })
    df.index = id_g

    df_res = pd.DataFrame(columns=df.columns, index=np.arange(p))
    df_res.loc[id_g,:] = df

    return df_res


def SQTE(
    Yg, W, D, W_D=None, B=1000, alpha=0.05, c=0.01, 
    varrho=0.5, family='poisson', **kwargs):
    '''
    Estimate the standardized quantile treatment effects (SQTEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    varrho : float
        The quantile level.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.

    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)
    
    n = W.shape[0]
    p = Yg.shape[1]

    upper = 0.75 if 'upper' not in kwargs else kwargs['upper']
    lower = 0.25 if 'lower' not in kwargs else kwargs['lower']
    id_g = np.where(np.quantile(Yg, varrho, axis=0)>np.min(Yg, axis=0))[0]

    pi, tau_arr, density_arr, nu_arr = cross_fitting(Yg[:,id_g], W, D, W_D, family=family, 
        estimand='quantile', qs_0=[lower,varrho,upper], **kwargs)

    # point estimation of the treatment effect    
    tau_0_upper, eta_0_lower = AIPW_quantile(Yg[:,id_g], 1-D, np.mean(tau_arr[0], axis=0), np.mean(density_arr[0], axis=0), nu_arr[0], 1-pi, lower, pseudo_outcome=True)
    tau_0, eta_0 = AIPW_quantile(Yg[:,id_g], 1-D, np.mean(tau_arr[1], axis=0), np.mean(density_arr[1], axis=0), nu_arr[1], 1-pi, varrho, pseudo_outcome=True)
    tau_0_lower, eta_0_upper = AIPW_quantile(Yg[:,id_g], 1-D, np.mean(tau_arr[2], axis=0), np.mean(density_arr[2], axis=0), nu_arr[2], 1-pi, upper, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_quantile(Yg[:,id_g], D, np.mean(tau_arr[-1], axis=0), np.mean(density_arr[-1], axis=0), nu_arr[-1], pi, varrho, pseudo_outcome=True)

    eta_iqr = eta_0_upper - eta_0_lower
    iqr_0 = tau_0_upper - tau_0_lower

    tau_estimate = (tau_1 - tau_0)/iqr_0
    
    idx = np.isnan(iqr_0) | (np.abs(tau_estimate) > 1e6)

    tau_estimate[idx] = 0.
    eta = ((eta_1 - eta_0) - tau_estimate[None,:] * eta_iqr) / iqr_0
    eta[:,idx] = 0.
    # eta = (eta_1 - eta_0) / iqr_0
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        
        'pvals': pvals, 
        'qvals': qvals
        })
    df.index = id_g

    df_res = pd.DataFrame(columns=df.columns, index=np.arange(p))
    df_res.loc[id_g,:] = df

    return df_res

