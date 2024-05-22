import numpy as np
from scipy.optimize import root_scalar
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import QuantileRegressor, LogisticRegression
from quantile_forest import RandomForestQuantileRegressor
from joblib import Parallel, delayed
import scipy as sp
from causarray.glm_test import fit_glm

n_jobs = 8


def density_estimate(y, x, **kwargs):
    from scipy.stats import nbinom, poisson, geom

    densities = {}
    likelihoods = {}

    mean = np.mean(y)
    var = np.var(y)
    y = np.round(y)
    likelihoods['poisson'] = poisson.logpmf(y, mean).sum()
    densities['poisson'] = sp.stats.poisson.pmf(x, mean)

    p = mean / var
    r = p * mean / (1-p)
    likelihoods['nbinom'] = nbinom.logpmf(y, r, p).sum()
    densities['nbinom'] = sp.stats.nbinom.pmf(x, r, p)

    p = 1 / mean
    likelihoods['geometric'] = geom.logpmf(y, p).sum()
    densities['geometric'] = sp.stats.geom.pmf(x, p)

    best_fit = max(likelihoods, key=lambda x: likelihoods[x])

    return densities[best_fit]



def AIPW_mean(y, A, mu, pi, pseudo_outcome=False):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    mu : array
        Conditional outcome distribution estimate.
    pi : array
        Propensity score.
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.

    Returns
    -------
    tau : array
        A point estimate of the expected potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.
    '''
    weight = A / pi
    if len(mu.shape)>1 and len(weight.shape)==1:
        weight = weight[:,None]
    pseudo_y = weight * (y - mu) + mu
    tau = np.mean(pseudo_y, axis=0)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau
    



def AIPW_quantile(y, A, tau_init, density, nu, pi, q, pseudo_outcome=False):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    tau_init : array
        The initial estimate of the quantile of the potential outcome.
    density : array
        The estimated density of the initial estimate of the quantile of the potential outcome.
    nu : array
        The estimated conditional cumulative distribution function P(Y(a) <= tau_init | X, A).
    pi : array
        The estimated propensity score.
    q : float
        Quantile to be computed (e.g., q = 0.5 for the median.)
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    
    Returns
    -------
    tau : array
        A point estimate of the quantile of the potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.
    '''
    weight = A / pi
    if len(nu.shape)>1:
        if len(weight.shape)==1:
            weight = weight[:,None]
        if len(tau_init.shape)==1:
            tau_init = tau_init[None,:]
        if len(density.shape)==1:
            density = density[None,:]

    pseudo_y = tau_init + (weight * (nu - (y <= tau_init)) - (nu - q)) / density
    tau = np.mean(pseudo_y, axis=0)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau








def IPW_quantile(y, A, pi, q, pseudo_outcome=False, **kwargs):
    """
    Inverse probability weighted estimator (IPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    pi : array
        Propensity score.
    q : float
        Quantile to be computed (e.g., q = 0.5 for the median.)
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    **kwargs :
        Keyword arguments for kernel density estimation.

    Returns
    -------
    tau : array
        A point estimate of the quantile of the potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.        
    """
    weight = A / pi

    def D(chiq):
        return np.mean(weight * ((y <= chiq) - q)) 

    tau = root_scalar(D, bracket=[y.min()-1000, y.max()+1000]).root

    if pseudo_outcome:
        pseudo_y = -1. / density_estimate(y, np.round(np.array([tau])))
        return tau, pseudo_y
    else:
        return tau


def gaussian_kernel(x, y):
    h = 0.9 * np.minimum(np.std(y), sp.stats.iqr(y)/1.35) * len(y) ** (-1 / 5)

    return np.exp(-((x - y) / h) ** 2 / 2) / h / np.sqrt(2 * np.pi)




def fit_qr(Y, X, A, pi, lower=0.25, upper=0.75, family='poisson', **kwargs):
    '''
    Fit quantile regression to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments
    pi : array
        n x 1 vector of propensity scores
    **kwargs : dict
        additional arguments to pass to QuantileRegressor
    
    Returns
    -------
    B : array
        p x d matrix of coefficients
    Yhat : array or tuple
        n x p matrix of predicted values or tuple of predicted potential outcomes
    '''
    d = X[:,:].shape[1]

    def fit_qr_j(j):
        rho_0, eta_0 = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, 0.5, pseudo_outcome=True)
        rho_1, eta_1 = AIPW_quantile(Y[:,j], X, A, Y_hat_1[:,j], pi, 0.5, pseudo_outcome=True)
        if lower is None:
            rho_0_lower, eta_0_lower = np.nan, np.nan
        else:
            rho_0_lower, eta_0_lower = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, lower, pseudo_outcome=True)
        if upper is None:
            rho_0_upper, eta_0_upper = np.nan, np.nan
        else:
            rho_0_upper, eta_0_upper = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, upper, pseudo_outcome=True)

        return rho_0, rho_1, rho_0_upper-rho_0_lower, eta_0, eta_1, eta_0_upper-eta_0_lower

    # generalized linear regression
    # quantiles_pred = [lower, 0.5, upper]
    Y_hat_0, Y_hat_1 = fit_glm(Y, X, A, family=family, impute=True)[1]
    # Q_hat_0 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_0) for i in quantiles_pred]]
    # Q_hat_1 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_1) for i in quantiles_pred]]

    with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
        res = parallel(delayed(fit_qr_j)(j) for j in range(Y.shape[1]))

    rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr = list(zip(*res))
    rho_0 = np.array(rho_0)
    rho_1 = np.array(rho_1)
    iqr = np.array(iqr)
    
    eta_0 = np.array(eta_0).T
    eta_1  = np.array(eta_1).T
    eta_iqr = np.array(eta_iqr).T
    
    return rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr



def estimate_nuisance_quantile(
    Y, X, A, mu_0, mu_1, pi, qs_0=0.5, qs_1=0.5,
    X_test=None, density_est='kernel', **kwargs):
    '''
    Estimate the nuisance parameters for the q-quantile estimands.

    Parameters
    ----------
    Y : array
        Outcomes.
    X : array
        Covariates.
    A : array
        Binary treatment indicator.
    mu_0, mu_1 : array
        The estimated conditional mean of the potential outcome.
    pi : array
        The estimated propensity score.
    qs_0, qs_1 : float or array
        Quantile(s) to be computed for control and treatment (e.g., q = 0.5 for the median.)

    Returns
    -------
    tau_init : array
        A initial estimate of the quantile of the potential outcome.
    density : array
        The estimated density of the initial estimate of the quantile of the potential outcome.
    nu : array
        The estimated conditional cumulative distribution function P(Y(a) <= tau_init | X, A).
    '''
    if X_test is None:
        X_test = X
    if np.isscalar(qs_0):
        qs_0 = [qs_0]
    if np.isscalar(qs_1):
        qs_1 = [qs_1]

    def fit_qr_j(y, A, mu, pi, q):
        tau_init = IPW_quantile(y, A, pi, q)
        weight = A / pi

        if density_est=='kernel':
            density_y = gaussian_kernel(tau_init, y)
            density_mu = gaussian_kernel(tau_init, mu)
            density = np.mean(weight * (density_y - density_mu) + density_mu)
        else:
            # density_y = density_estimate(y, np.round(np.array([tau_init])))
            # density_mu = density_estimate(mu, np.round(np.array([tau_init])))
            # 
            density = density_estimate(y[A==1], np.round(np.array([tau_init])))[0]
        
        density = np.maximum(density, 1e-3)

        if 0 < np.mean(y<=tau_init) < 1:
            clf = LogisticRegression(random_state=0, fit_intercept=False).fit(np.c_[X, A[:,None]], (y<=tau_init))
            nu = clf.predict_proba(np.c_[X_test, np.ones((X_test.shape[0],1))])[:,1]
        else:
            nu = np.full((X_test.shape[0],), np.mean(y<=tau_init))

        return tau_init, density, nu

    res = []
    for q in qs_0:
        with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
            _res = parallel(delayed(fit_qr_j)(Y[:,j], 1-A, mu_0[:,j], 1-pi, q) for j in range(Y.shape[1]))
        res.append([np.array(i) for i in list(zip(*_res))])

    for q in qs_1:
        with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
            _res = parallel(delayed(fit_qr_j)(Y[:,j], A, mu_1[:,j], pi, q) for j in range(Y.shape[1]))
        res.append([np.array(i) for i in list(zip(*_res))])

    return res


from sklearn.model_selection import KFold

def cross_fitting(
    Y, X, A, X_A=None, family='poisson', K=1, 
    estimand='mean', qs_0=0.5, qs_1=0.5,
    **kwargs):
    '''
    Cross-fitting for causal estimands.

    Parameters
    ----------
    Y : array
        Outcomes.
    X : array
        Covariates.
    A : array
        Binary treatment indicator.
    X_A : array, optional
        Covariates for the propensity score model. The default is None for using X.
    family : str, optional
        The family of the generalized linear model. The default is 'poisson'.
    K : int, optional
        The number of folds for cross-validation. The default is 1.
    estimand : str, optional
        The type of estimand. The default is 'mean'.
    qs_0, qs_1 : float or array, optional
        Quantile(s) to be computed for control and treatment (e.g., q = 0.5 for the median.)
    **kwargs : dict
        Additional arguments to pass to the model.

    Returns
    -------
    pi_arr : array
        Propensity score.
    Y_hat_0_arr : array
        Estimated potential outcome under control.
    Y_hat_1_arr : array
        Estimated potential outcome under treatment.    
    '''
    
    if X_A is None:
        X_A = X

    # Get the list of valid parameters for LogisticRegression
    valid_params = LogisticRegression().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}

    if K>1:
        # Initialize KFold cross-validator
        kf = KFold(n_splits=K, random_state=0, shuffle=True)
        folds = kf.split(X)
    else:
        folds = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]
    
    # Initialize lists to store results
    pi_arr = np.zeros(X.shape[0])

    if estimand=='mean':
        Y_hat_0_arr = np.zeros_like(Y)
        Y_hat_1_arr = np.zeros_like(Y)
    else:
        if np.isscalar(qs_0):
            qs_0 = [qs_0]
        if np.isscalar(qs_1):
            qs_1 = [qs_1]
        n_est = len(qs_0) + len(qs_1)
        tau_arr = [np.zeros_like(Y, dtype=float) for _ in range(n_est)]
        density_arr = [np.zeros_like(Y, dtype=float) for _ in range(n_est)]
        nu_arr = [np.zeros_like(Y, dtype=float) for _ in range(n_est)]

    # Perform cross-fitting
    for train_index, test_index in folds:
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Estimate the proposensity score function on training data
        clf = LogisticRegression(random_state=0, fit_intercept=False, **valid_params).fit(XA_train, A_train)

        # Predict on test data        
        pi = clf.predict_proba(XA_test)[:,1]

        # Fit GLM on training data and predict on test data
        res = fit_glm(Y_train, X_train, A_train, family=family, impute=X)
        
        # Store results
        pi_arr[test_index] = pi
        

        if estimand=='mean':
            Y_hat_0_arr[test_index] = res[1][0][test_index]
            Y_hat_1_arr[test_index] = res[1][1][test_index]
        elif estimand=='quantile':
            pi_train = clf.predict_proba(XA_train)[:,1]

            Y_hat_0_train = res[1][0][train_index]
            Y_hat_1_train = res[1][1][train_index]

            res = estimate_nuisance_quantile(
                Y_train, X_train, A_train, 
                Y_hat_0_train, Y_hat_1_train, pi_train, qs_0, qs_1, X_test, **kwargs)

            for i in range(n_est):
                tau_arr[i][test_index] = res[i][0][None,:]
                density_arr[i][test_index] = res[i][1][None,:]
                nu_arr[i][test_index] = res[i][2].T
                

    if estimand=='mean':
        return pi_arr, Y_hat_0_arr, Y_hat_1_arr
    elif estimand=='quantile':
        return pi_arr, tau_arr, density_arr, nu_arr
    

