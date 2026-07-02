"""Likelihood function implementation"""
import numpy as np

def run_class(theta):
    """
    Mock CLASS integration: compute Cl and Pk for given parameters
    
    In full implementation, this would interface with CLASS/classy
    to solve Boltzmann equations with modified three-scalar background
    
    Parameters:
    -----------
    theta : dict
        {"Gamma_phi_chi": float,
         "Gamma_phi_H": float,
         "Omega_m": float,
         "H0": float}
    
    Returns:
    --------
    Cl_th : array
        Theoretical CMB power spectrum
    Pk_th : array
        Theoretical matter power spectrum
    """
    # Extract parameters
    Gamma_chi = theta.get("Gamma_phi_chi", 0.001)
    Gamma_H = theta.get("Gamma_phi_H", 0.002)
    Omega_m = theta.get("Omega_m", 0.3)
    H0 = theta.get("H0", 67.0)
    
    # Mock modification to spectra based on decay rates
    # Real implementation would solve full Boltzmann equations
    
    n_ell = 100
    Cl_th = 2000 * np.exp(-np.arange(n_ell) / 50)
    
    # Modification from scalar decay
    decay_factor = 1.0 + 0.05 * (Gamma_chi + Gamma_H) / 0.003
    Cl_th = Cl_th * decay_factor
    
    # Omega_m dependence (matter affects CMB indirectly)
    Cl_th = Cl_th * (Omega_m / 0.3) ** 0.1
    
    # H0 dependence (angular diameter distance)
    Cl_th = Cl_th * (67.0 / H0) ** 0.05
    
    n_k = 50
    k_array = np.logspace(-3, -1, n_k)
    Pk_th = 1000 * (k_array / 0.01) ** 0.96
    
    # Modification from dark matter generation
    Pk_th = Pk_th * (1.0 + 0.1 * Gamma_chi / 0.001)
    
    # Matter density dependence (linear growth)
    Pk_th = Pk_th * (Omega_m / 0.3) ** 1.5
    
    return Cl_th, Pk_th


def loglike(theta, data):
    """
    Calculate log-likelihood from CMB + LSS data
    
    Log L = -0.5 * chi2
    chi2 = (Cl_th - Cl_obs)^T Cov^-1 (Cl_th - Cl_obs)
           + (Pk_th - Pk_obs)^T Cov^-1 (Pk_th - Pk_obs)
    
    Parameters:
    -----------
    theta : dict
        Model parameters
    data : dict
        Observational data from data_loader
    
    Returns:
    --------
    log_likelihood : float
    """
    # Get theoretical spectra
    Cl_th, Pk_th = run_class(theta)
    
    # Extract observed data
    Cl_obs = data["planck_cl"]
    Pk_obs = data["desi_pk"]
    Cov_inv_cmb = data["planck_cov"]
    Cov_inv_lss = data["desi_cov"]
    
    # Check dimensions
    if len(Cl_th) != len(Cl_obs):
        return -np.inf
    if len(Pk_th) != len(Pk_obs):
        return -np.inf
    
    # CMB chi-squared
    dCl = Cl_th - Cl_obs
    chi2_cmb = dCl.T @ Cov_inv_cmb @ dCl
    
    # LSS chi-squared
    dPk = Pk_th - Pk_obs
    chi2_lss = dPk.T @ Cov_inv_lss @ dPk
    
    # Total chi-squared
    chi2_total = chi2_cmb + chi2_lss
    
    # Log-likelihood
    log_L = -0.5 * chi2_total
    
    return log_L


def log_prior(theta):
    """
    Prior probability for parameters
    
    Uniform priors on all parameters within physical ranges
    
    Parameters:
    -----------
    theta : dict
        Model parameters
    
    Returns:
    --------
    log_prior : float
        0 if within prior range, -inf otherwise
    """
    Gamma_chi = theta.get("Gamma_phi_chi", -1)
    Gamma_H = theta.get("Gamma_phi_H", -1)
    Omega_m = theta.get("Omega_m", -1)
    H0 = theta.get("H0", -1)
    
    # Check bounds
    if not (0.0 < Gamma_chi < 0.01):
        return -np.inf
    if not (0.0 < Gamma_H < 0.01):
        return -np.inf
    if not (0.1 < Omega_m < 0.5):
        return -np.inf
    if not (60 < H0 < 75):
        return -np.inf
    
    return 0.0  # Log of uniform prior


def log_probability(theta, data):
    """
    Total log-probability = log-prior + log-likelihood
    
    Parameters:
    -----------
    theta : dict
        Model parameters
    data : dict
        Observational data
    
    Returns:
    --------
    log_prob : float
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = loglike(theta, data)
    return lp + ll
