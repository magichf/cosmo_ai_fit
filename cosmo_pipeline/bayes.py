"""Bayesian model comparison and evidence calculation"""
import numpy as np

def bayes_factor(logZ_model, logZ_lcdm):
    """
    Compute Bayes factor for model comparison
    
    B = exp(logZ_model - logZ_lcdm)
    
    Interpretation:
    - B > 150: Strong evidence for model
    - 15 < B < 150: Moderate evidence
    - 3 < B < 15: Weak evidence
    - B ~ 1: Data inconclusive
    
    Parameters:
    -----------
    logZ_model : float
        Log-evidence for three-scalar model
    logZ_lcdm : float
        Log-evidence for ΛCDM baseline
    
    Returns:
    --------
    BF : float
        Bayes factor
    """
    log_BF = logZ_model - logZ_lcdm
    BF = np.exp(log_BF)
    
    # Interpretation
    if BF > 150:
        evidence_str = "Very Strong"
    elif BF > 15:
        evidence_str = "Strong"
    elif BF > 3:
        evidence_str = "Moderate"
    elif BF > 1:
        evidence_str = "Weak"
    else:
        evidence_str = "Evidence against model"
    
    print(f"\nBayes Factor Analysis:")
    print("-" * 50)
    print(f"log(B) = {log_BF:.4f}")
    print(f"B = {BF:.4f}")
    print(f"Interpretation: {evidence_str}")
    
    return BF


def compute_evidence_from_samples(samples, log_prob_values, method="harmonic_mean"):
    """
    Estimate evidence from posterior samples
    
    Methods:
    - harmonic_mean: Simple but biased (not recommended for publication)
    - laplace: Gaussian approximation at maximum
    
    For publication use nested sampling (PyMultiNest, dynesty, etc.)
    
    Parameters:
    -----------
    samples : array [nsamples x ndim]
    log_prob_values : array [nsamples]
        Log-probability at each sample
    method : str
    
    Returns:
    --------
    logZ : float
        Log-evidence
    """
    if method == "harmonic_mean":
        # Harmonic mean estimator (simple, biased high)
        exp_logL = np.exp(log_prob_values - np.max(log_prob_values))
        Z_inv = np.mean(1.0 / exp_logL)
        logZ = np.max(log_prob_values) - np.log(Z_inv)
    
    elif method == "laplace":
        # Laplace approximation
        logL_max = np.max(log_prob_values)
        D = samples.shape[1]
        cov = np.cov(samples.T)
        det_cov = np.linalg.det(cov)
        if det_cov > 0:
            logZ = logL_max + 0.5 * D * np.log(2 * np.pi) + 0.5 * np.log(det_cov)
        else:
            logZ = logL_max
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return logZ


def model_comparison_summary(models_dict):
    """
    Summary of model comparison results
    
    Parameters:
    -----------
    models_dict : dict
        {"model_name": {"logZ": float, "nparams": int}, ...}
    
    Returns:
    --------
    Prints formatted comparison table
    """
    print("\nMODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'log(Z)':<15} {'# Params':<12} {'AIC':<15}")
    print("-" * 70)
    
    for name, info in models_dict.items():
        logZ = info.get('logZ', 0)
        nparams = info.get('nparams', 0)
        aic = -2 * logZ + 2 * nparams
        
        print(f"{name:<20} {logZ:<15.4f} {nparams:<12} {aic:<15.4f}")
    
    # Compute relative evidences
    logZ_ref = list(models_dict.values())[0]['logZ']
    
    print("\nRelative to first model:")
    for name, info in models_dict.items():
        logZ = info.get('logZ', 0)
        log_BF = logZ - logZ_ref
        BF = np.exp(log_BF)
        print(f"{name}: B = {BF:.2f}")
