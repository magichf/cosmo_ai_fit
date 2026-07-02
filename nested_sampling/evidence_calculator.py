"""Bayesian evidence calculation using nested sampling

This module provides several methods for computing the log-evidence (marginal likelihood)
needed for model comparison via Bayes factors.
"""

import numpy as np
import warnings
from scipy import special
from scipy.stats import gaussian_kde

try:
    import dynesty
    HAS_DYNESTY = True
except ImportError:
    HAS_DYNESTY = False
    warnings.warn("dynesty not installed. Install with: pip install dynesty")


class EvidenceCalculator:
    """Compute Bayesian evidence using multiple methods"""
    
    @staticmethod
    def harmonic_mean_evidence(log_prob_samples, max_samples=None):
        """
        Harmonic mean estimator for evidence
        
        WARNING: This estimator is biased high and should not be used for
        publication-quality results. Use only for rough estimates.
        
        Z ≈ exp(log_L_max) / <1/L>
        
        Parameters:
        -----------
        log_prob_samples : array
            Log-probability values from posterior samples
        max_samples : int
            Maximum number of samples to use (for efficiency)
        
        Returns:
        --------
        logZ : float
            Log-evidence (biased)
        """
        if max_samples is not None:
            # Subsample for efficiency
            idx = np.random.choice(len(log_prob_samples), 
                                   size=min(max_samples, len(log_prob_samples)),
                                   replace=False)
            log_prob_samples = log_prob_samples[idx]
        
        # Shift for numerical stability
        log_prob_max = np.max(log_prob_samples)
        
        # Harmonic mean
        exp_prob = np.exp(log_prob_samples - log_prob_max)
        Z_inv = np.mean(1.0 / exp_prob)
        
        logZ = log_prob_max - np.log(Z_inv)
        
        return logZ
    
    @staticmethod
    def laplace_evidence(samples, log_prob_samples):
        """
        Laplace approximation for evidence
        
        Assumes posterior is Gaussian at the mode.
        Better than harmonic mean, but still biased.
        
        log Z ≈ log L(θ_ML) + k/2 * log(2π) + 1/2 * log|Σ_post|
        
        Parameters:
        -----------
        samples : array [N x k]
            Posterior samples
        log_prob_samples : array [N]
            Log-probability at each sample
        
        Returns:
        --------
        logZ : float
            Log-evidence
        """
        # Maximum likelihood point
        idx_max = np.argmax(log_prob_samples)
        log_L_max = log_prob_samples[idx_max]
        
        # Posterior covariance
        k = samples.shape[1]  # Dimensionality
        cov = np.cov(samples.T)
        
        # Handle singular covariance
        try:
            det_cov = np.linalg.det(cov)
            if det_cov <= 0:
                logZ = log_L_max
            else:
                log_det_cov = np.log(det_cov)
                logZ = log_L_max + 0.5 * k * np.log(2 * np.pi) + 0.5 * log_det_cov
        except:
            logZ = log_L_max
        
        return logZ
    
    @staticmethod
    def importance_sampling_evidence(samples, log_prob_samples, 
                                    prior_volume=1.0, niter=100):
        """
        Importance sampling estimator
        
        Z = ∫ L(θ) π(θ) dθ ≈ (V_prior) * mean(L_i * w_i)
        
        where w_i is importance weight from reference distribution.
        
        Parameters:
        -----------
        samples : array [N x k]
        log_prob_samples : array [N]
        prior_volume : float
            Rough prior volume estimate
        niter : int
            Number of bootstrap iterations
        
        Returns:
        --------
        logZ : float
            Log-evidence
        logZ_err : float
            Uncertainty in log(Z)
        """
        k = samples.shape[1]
        n_samples = len(samples)
        
        # Reference distribution: KDE on posterior
        try:
            kde = gaussian_kde(samples.T, bw_method=0.1)
        except:
            # Fallback if KDE fails
            return EvidenceCalculator.laplace_evidence(samples, log_prob_samples), 0.0
        
        logZ_samples = []
        
        for _ in range(niter):
            # Resample
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            
            # Importance weights: L / q(θ)
            q_vals = kde.logpdf(samples[idx].T)  # Log of reference
            L_vals = np.exp(log_prob_samples[idx])
            
            weights = L_vals / np.exp(q_vals + 20)  # Shift for stability
            
            # Evidence estimate
            logZ_iter = np.log(np.mean(weights)) + q_vals.mean() + 20 - np.log(prior_volume)
            logZ_samples.append(logZ_iter)
        
        logZ = np.mean(logZ_samples)
        logZ_err = np.std(logZ_samples)
        
        return logZ, logZ_err
    
    @staticmethod
    def use_nested_sampling(log_prob_fn, prior_fn, ndim, 
                           nlive=1000, bound='multi'):
        """
        Use dynesty for proper nested sampling
        
        This is the recommended method for publication-quality evidence.
        
        Parameters:
        -----------
        log_prob_fn : callable
            Log-probability function: log_prob_fn(theta) -> float
        prior_fn : callable
            Prior transform: prior_fn(u) -> theta, where u ~ U[0,1]^k
        ndim : int
            Number of dimensions
        nlive : int
            Number of live points
        bound : str
            Bounding method: 'multi', 'balls', 'cubes'
        
        Returns:
        --------
        sampler : dynesty.NestedSampler
            Full sampler object
        results : dynesty.utils.Result
            Results object with evidence, samples, etc.
        """
        if not HAS_DYNESTY:
            raise ImportError("dynesty not installed. Install with: pip install dynesty")
        
        print(f"\nRunning nested sampling (nlive={nlive}, bound={bound})...")
        
        # Create sampler
        sampler = dynesty.NestedSampler(
            log_prob_fn,
            prior_fn,
            ndim,
            nlive=nlive,
            bound=bound,
            sample='auto',
            ndim_init=ndim if ndim < 50 else 50
        )
        
        # Run sampling
        sampler.run_nested(print_progress=True)
        
        # Get results
        results = sampler.results
        
        logZ = results['logz'][-1]
        logZ_err = results['logzerr'][-1]
        
        print(f"\nNested Sampling Results:")
        print(f"  log(Z) = {logZ:.4f} ± {logZ_err:.4f}")
        print(f"  Samples: {len(results['samples'])}")
        
        return sampler, results


def compute_bayes_factor(logZ_model_1, logZ_model_2, logZ_err_1=0, logZ_err_2=0):
    """
    Compute Bayes factor from two evidences
    
    B = exp(logZ_1 - logZ_2)
    
    Parameters:
    -----------
    logZ_model_1 : float
        Log-evidence for model 1 (test model)
    logZ_model_2 : float
        Log-evidence for model 2 (reference/null model)
    logZ_err_1 : float
        Uncertainty in logZ_1
    logZ_err_2 : float
        Uncertainty in logZ_2
    
    Returns:
    --------
    BF : float
        Bayes factor
    BF_err : float
        Uncertainty in Bayes factor (propagated)
    interpretation : str
        Interpretation of BF
    """
    log_BF = logZ_model_1 - logZ_model_2
    BF = np.exp(log_BF)
    
    # Error propagation: Δ(log BF) = sqrt(Δ(logZ_1)² + Δ(logZ_2)²)
    log_BF_err = np.sqrt(logZ_err_1**2 + logZ_err_2**2)
    BF_err = BF * log_BF_err
    
    # Interpretation
    if BF > 150:
        interp = "Very Strong evidence for model 1"
    elif BF > 15:
        interp = "Strong evidence for model 1"
    elif BF > 3:
        interp = "Moderate evidence for model 1"
    elif BF > 1:
        interp = "Weak evidence for model 1"
    elif BF > 1/3:
        interp = "Inconclusive"
    elif BF > 1/15:
        interp = "Weak evidence for model 2"
    elif BF > 1/150:
        interp = "Strong evidence for model 2"
    else:
        interp = "Very Strong evidence for model 2"
    
    return BF, BF_err, interp
